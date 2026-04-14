"""
Cosmos-WAM Policy for RoboTwin Evaluation.

This policy wraps the Cosmos-WAM model for RoboTwin benchmark evaluation.
"""

import hashlib
import logging
import os
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torchvision.transforms as T
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from PIL import Image

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "cosmos_wam"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

# Add cosmos-predict2.5 to path (from environment variable or config)
COSMOS_PREDICT2_PATH = os.environ.get("COSMOS_PREDICT2_PATH", "/home/jwhe/linyihan/cosmos-predict2.5")
if COSMOS_PREDICT2_PATH and Path(COSMOS_PREDICT2_PATH).exists():
    if COSMOS_PREDICT2_PATH not in sys.path:
        sys.path.insert(0, COSMOS_PREDICT2_PATH)
        print(f"[CosmosWAM] Added cosmos_predict2 path: {COSMOS_PREDICT2_PATH}", file=sys.stderr)

from cosmos_wam.datasets.lerobot.processors.fastwam_processor import FastWAMProcessor
from cosmos_wam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json
from cosmos_wam.datasets.dataset_utils import ResizeSmallestSideAspectPreserving, CenterCrop, Normalize
from cosmos_wam.models.cosmos_wam import CosmosWAM
from cosmos_wam.models.dit_wrapper import MiniTrainDIT, SACConfig
from cosmos_wam.models.vae_wrapper import Wan2pt1VAEInterface
from cosmos_wam.models.action_head_mot import ActionExpert
from cosmos_wam.models.mot import MoT

logger = logging.getLogger(__name__)


def _is_none_like(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().lower() in {"", "none", "null"}
    return False


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "y"}:
            return True
        if lowered in {"0", "false", "no", "n"}:
            return False
    raise ValueError(f"Cannot parse bool value: {value}")


def _parse_optional_int(value: Any) -> Optional[int]:
    if _is_none_like(value):
        return None
    return int(value)


def _parse_optional_float(value: Any) -> Optional[float]:
    if _is_none_like(value):
        return None
    return float(value)


def _mixed_precision_to_model_dtype(mixed_precision: str) -> torch.dtype:
    key = str(mixed_precision).strip().lower()
    if key in {"no", "fp32"}:
        return torch.float32
    if key == "fp16":
        return torch.float16
    return torch.bfloat16


def _compose_sim_cfg(
    sim_cfg_path: Optional[str],
    sim_task: Optional[str],
) -> DictConfig:
    """Compose simulation config using Hydra."""
    configs_root = (PROJECT_ROOT / "configs").resolve()
    
    config_name = "sim_robotwin.yaml"
    if not _is_none_like(sim_cfg_path):
        cfg_path = Path(str(sim_cfg_path)).expanduser().resolve()
        try:
            relative = cfg_path.relative_to(configs_root)
            config_name = relative.as_posix()
        except ValueError:
            pass
    
    overrides = []
    if not _is_none_like(sim_task):
        overrides.append(f"task={str(sim_task)}")
    
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    
    with initialize_config_dir(version_base="1.3", config_dir=str(configs_root)):
        cfg = compose(config_name=config_name, overrides=overrides)
    return cfg


def _resolve_dataset_stats_path(dataset_stats_path: Optional[str]) -> Path:
    """Resolve dataset stats path."""
    if _is_none_like(dataset_stats_path):
        raise FileNotFoundError(
            "`dataset_stats_path` is required. "
            "Please pass it from eval entrypoint overrides."
        )
    resolved = Path(str(dataset_stats_path)).expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Dataset stats path not found: {resolved}")
    return resolved


def _get_text_embedding_cache(prompt: str, cache_dir: Path, context_len: int = 512) -> tuple[torch.Tensor, torch.Tensor]:
    """Load precomputed text embedding from cache."""
    hashed = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    cache_path = cache_dir / f"{hashed}.t5_len{context_len}.pt"
    
    # Search recursively in subdirectories if not found directly
    if not cache_path.exists():
        for root, dirs, files in os.walk(cache_dir):
            if f"{hashed}.t5_len{context_len}.pt" in files:
                cache_path = Path(root) / f"{hashed}.t5_len{context_len}.pt"
                break
    if not cache_path.exists():
        raise FileNotFoundError(f"Missing text embedding cache for prompt: {prompt[:50]}...")
    
    payload = torch.load(cache_path, map_location="cpu")
    return payload["context"], payload["mask"].bool()


class OnlineTextEncoder:
    """Online text encoder using Cosmos-Reason1-7B (Qwen2.5-VL-7B).
    
    This encoder computes text embeddings on-the-fly without precomputed cache.
    It can run on a different GPU than the main inference model.
    """
    
    NUM_EMBEDDING_PADDING_TOKENS = 512
    FULL_CONCAT_DIM = 28 * 3584  # 100352
    CONTEXT_LEN = 512
    
    def __init__(self, model_path: str, device: str = "cuda:0"):
        """Initialize the online text encoder.
        
        Args:
            model_path: Path to Cosmos-Reason1-7B model
            device: Device to run the encoder on (e.g., "cuda:0" or "cuda:1")
        """
        self.device = device
        logger.info(f"Loading online text encoder from: {model_path}")
        logger.info(f"Text encoder device: {device}")
        
        try:
            from transformers import AutoTokenizer, Qwen2_5_VLForConditionalGeneration
        except ImportError:
            raise ImportError("Please install transformers: pip install transformers")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map=device,
            trust_remote_code=True,
        )
        self.model.eval()
        self.pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        logger.info("Online text encoder loaded successfully")
    
    @torch.no_grad()
    def encode(self, task_text: str) -> torch.Tensor:
        """Encode a single task text to embedding.
        
        Args:
            task_text: The task instruction text
            
        Returns:
            context: [CONTEXT_LEN, FULL_CONCAT_DIM] tensor (bfloat16)
        """
        # Build chat template (same as precompute_text_embeddings.py)
        conversations = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "You are a helpful assistant who will provide prompts to an image generator.",
                    }
                ],
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": task_text}],
            },
        ]
        
        input_ids = self.tokenizer.apply_chat_template(
            conversations,
            tokenize=True,
            add_generation_prompt=False,
            return_tensors=None,
        )
        
        # Pad or truncate to CONTEXT_LEN
        if len(input_ids) < self.NUM_EMBEDDING_PADDING_TOKENS:
            input_ids = input_ids + [self.pad_id] * (self.NUM_EMBEDDING_PADDING_TOKENS - len(input_ids))
        else:
            input_ids = input_ids[:self.NUM_EMBEDDING_PADDING_TOKENS]
        
        input_ids_tensor = torch.LongTensor(input_ids).unsqueeze(0).to(self.device)
        
        # Get hidden states
        outputs = self.model(
            input_ids=input_ids_tensor,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden_states = outputs.hidden_states  # tuple of (num_layers+1) tensors
        
        # 28 层 mean-normalize 后在最后一维 concat -> (1, 512, 28*3584 = 100352)
        normalized = []
        for layer_idx in range(1, len(hidden_states)):
            h = hidden_states[layer_idx].float()
            h = (h - h.mean(dim=-1, keepdim=True)) / (h.std(dim=-1, keepdim=True) + 1e-8)
            normalized.append(h)
        
        text_emb = torch.cat(normalized, dim=-1)  # (1, 512, 100352)
        text_emb = text_emb.squeeze(0).to(torch.bfloat16)  # (512, 100352)
        
        assert text_emb.shape[-1] == self.FULL_CONCAT_DIM, \
            f"Unexpected Reason1 dim {text_emb.shape[-1]}, expected {self.FULL_CONCAT_DIM}"
        
        return text_emb


class CosmosWAMRobotWinPolicy:
    """Cosmos-WAM policy for RoboTwin evaluation."""
    
    def __init__(
        self,
        model_cfg: DictConfig,
        processor_cfg: DictConfig,
        checkpoint_path: Path,
        dataset_stats_path: Path,
        text_embedding_cache_dir: Path,
        device: str,
        model_dtype: torch.dtype,
        action_horizon: int,
        replan_steps: int,
        num_inference_steps: int,
        sigma_shift: Optional[float],
        seed: Optional[int],
        num_video_frames: int,
        context_len: int = 512,
        fixed_text_embedding_path: Optional[Path] = None,
        task_name: Optional[str] = None,
        use_online_text_encoder: bool = False,
        online_text_encoder_path: Optional[str] = None,
        text_encoder_device: Optional[str] = None,
    ) -> None:
        """Initialize the policy.
        
        Args:
            use_online_text_encoder: Whether to compute text embeddings online
            online_text_encoder_path: Path to Cosmos-Reason1-7B model (required if use_online_text_encoder=True)
            text_encoder_device: Device for text encoder (e.g., "cuda:0"). If None, uses same device as model.
            model_cfg: Model configuration (from yaml)
            processor_cfg: Processor configuration for data preprocessing
            checkpoint_path: Path to model checkpoint
            dataset_stats_path: Path to dataset statistics JSON
            text_embedding_cache_dir: Directory containing precomputed text embeddings
            device: Device to run model on
            model_dtype: Model data type (fp32/fp16/bf16)
            action_horizon: Number of actions to predict
            replan_steps: Number of steps to execute before replanning
            num_inference_steps: Number of diffusion steps for inference
            sigma_shift: Sigma shift for sampling (optional)
            seed: Random seed (optional)
            num_video_frames: Number of video frames for context
            context_len: Text context length
            task_name: Task name for text embedding lookup (e.g., "beat_block_hammer")
        """
        self.device = device
        self.model_dtype = model_dtype
        self.action_horizon = int(action_horizon)
        self.replan_steps = int(max(1, min(replan_steps, action_horizon)))
        self.num_inference_steps = int(num_inference_steps)
        self.sigma_shift = sigma_shift
        self.seed = seed
        self.num_video_frames = int(num_video_frames)
        self.context_len = context_len
        self.text_embedding_cache_dir = Path(text_embedding_cache_dir)
        self.fixed_text_embedding_path = Path(fixed_text_embedding_path) if fixed_text_embedding_path else None
        self.task_name = task_name
        self.use_online_text_encoder = use_online_text_encoder
        
        # Initialize online text encoder if needed
        self.text_encoder = None
        if use_online_text_encoder:
            if online_text_encoder_path is None:
                raise ValueError("online_text_encoder_path is required when use_online_text_encoder=True")
            text_enc_device = text_encoder_device if text_encoder_device is not None else device
            self.text_encoder = OnlineTextEncoder(online_text_encoder_path, device=text_enc_device)
        
        # Build model components
        self._build_model(model_cfg)
        
        # Load checkpoint
        print("------------------------------------------")
        print("checkpoint_path: ",checkpoint_path)
        self._load_checkpoint(checkpoint_path)
        
        # Verify model weights are loaded (not random)
        # Check action head first layer weight
        action_head_first_weight = None
        for name, param in self.model.action_head.named_parameters():
            if 'weight' in name and action_head_first_weight is None:
                action_head_first_weight = param.data
                break
        
        if action_head_first_weight is not None:
            w_mean = action_head_first_weight.float().mean().item()
            w_std = action_head_first_weight.float().std().item()
            logger.info(f"Action head weight check: mean={w_mean:.6f}, std={w_std:.6f}")
            if abs(w_mean) < 0.01 and w_std < 0.01:
                logger.error("WARNING: Action head weights appear to be uninitialized (random)!")
                logger.error("Checkpoint may not have been loaded correctly!")
            else:
                logger.info("Action head weights appear to be loaded correctly.")
        
        # Setup processor for normalization
        self.processor: FastWAMProcessor = instantiate(processor_cfg).eval()
        dataset_stats = load_dataset_stats_from_json(str(dataset_stats_path))
        self.processor.set_normalizer_from_stats(dataset_stats)
        
        # Action queue for replanning
        self.pending_actions: deque[np.ndarray] = deque()
        self.episode_count = 0
        self.step_count = 0
        
        # Image transform: resize to 240x320 (training size)
        # Image transforms matching training (ResizeSmallestSideAspectPreserving + CenterCrop + Normalize)
        self.resize_transform = ResizeSmallestSideAspectPreserving(
            args={"img_w": 320, "img_h": 240}
        )
        self.crop_transform = CenterCrop(
            args={"img_w": 320, "img_h": 240}
        )
        self.normalize_transform = Normalize(
            args={"mean": 0.5, "std": 0.5}
        )
        
        logger.info(
            "Initialized CosmosWAMRobotWinPolicy | ckpt=%s | horizon=%d | replan=%d | device=%s",
            checkpoint_path,
            self.action_horizon,
            self.replan_steps,
            device,
        )
    
    def _build_model(self, model_cfg: DictConfig):
        """Build Cosmos-WAM model components."""
        # 1. Build VAE
        vae = Wan2pt1VAEInterface(
            vae_pth=model_cfg.vae_checkpoint,
            temporal_window=model_cfg.get("vae_temporal_window", 16),
            device=torch.device(self.device),
        )
        vae.eval()
        
        # 2. Build DiT
        dit = MiniTrainDIT(
            max_img_h=model_cfg.dit_config.max_img_h,
            max_img_w=model_cfg.dit_config.max_img_w,
            max_frames=model_cfg.dit_config.max_frames,
            in_channels=model_cfg.dit_config.in_channels,
            out_channels=model_cfg.dit_config.out_channels,
            patch_spatial=model_cfg.dit_config.patch_spatial,
            patch_temporal=model_cfg.dit_config.patch_temporal,
            model_channels=model_cfg.dit_config.model_channels,
            num_blocks=model_cfg.dit_config.num_blocks,
            num_heads=model_cfg.dit_config.num_heads,
            mlp_ratio=model_cfg.dit_config.get("mlp_ratio", 4.0),
            atten_backend=model_cfg.dit_config.get("atten_backend", "minimal_a2a"),
            crossattn_emb_channels=model_cfg.dit_config.crossattn_emb_channels,
            use_crossattn_projection=model_cfg.dit_config.get("use_crossattn_projection", False),
            crossattn_proj_in_channels=model_cfg.dit_config.get("crossattn_proj_in_channels", 1024),
            pos_emb_cls=model_cfg.dit_config.get("pos_emb_cls", "rope3d"),
            pos_emb_learnable=model_cfg.dit_config.get("pos_emb_learnable", False),
            rope_h_extrapolation_ratio=model_cfg.dit_config.get("rope_h_extrapolation_ratio", 1.0),
            rope_w_extrapolation_ratio=model_cfg.dit_config.get("rope_w_extrapolation_ratio", 1.0),
            rope_t_extrapolation_ratio=model_cfg.dit_config.get("rope_t_extrapolation_ratio", 1.0),
            use_wan_fp32_strategy=model_cfg.dit_config.get("use_wan_fp32_strategy", False),
            adaln_lora_dim=model_cfg.dit_config.get("adaln_lora_dim", 256),
            use_t_embedding_adaln_lora=model_cfg.dit_config.get("use_t_embedding_adaln_lora", True),
        )
        
        # 3. Build Action Head
        action_head = ActionExpert(
            action_dim=model_cfg.action_head.action_dim,
            hidden_dim=model_cfg.action_head.hidden_dim,
            num_layers=model_cfg.dit_config.num_blocks,
            num_heads=model_cfg.action_head.num_heads,
            text_dim=model_cfg.dit_config.get("crossattn_proj_in_channels", 100352),
            mlp_ratio=model_cfg.action_head.get("mlp_ratio", 4.0),
            backend=model_cfg.dit_config.get("atten_backend", "minimal_a2a"),
            use_wan_fp32_strategy=model_cfg.dit_config.get("use_wan_fp32_strategy", False),
            adaln_lora_dim=model_cfg.dit_config.get("adaln_lora_dim", 256),
        )

        # 4. Build MoT
        mot = MoT(
            mixtures={"video": dit, "action": action_head},
            mot_checkpoint_mixed_attn=model_cfg.get("mot_checkpoint_mixed_attn", True),
        )

        # 5. Build CosmosWAM
        self.model = CosmosWAM(
            mot=mot,
            vae=vae,
            lambda_action=model_cfg.get("lambda_action", 1.0),
            num_cond_frames=model_cfg.get("num_cond_frames", 1),
            actions_per_latent=model_cfg.action_head.get("actions_per_latent", 8),
        )
        self.model = self.model.to(self.device).to(self.model_dtype)
        self.model.eval()
    
    def _load_checkpoint(self, checkpoint_path: Path):
        """Load model checkpoint."""
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        
        if isinstance(ckpt, dict) and "model" in ckpt:
            state_dict = ckpt["model"]
        else:
            state_dict = ckpt
            print(f"[DEBUG] Checkpoint is state_dict with {len(state_dict)} keys", flush=True)
        
        # Check action_head weights specifically
        action_head_keys = [k for k in state_dict.keys() if 'action_head' in k and 'weight' in k]
        if action_head_keys:
            for k in action_head_keys[:3]:
                v = state_dict[k]
        
        # Load state dict (filtering out VAE if present, since we have our own)
        filtered_state = {k: v for k, v in state_dict.items() if not k.startswith("vae.")}
        
        # Convert dtype if needed
        if self.model_dtype == torch.bfloat16:
            filtered_state = {k: v.to(torch.bfloat16) if v.dtype == torch.float32 else v 
                             for k, v in filtered_state.items()}
        
        missing, unexpected = self.model.load_state_dict(filtered_state, strict=False)
        
        if missing:
            logger.warning(f"Missing keys ({len(missing)}): {missing[:5]}")
        if unexpected:
            logger.warning(f"Unexpected keys ({len(unexpected)}): {unexpected[:5]}")
        
        logger.info(f"Loaded checkpoint with {len(filtered_state)} keys")
    
    def _normalize_state(self, state: np.ndarray) -> torch.Tensor:
        """Normalize proprioception state."""
        state_meta = self.processor.shape_meta["state"]
        if len(state_meta) != 1:
            raise ValueError("Expected exactly one merged state key in shape_meta['state'].")
        state_key = state_meta[0]["key"]
        
        state_batch = {"state": {state_key: torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)}}
        state_batch = self.processor.action_state_transform(state_batch)
        state_batch = self.processor.normalizer.forward(state_batch)
        return state_batch["state"][state_key]
    
    def _denormalize_action(self, action: torch.Tensor) -> np.ndarray:
        """Denormalize predicted action."""
        if action.ndim == 2:
            action = action.unsqueeze(0)
        if action.ndim != 3:
            raise ValueError(f"Expected action tensor [B,T,D], got {tuple(action.shape)}")
        
        action_meta = self.processor.shape_meta["action"]
        if len(action_meta) != 1:
            raise ValueError("Expected exactly one merged action key in shape_meta['action'].")
        
        action_key = action_meta[0]["key"]
        normalizer = self.processor.normalizer.normalizers["action"][action_key]
        denorm = normalizer.backward(action.to(dtype=torch.float32, device="cpu"))
        
        return denorm.numpy()
    
    def _build_image_tensor(self, observation: Dict[str, Any]) -> torch.Tensor:
        """Build image tensor from observation.
        
        Uses only head_camera (240x320) as trained.
        Matches training preprocessing exactly.
        """
        obs_data = observation["observation"]
        head_image = obs_data["head_camera"]["rgb"]  # [H, W, 3] numpy array
        
        # Handle different input ranges
        # RoboTwin may return float [0, 1] or uint8 [0, 255]
        if head_image.dtype == np.float32 or head_image.dtype == np.float64:
            if head_image.max() <= 1.0:
                # Already in [0, 1], convert to uint8
                head_image = (head_image * 255).astype(np.uint8)
            else:
                # Assume in [0, 255] but stored as float
                head_image = head_image.astype(np.uint8)
        else:
            # uint8 or other integer type
            head_image = head_image.astype(np.uint8)
        
        # Convert to PIL then tensor [0, 255] -> [0, 1]
        from PIL import Image
        pil_image = Image.fromarray(head_image, mode="RGB")
        image_tensor = T.ToTensor()(pil_image)  # [3, H, W]
        
        # Apply same transforms as training: resize -> crop -> normalize
        image_tensor = self.resize_transform(image_tensor)  # ResizeSmallestSideAspectPreserving
        image_tensor = self.crop_transform(image_tensor)    # CenterCrop to 240x320
        image_tensor = self.normalize_transform(image_tensor)  # Normalize to [-1, 1]
        
        # Move to device and add dimensions: [3, 240, 320] -> [1, 3, 1, 240, 320]
        image_tensor = image_tensor.to(device=self.device, dtype=self.model_dtype)
        image_tensor = image_tensor.unsqueeze(0).unsqueeze(2)  # [B=1, C=3, T=1, H=240, W=320]
        
        return image_tensor
    
    def _get_text_context(self, instruction: str) -> torch.Tensor:
        """Get text context from cache or online encoder.
        
        Matches training format: "A video recorded from a robot's point of view executing the following instruction: {task}"
        where {task} is the detailed instruction from tasks.jsonl (not task_name).
        """
        # Use online text encoder if enabled
        if self.use_online_text_encoder and self.text_encoder is not None:
            logger.info(f"Computing text embedding online for: {instruction[:50]}...")
            context = self.text_encoder.encode(instruction)
            context = context.to(device=self.device, dtype=self.model_dtype)
            return context.unsqueeze(0)  # [1, 512, 100352]
        
        # If fixed path is provided, use it directly (skip hash lookup)
        if self.fixed_text_embedding_path is not None:
            logger.info(f"Using fixed text embedding: {self.fixed_text_embedding_path}")
            payload = torch.load(self.fixed_text_embedding_path, map_location="cpu")
            context = payload["context"]
            context = context.unsqueeze(0).to(device=self.device, dtype=self.model_dtype)
            return context
        
        # Use the instruction directly (detailed description from tasks.jsonl)
        # This matches how precompute_text_embeddings.py works
        prompt = f"A video recorded from a robot's point of view executing the following instruction: {instruction}"
        
        logger.info(f"Looking for text embedding with prompt: {prompt[:80]}...")
        
        try:
            context, mask = _get_text_embedding_cache(
                prompt, 
                self.text_embedding_cache_dir, 
                self.context_len
            )
        except FileNotFoundError as e:
            # Log available files for debugging
            logger.error(f"Text embedding not found. Prompt hash: {hashlib.sha256(prompt.encode('utf-8')).hexdigest()[:16]}...")
            cache_files = list(self.text_embedding_cache_dir.rglob("*.t5_len*.pt"))
            logger.error(f"Available cache files ({len(cache_files)}): {[f.name[:20] + '...' for f in cache_files[:5]]}")
            raise e
        
        # Move to device and add batch dimension
        context = context.unsqueeze(0).to(device=self.device, dtype=self.model_dtype)
        return context
    
    def _infer_action_chunk(self, observation: Dict[str, Any], instruction: str) -> np.ndarray:
        """Infer action chunk from observation and instruction."""
        # Build inputs
        image_tensor = self._build_image_tensor(observation)  # [1, 3, 240, 320]
        context = self._get_text_context(instruction)  # [1, L, D]
        
        # Run inference
        with torch.no_grad():
            action = self.model.infer_action(
                first_frame_pixels=image_tensor,
                action_horizon=self.action_horizon,
                context=context,
                num_inference_steps=self.num_inference_steps,
            )    
        action_chunk = self._denormalize_action(action)[0]  # [T, D]      
        return action_chunk
    
    def _fill_action_queue(self, observation: Dict[str, Any], instruction: str) -> None:
        """Fill action queue with new predictions."""
        action_chunk = self._infer_action_chunk(observation=observation, instruction=instruction)
        n_exec = min(self.replan_steps, action_chunk.shape[0])
        for i in range(n_exec):
            self.pending_actions.append(np.asarray(action_chunk[i], dtype=np.float32))
    
    def should_request_observation(self) -> bool:
        """Check if new observation is needed (for replanning)."""
        return not self.pending_actions
    
    def step(self, task_env, observation: Optional[Dict[str, Any]]) -> None:
        """Execute one step in the environment.
        
        Args:
            task_env: RoboTwin task environment
            observation: Current observation (required when replanning)
        """
        if not self.pending_actions:
            if observation is None:
                raise ValueError(
                    "Observation is required when action queue is empty "
                    "(replan step for cosmos-wam)."
                )
            instruction = task_env.get_instruction()
            self._fill_action_queue(observation=observation, instruction=instruction)
        
        if not self.pending_actions:
            logger.warning("No action generated; skip current eval step.")
            return
        
        action = self.pending_actions.popleft()
        task_env.take_action(action, action_type="ee")
        self.step_count += 1
    
    def reset(self) -> None:
        """Reset policy for new episode."""
        self.pending_actions.clear()
        self.episode_count += 1
        self.step_count = 0


def encode_obs(observation: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Encode observation (identity function for compatibility)."""
    return observation


def get_model(usr_args: Dict[str, Any]):
    """Create and return policy model.
    
    This function is called by RoboTwin's eval_policy.py.
    """
    # Compose config
    sim_cfg_path = usr_args.get("sim_cfg_path")
    sim_task = usr_args.get("sim_task")
    cfg = _compose_sim_cfg(sim_cfg_path=sim_cfg_path, sim_task=sim_task)
    
    # Get checkpoint path
    checkpoint_path = usr_args.get("ckpt_setting")
    if _is_none_like(checkpoint_path):
        raise ValueError("`ckpt_setting` is required and must be a valid checkpoint path.")
    checkpoint_path = Path(str(checkpoint_path)).expanduser().resolve()
    
    # Get device
    device = str(usr_args.get("device") or cfg.EVALUATION.get("device") or "cuda")
    
    # Check if using online text encoder (dual GPU mode)
    use_online_text_encoder = _parse_bool(usr_args.get("use_online_text_encoder", False))
    
    # If CUDA_VISIBLE_DEVICES is set, we need to remap device IDs
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cuda_visible is not None and device.startswith("cuda") and not use_online_text_encoder:
        # Single GPU mode: use cuda:0 since that's the only GPU visible in this subprocess
        device = "cuda:0"
        logger.info(f"CUDA_VISIBLE_DEVICES={cuda_visible}, using cuda:0")
    elif cuda_visible is not None and use_online_text_encoder:
        # Dual GPU mode: CUDA_VISIBLE_DEVICES contains multiple GPUs
        # The device IDs should already be correct (cuda:0, cuda:1, etc.)
        logger.info(f"Dual GPU mode: CUDA_VISIBLE_DEVICES={cuda_visible}, using {device}")
    
    if device.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA is unavailable; fallback device to cpu.")
        device = "cpu"
    
    # Get dtype
    mixed_precision = str(usr_args.get("mixed_precision") or cfg.get("mixed_precision", "bf16"))
    model_dtype = _mixed_precision_to_model_dtype(mixed_precision)
    
    # Get dataset stats
    dataset_stats_path = _resolve_dataset_stats_path(
        dataset_stats_path=usr_args.get("dataset_stats_path"),
    )
    
    # Get text embedding cache dir (fallback if fixed path not set)
    text_embedding_cache_dir = usr_args.get("text_embedding_cache_dir")
    if _is_none_like(text_embedding_cache_dir):
        # Try to get from data.train config
        text_embedding_cache_dir = cfg.data.train.get("text_embedding_cache_dir")
    if not _is_none_like(text_embedding_cache_dir):
        text_embedding_cache_dir = Path(str(text_embedding_cache_dir)).expanduser().resolve()
    
    # Get fixed text embedding path (optional, for testing)
    fixed_text_embedding_path = usr_args.get("fixed_text_embedding_path")
    if _is_none_like(fixed_text_embedding_path):
        fixed_text_embedding_path = cfg.EVALUATION.get("fixed_text_embedding_path")
    if not _is_none_like(fixed_text_embedding_path):
        fixed_text_embedding_path = Path(str(fixed_text_embedding_path)).expanduser().resolve()
    
    # Get online text encoder settings
    use_online_text_encoder = _parse_bool(usr_args.get("use_online_text_encoder", False))
    online_text_encoder_path = usr_args.get("online_text_encoder_path")
    if not _is_none_like(online_text_encoder_path):
        online_text_encoder_path = str(Path(str(online_text_encoder_path)).expanduser().resolve())
    elif use_online_text_encoder:
        # Try to get from config
        online_text_encoder_path = cfg.EVALUATION.get("online_text_encoder_path")
    
    text_encoder_device = usr_args.get("text_encoder_device")
    if _is_none_like(text_encoder_device):
        text_encoder_device = cfg.EVALUATION.get("text_encoder_device")
    
    # Remap device IDs based on CUDA_VISIBLE_DEVICES for dual GPU mode
    if use_online_text_encoder and cuda_visible is not None:
        visible_gpus = [int(x.strip()) for x in cuda_visible.split(",")]
        
        def remap_device(dev_str):
            if dev_str is None or not dev_str.startswith("cuda:"):
                return dev_str
            gpu_id = int(dev_str.split(":")[1])
            if gpu_id in visible_gpus:
                new_id = visible_gpus.index(gpu_id)
                return f"cuda:{new_id}"
            return dev_str
        
        device = remap_device(device)
        text_encoder_device = remap_device(text_encoder_device)
        logger.info(f"Remapped devices: model={device}, text_encoder={text_encoder_device}")
    
    # Validate that at least one text embedding source is provided
    has_cache = not _is_none_like(text_embedding_cache_dir)
    has_fixed = not _is_none_like(fixed_text_embedding_path)
    has_online = use_online_text_encoder and not _is_none_like(online_text_encoder_path)
    
    if not (has_cache or has_fixed or has_online):
        raise ValueError(
            "Either `text_embedding_cache_dir`, `fixed_text_embedding_path`, or `use_online_text_encoder` with `online_text_encoder_path` must be provided."
        )
    
    # Get action horizon
    action_horizon = _parse_optional_int(usr_args.get("action_horizon"))
    if action_horizon is None:
        eval_horizon = _parse_optional_int(cfg.EVALUATION.get("action_horizon"))
        action_horizon = eval_horizon if eval_horizon is not None else 8
    if action_horizon <= 0:
        raise ValueError(f"`action_horizon` must be positive, got {action_horizon}")
    
    # Get replan steps
    replan_steps = _parse_optional_int(usr_args.get("replan_steps"))
    if replan_steps is None:
        replan_steps = int(cfg.EVALUATION.get("replan_steps", 4))
    
    # Get num inference steps
    num_inference_steps = _parse_optional_int(usr_args.get("num_inference_steps"))
    if num_inference_steps is None:
        num_inference_steps = int(cfg.EVALUATION.get("num_inference_steps", 20))
    
    # Get sigma shift
    sigma_shift = _parse_optional_float(usr_args.get("sigma_shift"))
    if sigma_shift is None:
        sigma_shift = _parse_optional_float(cfg.EVALUATION.get("sigma_shift"))
    
    # Get seed
    seed = _parse_optional_int(usr_args.get("seed"))
    
    # Get context length
    context_len = int(usr_args.get("context_len", cfg.data.train.get("context_len", 512)))
    
    # Get task name for text embedding lookup
    task_name = usr_args.get("task_name")
    if _is_none_like(task_name):
        task_name = cfg.EVALUATION.get("task_name")
    
    # Create policy
    policy = CosmosWAMRobotWinPolicy(
        model_cfg=cfg.model,
        processor_cfg=cfg.data.train.processor,
        checkpoint_path=checkpoint_path,
        dataset_stats_path=dataset_stats_path,
        text_embedding_cache_dir=text_embedding_cache_dir if not _is_none_like(text_embedding_cache_dir) else Path("."),
        device=device,
        model_dtype=model_dtype,
        action_horizon=action_horizon,
        replan_steps=replan_steps,
        num_inference_steps=num_inference_steps,
        sigma_shift=sigma_shift,
        seed=seed,
        num_video_frames=(int(cfg.data.train.num_frames) - 1) // int(cfg.data.train.action_video_freq_ratio) + 1,
        context_len=context_len,
        fixed_text_embedding_path=fixed_text_embedding_path,
        task_name=task_name,
        use_online_text_encoder=use_online_text_encoder,
        online_text_encoder_path=online_text_encoder_path,
        text_encoder_device=text_encoder_device,
    )
    
    return policy


def eval(TASK_ENV, model, observation: Optional[Dict[str, Any]]):
    """Evaluation step (called by RoboTwin)."""
    obs = encode_obs(observation)
    model.step(TASK_ENV, obs)


def reset_model(model):
    """Reset model (called by RoboTwin)."""
    model.reset()
