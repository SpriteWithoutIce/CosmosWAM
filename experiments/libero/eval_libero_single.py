"""
Cosmos-WAM LIBERO single-task evaluation entrypoint.

Based on FastWAM's LIBERO evaluation but adapted for Cosmos-WAM model interface.
"""

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Optional

# Add LIBERO to path before other imports
LIBERO_PATH = os.environ.get("LIBERO_PATH", "/home/jwhe/linyihan/LIBERO")
if LIBERO_PATH not in sys.path:
    sys.path.insert(0, LIBERO_PATH)

import hydra
import numpy as np
import torch
from accelerate import PartialState
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from tqdm import tqdm

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from experiments.libero.libero_utils import (
    LIBERO_ENV_RESOLUTION,
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    invert_gripper_action,
    quat2axisangle,
    save_rollout_video,
)
from cosmos_wam.datasets.lerobot.processors.fastwam_processor import FastWAMProcessor
from cosmos_wam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json
from libero.libero import benchmark

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def _center_crop_resize(image: np.ndarray, width: int, height: int) -> np.ndarray:
    """Center crop and resize image."""
    pil_image = Image.fromarray(image)
    src_w, src_h = pil_image.size
    scale = max(width / src_w, height / src_h)
    resized = pil_image.resize((round(src_w * scale), round(src_h * scale)), resample=Image.BILINEAR)
    rw, rh = resized.size
    left = max((rw - width) // 2, 0)
    top = max((rh - height) // 2, 0)
    cropped = resized.crop((left, top, left + width, top + height))
    return np.asarray(cropped, dtype=np.uint8)


def _extract_sim_state(obs: dict) -> np.ndarray:
    """Build simulator state from current observation."""
    state = np.concatenate(
        (
            obs["robot0_eef_pos"],
            quat2axisangle(obs["robot0_eef_quat"]),
            obs["robot0_gripper_qpos"],
        )
    ).astype(np.float32)
    return state


def _normalize_proprio(proprio: np.ndarray, processor: FastWAMProcessor) -> torch.Tensor:
    """Normalize proprioception state."""
    state_meta = processor.shape_meta["state"]
    if len(state_meta) != 1:
        raise ValueError("LIBERO eval expects a single merged state key.")
    state_key = state_meta[0]["key"]

    state_batch = {"state": {state_key: torch.as_tensor(proprio, dtype=torch.float32).unsqueeze(0)}}
    state_batch = processor.action_state_transform(state_batch)
    state_batch = processor.normalizer.forward(state_batch)
    return state_batch["state"][state_key]


def _denormalize_action(action: torch.Tensor, processor: FastWAMProcessor) -> np.ndarray:
    """Denormalize predicted action."""
    if action.ndim == 2:
        action = action.unsqueeze(0)
    if action.ndim != 3:
        raise ValueError(f"Expected action tensor [B,T,D], got {tuple(action.shape)}")

    action_meta = processor.shape_meta["action"]
    if len(action_meta) != 1:
        raise ValueError("LIBERO eval expects a single merged action key.")

    action_key = action_meta[0]["key"]
    normalizer = processor.normalizer.normalizers["action"][action_key]
    action = action.to(dtype=torch.float32, device="cpu")
    denorm = normalizer.backward(action)
    return denorm.numpy()


def _obs_to_model_input(
    obs: dict,
    cfg: DictConfig,
    processor: FastWAMProcessor,
    width: int,
    height: int,
    device: str,
    dtype: torch.dtype,
):
    """Convert LIBERO observation to model input."""
    imgs = get_libero_image(obs)
    image_meta = processor.shape_meta["images"]
    
    if len(image_meta) < int(processor.num_output_cameras):
        raise ValueError(
            f"shape_meta.images has {len(image_meta)} entries, "
            f"but num_output_cameras={processor.num_output_cameras}."
        )

    def _meta_to_hw(meta: dict, camera_idx: int) -> tuple[int, int]:
        shape = meta["shape"]
        if len(shape) != 3:
            raise ValueError(f"shape_meta.images[{camera_idx}].shape must be [C,H,W], got {shape}")
        return int(shape[1]), int(shape[2])

    concatenation = cfg.data.train.get("concat_multi_camera", "horizontal")
    num_cameras = processor.num_output_cameras
    
    if num_cameras == 1:
        primary_h, primary_w = _meta_to_hw(image_meta[0], camera_idx=0)
        rgb = _center_crop_resize(imgs["image"], width=primary_w, height=primary_h)
    elif num_cameras == 2:
        primary_h, primary_w = _meta_to_hw(image_meta[0], camera_idx=0)
        wrist_h, wrist_w = _meta_to_hw(image_meta[1], camera_idx=1)
        primary = _center_crop_resize(imgs["image"], width=primary_w, height=primary_h)
        wrist = _center_crop_resize(imgs["wrist_image"], width=wrist_w, height=wrist_h)
        if concatenation == "horizontal":
            rgb = np.concatenate([primary, wrist], axis=1)
        elif concatenation == "vertical":
            rgb = np.concatenate([primary, wrist], axis=0)
        else:
            raise ValueError(f"Invalid concat_multi_camera: {concatenation}")
    else:
        raise ValueError(f"LIBERO eval supports num_output_cameras in [1, 2], got {num_cameras}.")

    # Convert to tensor and normalize to [-1, 1]
    x = torch.tensor(rgb).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=dtype)
    x = x * (2.0 / 255.0) - 1.0

    proprio = _normalize_proprio(_extract_sim_state(obs), processor)

    return x, proprio, imgs

DEFAULT_PROMPT = "A video recorded from a robot's point of view executing the following instruction: {task}"
def format_prompt(task: str, template: str) -> str:
    return template.format(task=task)

class OnlineTextEncoder:
    """Online text encoder for LIBERO (lazy initialization)."""
    
    NUM_EMBEDDING_PADDING_TOKENS = 128
    FULL_CONCAT_DIM = 28 * 3584  # 100352
    CONTEXT_LEN = 128
    
    def __init__(self, model_path: str, device: str = "cuda:0"):
        self.device = device
        self.model_path = model_path
        self._model = None
        self._tokenizer = None
        
    def _init_model(self):
        """Lazy initialize model."""
        if self._model is not None:
            return
        
        print(f"[OnlineTextEncoder] Loading model from: {self.model_path}")
        from transformers import AutoTokenizer, Qwen2_5_VLForConditionalGeneration
        
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            device_map=self.device,
            trust_remote_code=True,
        )
        self._model.eval()
        self._pad_id = self._tokenizer.pad_token_id or self._tokenizer.eos_token_id
        print("[OnlineTextEncoder] Model loaded")
    
    @torch.no_grad()
    def encode(self, task_text: str) -> torch.Tensor:
        """Encode task text to embedding."""
        self._init_model()
        
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
        
        input_ids = self._tokenizer.apply_chat_template(
            conversations,
            tokenize=True,
            add_generation_prompt=False,
            return_tensors=None,
        )
        
        # Pad or truncate
        if len(input_ids) < self.NUM_EMBEDDING_PADDING_TOKENS:
            input_ids = input_ids + [self._pad_id] * (self.NUM_EMBEDDING_PADDING_TOKENS - len(input_ids))
        else:
            input_ids = input_ids[:self.NUM_EMBEDDING_PADDING_TOKENS]
        
        input_ids_tensor = torch.LongTensor(input_ids).unsqueeze(0).to(self.device)
        
        outputs = self._model(
            input_ids=input_ids_tensor,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden_states = outputs.hidden_states
        
        # 28 layers mean-normalize and concat
        normalized = []
        for layer_idx in range(1, len(hidden_states)):
            h = hidden_states[layer_idx].float()
            h = (h - h.mean(dim=-1, keepdim=True)) / (h.std(dim=-1, keepdim=True) + 1e-8)
            normalized.append(h)
        
        text_emb = torch.cat(normalized, dim=-1)  # (1, 128, 100352)
        text_emb = text_emb.squeeze(0).to(torch.bfloat16)  # (128, 100352)
        
        return text_emb


def _get_text_embedding_direct(task_text: str, cache_dir: Path, context_len: int = 128, online_encoder=None):
    """Get text embedding using direct task text (no prompt template)."""
    # Use online encoder if available
    if online_encoder is not None:
        context = online_encoder.encode(task_text)
        mask = torch.ones(context_len, dtype=torch.bool)
        return context, mask
    
    # Otherwise use cache
    import hashlib
    hashed = hashlib.sha256(format_prompt(task_text, DEFAULT_PROMPT).encode("utf-8")).hexdigest()
    cache_path = cache_dir / f"{hashed}.t5_len{context_len}.pt"
    
    if not cache_path.exists():
        # Search recursively
        for root, dirs, files in os.walk(cache_dir):
            if f"{hashed}.t5_len{context_len}.pt" in files:
                cache_path = Path(root) / f"{hashed}.t5_len{context_len}.pt"
                break
    
    if not cache_path.exists():
        raise FileNotFoundError(f"Missing text embedding cache for task: {task_text[:50]}... (hash: {hashed[:16]}...)")
    
    payload = torch.load(cache_path, map_location="cpu")
    return payload["context"], payload["mask"].bool()


def _predict_action_chunk(
    obs: dict,
    task_description: str,
    model: torch.nn.Module,
    processor: FastWAMProcessor,
    cfg: DictConfig,
    action_horizon: int,
    input_w: int,
    input_h: int,
    device: str,
    online_encoder: OnlineTextEncoder = None,
) -> np.ndarray:
    """Predict action chunk from observation."""
    num_inference_steps = int(cfg.EVALUATION.get("num_inference_steps", 4))

    image, proprio, _ = _obs_to_model_input(
        obs, cfg=cfg, processor=processor, width=input_w, height=input_h,
        device=device, dtype=model.model_dtype if hasattr(model, 'model_dtype') else torch.float32
    )

    # Build text context from cache or online encoder
    cache_dir = Path(cfg.EVALUATION.text_embedding_cache_dir) if cfg.EVALUATION.get("text_embedding_cache_dir") else None
    context, _ = _get_text_embedding_direct(
        task_description, 
        cache_dir, 
        cfg.EVALUATION.get("context_len", 128),
        online_encoder=online_encoder
    )
    context = context.unsqueeze(0).to(device=device)

    # Add time dimension: [B,C,H,W] -> [B,C,T,H,W] where T=1
    if image.ndim == 4:
        image = image.unsqueeze(2)  # Add T dimension
    
    # Run inference
    with torch.no_grad():
        action = model.infer_action(
            first_frame_pixels=image,
            action_horizon=action_horizon,
            context=context,
            num_inference_steps=num_inference_steps,
        )

    # Denormalize action
    action = _denormalize_action(action, processor)[0]

    # Post-process gripper: flip back from dataloader format to env format
    action[..., -1] = action[..., -1] * 2 - 1
    action = invert_gripper_action(action)
    
    if bool(cfg.EVALUATION.get("binarize_gripper", False)):
        action[..., -1] = np.sign(action[..., -1])
    
    return action


def run_single_episode(
    env,
    initial_state,
    task_description: str,
    model: torch.nn.Module,
    processor: FastWAMProcessor,
    cfg: DictConfig,
    episode_idx: int,
    action_horizon: int,
    input_w: int,
    input_h: int,
    device: str,
    online_encoder: OnlineTextEncoder = None,
) -> tuple[bool, list]:
    """Run a single episode."""
    max_steps = {
        "libero_spatial": 500,
        "libero_object": 400,
        "libero_goal": 400,
        "libero_10": 700,
        "libero_90": 700,
    }.get(cfg.EVALUATION.task_suite_name, 400)
    
    replan_steps = int(cfg.EVALUATION.get("replan_steps", 5))
    num_steps_wait = int(cfg.EVALUATION.get("num_steps_wait", 5))

    env.reset()
    obs = env.set_init_state(initial_state)

    replay_images = []
    pending_actions = []
    success = False

    t = 0
    pbar = tqdm(total=max_steps + num_steps_wait, desc=f"Episode {episode_idx + 1}")
    
    while t < max_steps + num_steps_wait:
        pbar.update(1)
        
        if t < num_steps_wait:
            obs, _, done, _ = env.step(get_libero_dummy_action())
            t += 1
            continue

        if len(pending_actions) == 0:
            action_chunk = _predict_action_chunk(
                obs=obs,
                task_description=task_description,
                model=model,
                processor=processor,
                cfg=cfg,
                action_horizon=action_horizon,
                input_w=input_w,
                input_h=input_h,
                device=device,
                online_encoder=online_encoder,
            )
            pending_actions = action_chunk[:replan_steps].tolist()
            imgs = get_libero_image(obs)
            replay_images.append(imgs.copy())
        else:
            imgs = get_libero_image(obs)
            replay_images.append(imgs.copy())

        obs, _, done, _ = env.step(pending_actions.pop(0))
        
        if done:
            success = True
            break
        t += 1
    
    pbar.close()
    return success, replay_images


def run_single_task(
    task,
    initial_states,
    model: torch.nn.Module,
    processor: FastWAMProcessor,
    cfg: DictConfig,
    video_dir: Path,
    action_horizon: int,
    input_w: int,
    input_h: int,
    device: str,
    online_encoder: OnlineTextEncoder = None,
) -> dict:
    """Run evaluation for a single task."""
    env, task_description = get_libero_env(task, LIBERO_ENV_RESOLUTION, cfg.get("seed"))
    
    results = {
        "successes": 0,
        "failure_episodes": [],
        "success_episodes": [],
        "task_description": task_description,
    }

    for trial_idx in range(int(cfg.EVALUATION.num_trials)):
        success, replay_images = run_single_episode(
            env=env,
            initial_state=initial_states[trial_idx],
            task_description=task_description,
            model=model,
            processor=processor,
            cfg=cfg,
            episode_idx=trial_idx,
            action_horizon=action_horizon,
            input_w=input_w,
            input_h=input_h,
            device=device,
            online_encoder=online_encoder,
        )
        
        if success:
            results["successes"] += 1
            results["success_episodes"].append(trial_idx)
        else:
            results["failure_episodes"].append(trial_idx)

        save_rollout_video(
            video_dir,
            replay_images,
            trial_idx,
            success=success,
            task_description=task_description,
        )

    return results


@hydra.main(version_base="1.3", config_path="../../configs", config_name="sim_libero")
def eval_single_process(cfg: DictConfig):
    """Main evaluation function."""
    start_time = time.time()
    partial_state = PartialState()
    partial_state.config = cfg

    if cfg.ckpt is None:
        raise ValueError("cfg.ckpt must not be None.")

    device = cfg.EVALUATION.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    
    # Ensure device has a specific index
    if device == "cuda" or (isinstance(device, torch.device) and device.type == "cuda" and device.index is None):
        device = "cuda:0"
    
    # Import and build model
    from cosmos_wam.models.cosmos_wam import CosmosWAM
    from cosmos_wam.models.dit_wrapper import MiniTrainDIT
    from cosmos_wam.models.vae_wrapper import Wan2pt1VAEInterface
    from cosmos_wam.models.action_head import ActionDiT
    
    # Build model
    vae = Wan2pt1VAEInterface(
        vae_pth=cfg.model.vae_checkpoint,
        temporal_window=cfg.model.get("vae_temporal_window", 16),
        device=torch.device(device),
    )
    vae.eval()
    
    dit = MiniTrainDIT(
        max_img_h=cfg.model.dit_config.max_img_h,
        max_img_w=cfg.model.dit_config.max_img_w,
        max_frames=cfg.model.dit_config.max_frames,
        in_channels=cfg.model.dit_config.in_channels,
        out_channels=cfg.model.dit_config.out_channels,
        patch_spatial=cfg.model.dit_config.patch_spatial,
        patch_temporal=cfg.model.dit_config.patch_temporal,
        model_channels=cfg.model.dit_config.model_channels,
        num_blocks=cfg.model.dit_config.num_blocks,
        num_heads=cfg.model.dit_config.num_heads,
        mlp_ratio=cfg.model.dit_config.get("mlp_ratio", 4.0),
        atten_backend=cfg.model.dit_config.get("atten_backend", "minimal_a2a"),
        crossattn_emb_channels=cfg.model.dit_config.crossattn_emb_channels,
        use_crossattn_projection=cfg.model.dit_config.get("use_crossattn_projection", False),
        crossattn_proj_in_channels=cfg.model.dit_config.get("crossattn_proj_in_channels", 1024),
        pos_emb_cls=cfg.model.dit_config.get("pos_emb_cls", "rope3d"),
        pos_emb_learnable=cfg.model.dit_config.get("pos_emb_learnable", False),
        rope_h_extrapolation_ratio=cfg.model.dit_config.get("rope_h_extrapolation_ratio", 1.0),
        rope_w_extrapolation_ratio=cfg.model.dit_config.get("rope_w_extrapolation_ratio", 1.0),
        rope_t_extrapolation_ratio=cfg.model.dit_config.get("rope_t_extrapolation_ratio", 1.0),
        use_wan_fp32_strategy=cfg.model.dit_config.get("use_wan_fp32_strategy", False),
        adaln_lora_dim=cfg.model.dit_config.get("adaln_lora_dim", 256),
        use_t_embedding_adaln_lora=cfg.model.dit_config.get("use_t_embedding_adaln_lora", True),
    )
    
    action_head = ActionDiT(
        action_dim=cfg.model.action_head.action_dim,
        hidden_dim=cfg.model.action_head.hidden_dim,
        num_layers=cfg.model.action_head.num_layers,
        num_heads=cfg.model.action_head.num_heads,
        video_dim=cfg.model.action_head.video_dim,
        mlp_ratio=cfg.model.action_head.get("mlp_ratio", 4.0),
        actions_per_latent=cfg.model.action_head.get("actions_per_latent", 8),
    )
    
    model = CosmosWAM(
        dit=dit,
        vae=vae,
        action_head=action_head,
        lambda_action=cfg.model.get("lambda_action", 1.0),
        num_cond_frames=cfg.model.get("num_cond_frames", 1),
    )
    model = model.to(device).eval()
    
    # Load checkpoint
    ckpt = torch.load(cfg.ckpt, map_location="cpu")
    state_dict = ckpt["model"] if "model" in ckpt else ckpt
    filtered_state = {k: v for k, v in state_dict.items() if not k.startswith("vae.")}
    model.load_state_dict(filtered_state, strict=False)
    logging.info("Loaded checkpoint: %s", cfg.ckpt)

    # Setup processor
    dataset_stats_path = cfg.EVALUATION.get("dataset_stats_path")
    if dataset_stats_path is None:
        raise ValueError("EVALUATION.dataset_stats_path is required.")
    
    dataset_stats = load_dataset_stats_from_json(dataset_stats_path)
    processor: FastWAMProcessor = instantiate(cfg.data.train.processor).eval()
    processor.set_normalizer_from_stats(dataset_stats)

    action_horizon = int(cfg.EVALUATION.get("action_horizon", cfg.data.train.num_frames - 1))
    video_size = cfg.data.train.get("video_size", [224, 224])
    input_h, input_w = int(video_size[0]), int(video_size[1])

    # Setup output directories
    local_log_dir = Path(cfg.EVALUATION.output_dir)
    local_log_dir.mkdir(parents=True, exist_ok=True)
    video_dir = local_log_dir / cfg.EVALUATION.task_suite_name / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)

    # Get task
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.EVALUATION.task_suite_name]()
    task = task_suite.get_task(cfg.EVALUATION.task_id)
    initial_states = task_suite.get_task_init_states(cfg.EVALUATION.task_id)

    while len(initial_states) < int(cfg.EVALUATION.num_trials):
        initial_states.extend(initial_states[: (int(cfg.EVALUATION.num_trials) - len(initial_states))])

    results = {
        "task_suite": cfg.EVALUATION.task_suite_name,
        "task_id": cfg.EVALUATION.task_id,
        "task_description": None,
        "successes": 0,
        "total_episodes": int(cfg.EVALUATION.num_trials),
        "gpu_id": int(cfg.EVALUATION.get("gpu_id", 0)),
        "success_episodes": [],
        "failure_episodes": [],
        "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "duration": 0,
    }

    # Initialize online text encoder if enabled
    online_encoder = None
    if cfg.EVALUATION.get("use_online_text_encoder", False):
        online_encoder_path = cfg.EVALUATION.get("online_text_encoder_path")
        if online_encoder_path is None:
            raise ValueError("online_text_encoder_path is required when use_online_text_encoder=True")
        text_enc_device = cfg.EVALUATION.get("text_encoder_device", "cuda:0")
        online_encoder = OnlineTextEncoder(online_encoder_path, device=text_enc_device)
        logging.info(f"Using online text encoder on {text_enc_device}")
    
    logging.info("Running LIBERO evaluation")
    task_results = run_single_task(
        task=task,
        initial_states=initial_states,
        model=model,
        processor=processor,
        cfg=cfg,
        video_dir=video_dir,
        action_horizon=action_horizon,
        input_w=input_w,
        input_h=input_h,
        device=device,
        online_encoder=online_encoder,
    )
    results.update(task_results)

    results["duration"] = time.time() - start_time
    output_dir = Path(cfg.EVALUATION.output_dir) / cfg.EVALUATION.task_suite_name
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"gpu{cfg.EVALUATION.get('gpu_id', 0)}_task{cfg.EVALUATION.task_id}_results.json"

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, cls=NumpyEncoder)

    print(f"Task {cfg.EVALUATION.task_id} completed: {results['successes']}/{cfg.EVALUATION.num_trials} successes")
    print(f"Time taken: {results['duration']:.2f} seconds")
    return results


if __name__ == "__main__":
    eval_single_process()
