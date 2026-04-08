"""
Cosmos-WAM RoboTwin single-task evaluation entrypoint (Hydra).

Features:
- Read `configs/sim_robotwin.yaml`.
- Check or create the symlink for policy.
- Forward config overrides to the official RoboTwin entrypoint
  `script/eval_policy.py` and save logs.

Common arguments:
- `ckpt`: path to the Cosmos-WAM checkpoint (required).
- `EVALUATION.task_name`: task name to evaluate (required).
- `gpu_id`: sets `CUDA_VISIBLE_DEVICES`.

Examples:
1) Minimal run
   python experiments/robotwin/eval_robotwin_single.py \
     ckpt=/path/to/ckpt.pt \
     EVALUATION.task_name=click_alarmclock \
     EVALUATION.dataset_stats_path=/path/to/dataset_stats.json \
     EVALUATION.text_embedding_cache_dir=/path/to/text_embeds

2) Run with more evaluation overrides
   python experiments/robotwin/eval_robotwin_single.py \
     ckpt=/path/to/ckpt.pt \
     EVALUATION.task_name=click_alarmclock \
     EVALUATION.task_config=demo_randomized \
     EVALUATION.replan_steps=4 \
     EVALUATION.num_inference_steps=20 \
     gpu_id=0
"""

import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[2]
POLICY_NAME = "cosmos_wam_policy"


def _resolve_path(path_str: str, *, base: Path) -> Path:
    path = Path(os.path.expanduser(os.path.expandvars(str(path_str))))
    if not path.is_absolute():
        path = (base / path).resolve()
    return path.resolve()


def _resolve_optional_path(path_value: Any, *, base: Path) -> Path | None:
    if path_value is None:
        return None
    text = str(path_value).strip()
    if text == "" or text.lower() in {"none", "null"}:
        return None
    return _resolve_path(text, base=base)


def _resolve_dataset_stats_path(cfg: DictConfig, ckpt_path: Path) -> Path:
    """Find dataset_stats.json from config or checkpoint parent dirs."""
    explicit = _resolve_optional_path(cfg.EVALUATION.dataset_stats_path, base=PROJECT_ROOT)
    candidates: list[Path] = []
    if explicit is not None:
        candidates.append(explicit)
    
    # Search in checkpoint parent directories
    for parent in list(ckpt_path.parents)[:4]:
        candidates.append((parent / "dataset_stats.json").resolve())
    
    seen: set[Path] = set()
    for path in candidates:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.exists():
            return resolved
    
    raise FileNotFoundError(
        "Failed to locate dataset_stats.json. Tried explicit "
        "EVALUATION.dataset_stats_path and checkpoint parent directories. "
        "Please pass EVALUATION.dataset_stats_path=/path/to/dataset_stats.json."
    )


def _resolve_ckpt_tag(ckpt_path: Path) -> str:
    """Generate checkpoint tag for output directory naming."""
    parts = ckpt_path.resolve().parts
    if "outputs" in parts:
        outputs_idx = parts.index("outputs")
        if outputs_idx + 1 >= len(parts):
            return ckpt_path.stem
        # Use the output dir name (with timestamp)
        return parts[outputs_idx + 1]
    return ckpt_path.stem


def _ensure_policy_symlink(robotwin_root: Path, policy_source_dir: Path) -> Path:
    """Ensure policy symlink exists in RoboTwin/policy directory."""
    policy_root = robotwin_root / "policy"
    if not policy_root.is_dir():
        raise FileNotFoundError(f"RoboTwin policy directory not found: {policy_root}")
    
    policy_target = policy_root / POLICY_NAME
    source_resolved = policy_source_dir.resolve()
    
    if not policy_target.exists() and not policy_target.is_symlink():
        policy_target.symlink_to(source_resolved, target_is_directory=True)
        print(f"Created symlink: {policy_target} -> {source_resolved}")
        return policy_target
    
    if policy_target.is_symlink():
        target_resolved = policy_target.resolve()
        if target_resolved != source_resolved:
            # Update symlink
            policy_target.unlink()
            policy_target.symlink_to(source_resolved, target_is_directory=True)
            print(f"Updated symlink: {policy_target} -> {source_resolved}")
        return policy_target
    
    raise RuntimeError(
        f"Path already exists and is not a symlink: {policy_target}. "
        "Please handle it manually to avoid overriding existing policy files."
    )


def _format_override_value(value: Any) -> str:
    if isinstance(value, bool):
        return "True" if value else "False"
    if value is None:
        return "None"
    if isinstance(value, (int, float)):
        return str(value)
    return repr(str(value))


def _append_override(overrides: list[str], key: str, value: Any, *, skip_none: bool = True) -> None:
    if skip_none and value is None:
        return
    overrides.extend([f"--{key}", _format_override_value(value)])


@hydra.main(version_base="1.3", config_path="../../configs", config_name="sim_robotwin")
def main(cfg: DictConfig):
    if cfg.ckpt is None:
        raise ValueError("`ckpt` must not be None.")
    if cfg.EVALUATION.task_name is None:
        raise ValueError("`EVALUATION.task_name` must not be None.")
    
    ckpt_path = _resolve_path(str(cfg.ckpt), base=PROJECT_ROOT)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    ckpt_tag = _resolve_ckpt_tag(ckpt_path)
    
    # Resolve RoboTwin root
    robotwin_root = _resolve_path(str(cfg.EVALUATION.robotwin_root), base=PROJECT_ROOT)
    if not robotwin_root.exists():
        raise FileNotFoundError(f"RoboTwin root not found: {robotwin_root}")
    
    # Ensure policy symlink
    policy_source_dir = (PROJECT_ROOT / "experiments" / "robotwin" / POLICY_NAME).resolve()
    if not policy_source_dir.is_dir():
        raise FileNotFoundError(f"Policy source directory not found: {policy_source_dir}")
    
    _ensure_policy_symlink(robotwin_root=robotwin_root, policy_source_dir=policy_source_dir)
    
    # Resolve dataset stats
    dataset_stats_path = _resolve_dataset_stats_path(cfg, ckpt_path)
    
    # Resolve text embedding cache dir
    text_embedding_cache_dir = _resolve_optional_path(
        cfg.EVALUATION.get("text_embedding_cache_dir"), 
        base=PROJECT_ROOT
    )
    if text_embedding_cache_dir is None:
        # Try to get from data config
        text_embedding_cache_dir = _resolve_optional_path(
            cfg.data.train.get("text_embedding_cache_dir"),
            base=PROJECT_ROOT
        )
    if text_embedding_cache_dir is None:
        raise ValueError(
            "`EVALUATION.text_embedding_cache_dir` or `data.train.text_embedding_cache_dir` "
            "must be set to the directory containing precomputed text embeddings."
        )
    
    # Setup output directories
    output_dir = _resolve_path(str(cfg.EVALUATION.output_dir), base=PROJECT_ROOT)
    run_ts = output_dir.name
    if run_ts == "":
        raise ValueError(f"Invalid EVALUATION.output_dir (missing run_ts): {output_dir}")
    
    run_output_dir = (
        PROJECT_ROOT
        / "evaluate_results"
        / "robotwin"
        / ckpt_tag
        / run_ts
    )
    run_output_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = run_output_dir / (
        f"eval_{str(cfg.EVALUATION.task_name)}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )
    robotwin_eval_base = (
        PROJECT_ROOT
        / "evaluate_results"
        / "robotwin"
        / ckpt_tag
        / run_ts
        / str(cfg.EVALUATION.task_name)
    )
    
    sim_cfg_path = (PROJECT_ROOT / "configs" / "sim_robotwin.yaml").resolve()
    sim_task = HydraConfig.get().runtime.choices.get("task")
    
    # Build overrides for RoboTwin eval_policy.py
    overrides: list[str] = []
    _append_override(overrides, "task_name", cfg.EVALUATION.task_name)
    _append_override(overrides, "task_config", cfg.EVALUATION.task_config)
    _append_override(overrides, "ckpt_setting", str(ckpt_path))
    _append_override(overrides, "seed", cfg.seed)
    _append_override(overrides, "policy_name", cfg.EVALUATION.policy_name)
    _append_override(overrides, "instruction_type", cfg.EVALUATION.instruction_type)
    _append_override(overrides, "eval_num_episodes", cfg.EVALUATION.eval_num_episodes)
    
    _append_override(overrides, "sim_cfg_path", str(sim_cfg_path))
    _append_override(overrides, "sim_task", sim_task)
    _append_override(overrides, "eval_output_dir", str(robotwin_eval_base))
    _append_override(overrides, "mixed_precision", cfg.mixed_precision)
    _append_override(overrides, "device", cfg.EVALUATION.device)
    _append_override(overrides, "dataset_stats_path", str(dataset_stats_path))
    _append_override(overrides, "text_embedding_cache_dir", str(text_embedding_cache_dir))
    
    # Add fixed_text_embedding_path if provided
    fixed_emb_path = _resolve_optional_path(cfg.EVALUATION.get("fixed_text_embedding_path"), base=PROJECT_ROOT)
    if fixed_emb_path is not None:
        _append_override(overrides, "fixed_text_embedding_path", str(fixed_emb_path))
    
    _append_override(overrides, "action_horizon", cfg.EVALUATION.action_horizon)
    _append_override(overrides, "replan_steps", cfg.EVALUATION.replan_steps)
    _append_override(overrides, "num_inference_steps", cfg.EVALUATION.num_inference_steps)
    _append_override(overrides, "sigma_shift", cfg.EVALUATION.sigma_shift)
    _append_override(overrides, "context_len", cfg.EVALUATION.get("context_len", 512))
    _append_override(
        overrides,
        "skip_get_obs_within_replan",
        cfg.EVALUATION.skip_get_obs_within_replan,
    )
    
    # Add online text encoder settings if enabled
    if cfg.EVALUATION.get("use_online_text_encoder", False):
        _append_override(overrides, "use_online_text_encoder", True)
        _append_override(overrides, "online_text_encoder_path", cfg.EVALUATION.get("online_text_encoder_path"))
        _append_override(overrides, "text_encoder_device", cfg.EVALUATION.get("text_encoder_device"))
        # Also pass the device for the main model
        _append_override(overrides, "device", cfg.EVALUATION.get("device"))
    
    cmd = [
        sys.executable,
        "-u",
        "script/eval_policy.py",
        "--config",
        f"policy/{POLICY_NAME}/deploy_policy.yml",
        "--overrides",
        *overrides,
    ]
    
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    
    # Handle GPU visibility for online text encoder mode
    if cfg.EVALUATION.get("use_online_text_encoder", False):
        # When using online text encoder with dual GPUs, both GPUs need to be visible
        # The devices are controlled via text_encoder_device and device parameters
        text_enc_device = cfg.EVALUATION.get("text_encoder_device", "cuda:0")
        model_device = cfg.EVALUATION.get("device", "cuda")
        
        # Extract GPU IDs from device strings
        def extract_gpu_id(device_str):
            if device_str.startswith("cuda:"):
                return int(device_str.split(":")[1])
            return 0
        
        text_enc_gpu = extract_gpu_id(text_enc_device)
        
        # If model device specifies a GPU, use both
        if model_device.startswith("cuda:"):
            model_gpu = extract_gpu_id(model_device)
        else:
            # Default to gpu_id if device is just "cuda"
            model_gpu = cfg.gpu_id
        
        # Make both GPUs visible
        visible_gpus = sorted(set([text_enc_gpu, model_gpu]))
        env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, visible_gpus))
        
        print(f"Online text encoder mode: Using GPUs {visible_gpus}")
        print(f"  Text encoder on: cuda:{text_enc_gpu}")
        print(f"  Main model on: cuda:{model_gpu}")
    else:
        # Normal mode: only one GPU visible
        env["CUDA_VISIBLE_DEVICES"] = str(cfg.gpu_id)
    
    # Pass cosmos_predict2 path to subprocess
    cosmos_predict2_path = os.environ.get("COSMOS_PREDICT2_PATH", "/home/jwhe/linyihan/cosmos-predict2.5")
    if cosmos_predict2_path:
        env["COSMOS_PREDICT2_PATH"] = cosmos_predict2_path
        # Also add to PYTHONPATH for subprocess
        pythonpath = env.get("PYTHONPATH", "")
        if pythonpath:
            env["PYTHONPATH"] = f"{cosmos_predict2_path}:{pythonpath}"
        else:
            env["PYTHONPATH"] = cosmos_predict2_path
    
    print(f"Running command: {' '.join(cmd)}")
    print(f"Working directory: {robotwin_root}")
    print(f"Log file: {log_file}")
    
    with open(log_file, "w", encoding="utf-8") as log_f:
        process = subprocess.Popen(
            cmd,
            cwd=str(robotwin_root),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log_f.write(line)
            log_f.flush()
        return_code = process.wait()
    
    if return_code != 0:
        raise RuntimeError(f"RoboTwin evaluation failed with return code {return_code}. Log: {log_file}")
    
    print(f"Evaluation finished successfully. Log saved to: {log_file}")
    OmegaConf.save(
        config=cfg,
        f=str(run_output_dir / f"eval_config_{str(cfg.EVALUATION.task_name)}.yaml"),
    )


if __name__ == "__main__":
    main()
