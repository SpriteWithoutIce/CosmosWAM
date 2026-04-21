import os
import math
from datetime import datetime
import torch
from torch.utils.data import DataLoader
from accelerate import Accelerator
from accelerate.utils import DeepSpeedPlugin
from omegaconf import DictConfig, OmegaConf

from .utils.samplers import ResumableEpochSampler
from .utils.logging_config import get_logger

# Optional swanlab import
try:
    import swanlab
    HAS_SWANLAB = True
except ImportError:
    HAS_SWANLAB = False
    swanlab = None

logger = get_logger(__name__)


def set_global_seed(seed: int, get_worker_init_fn: bool = False):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    def worker_init_fn(worker_id):
        worker_seed = seed + worker_id
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    return worker_init_fn if get_worker_init_fn else None


class CosmosWAMTrainer:
    def __init__(self, model, train_dataset, val_dataset=None, *, cfg: DictConfig):
        self.data_cfg = cfg.data
        cfg = cfg.trainer
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.cfg = cfg
        self.output_dir = str(cfg.output_dir)
        self.learning_rate = float(cfg.learning_rate)
        self.weight_decay = float(cfg.weight_decay)
        self.batch_size = int(cfg.batch_size)
        self.num_workers = int(cfg.num_workers)
        self.prefetch_factor = cfg.get("prefetch_factor", None)
        self.persistent_workers = bool(cfg.get("persistent_workers", False))
        self.num_epochs = int(cfg.num_epochs)
        max_steps = cfg.get("max_steps", None)
        self.max_steps = int(max_steps) if max_steps is not None else None
        self.log_every = int(cfg.log_every)
        self.save_every = int(cfg.save_every)
        self.eval_every = int(cfg.eval_every)
        self.gradient_accumulation_steps = int(cfg.gradient_accumulation_steps)
        self.max_grad_norm = float(cfg.max_grad_norm)
        self.seed = int(cfg.seed)
        self.resume = cfg.get("resume", False)
        self.mixed_precision = str(cfg.get("mixed_precision", "bf16")).strip().lower()

        # Handle DeepSpeed configuration
        deepspeed_plugin = None
        if cfg.get("deepspeed"):
            deepspeed_cfg = cfg.deepspeed
            if isinstance(deepspeed_cfg, DictConfig):
                deepspeed_cfg = OmegaConf.to_container(deepspeed_cfg, resolve=True)
            
            # Map nested config to DeepSpeedPlugin parameters
            ds_kwargs = {}
            if "zero_optimization" in deepspeed_cfg:
                zero_cfg = deepspeed_cfg["zero_optimization"]
                ds_kwargs["zero_stage"] = zero_cfg.get("stage", 2)
            else:
                ds_kwargs["zero_stage"] = deepspeed_cfg.get("zero_stage", 2)
            
            # Other supported parameters
            ds_kwargs["gradient_accumulation_steps"] = deepspeed_cfg.get("gradient_accumulation_steps")
            ds_kwargs["gradient_clipping"] = deepspeed_cfg.get("gradient_clipping")
            ds_kwargs["offload_optimizer_device"] = deepspeed_cfg.get("offload_optimizer_device")
            ds_kwargs["offload_param_device"] = deepspeed_cfg.get("offload_param_device")
            
            deepspeed_plugin = DeepSpeedPlugin(**ds_kwargs)
        
        self.accelerator = Accelerator(
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            mixed_precision=self.mixed_precision,
            step_scheduler_with_optimizer=False,
            deepspeed_plugin=deepspeed_plugin,
        )

        logger.info(
            "Accelerator: distributed_type=%s world_size=%d process_index=%d mixed_precision=%s",
            self.accelerator.distributed_type,
            self.accelerator.num_processes,
            self.accelerator.process_index,
            self.mixed_precision,
        )

        worker_init_fn = set_global_seed(self.seed, get_worker_init_fn=True)
        self.train_loader = self._build_loader(self.train_dataset, worker_init_fn=worker_init_fn)

        # Optimizer: different LR for video (dit) and action head if configured
        video_params = list(self.model.dit.parameters())
        action_params = list(self.model.action_head.parameters())
        param_groups = [
            {"params": video_params, "lr": self.learning_rate, "weight_decay": self.weight_decay},
            {"params": action_params, "lr": float(cfg.get("action_learning_rate", self.learning_rate)), "weight_decay": self.weight_decay},
        ]

        # Read optimizer-specific settings from config
        opt_cfg = cfg.get("optimizer", {})
        adam_betas = tuple(opt_cfg.get("betas", [0.9, 0.95]))
        adam_eps = float(opt_cfg.get("eps", 1.0e-08))
        
        self.optimizer = torch.optim.AdamW(
            param_groups,
            betas=adam_betas,
            eps=adam_eps,
        )

        # Print dataset info
        dataset_size = len(self.train_dataset)
        logger.info("Dataset info: total_samples=%d, batch_size=%d, num_workers=%d", 
                    dataset_size, self.batch_size, self.num_workers)
        
        # Calculate total steps based on dataset size and distributed setup
        # steps_per_epoch = num_samples / (batch_size * num_processes * grad_accum)
        effective_batch_size = self.batch_size * self.accelerator.num_processes * self.gradient_accumulation_steps
        steps_per_epoch = dataset_size // effective_batch_size
        
        # Calculate total steps
        if self.max_steps is None:
            # Train by epochs: calculate steps from epochs
            total_steps = steps_per_epoch * self.num_epochs
            logger.info("Training by epochs: %d epochs * %d steps/epoch = %d total steps (effective_batch_size=%d)", 
                        self.num_epochs, steps_per_epoch, total_steps, effective_batch_size)
        else:
            # Train by fixed steps
            total_steps = self.max_steps
            logger.info("Training by steps: %d total steps (effective_batch_size=%d)", 
                        total_steps, effective_batch_size)
        
        self.max_steps = total_steps
        
        # Use explicit warmup_steps if provided, otherwise use 5% of total
        warmup_steps = cfg.get("warmup_steps", None)
        if warmup_steps is None:
            warmup_steps = int(total_steps * 0.05)
        else:
            warmup_steps = int(warmup_steps)
        logger.info("Scheduler: lr_schedule=%s, warmup_steps=%d, total_steps=%d", 
                    cfg.get("lr_schedule", "cosine"), warmup_steps, total_steps)
        
        self.scheduler = self._build_scheduler(self.optimizer, total_steps, warmup_steps)

        self.global_step = 0
        self.epoch = 0

        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "eval"), exist_ok=True)

        # Load checkpoint BEFORE prepare() - this is important for distributed training
        # Loading weights before accelerator.prepare() ensures they are properly propagated
        resume_ckpt_path = cfg.get("resume_from_checkpoint", None)
        resume_reset_step = cfg.get("resume_reset_step", True)  # 默认 True：从头开始计数
        if resume_ckpt_path and os.path.exists(resume_ckpt_path):
            self._load_checkpoint(resume_ckpt_path, reset_step=resume_reset_step)

        # Log dataloader length before prepare
        len_before = len(self.train_loader)
        
        self.model, self.optimizer, self.train_loader, self.scheduler = self.accelerator.prepare(
            self.model, self.optimizer, self.train_loader, self.scheduler
        )
        self.optimizer.zero_grad(set_to_none=True)
        
        # Log dataloader length after prepare
        len_after = len(self.train_loader)
        logger.info(f"DataLoader length: before_prepare={len_before}, after_prepare={len_after}")
        
        # Initialize swanlab if enabled (only on main process)
        swanlab_config = cfg.get("swanlab", {})
        self.use_swanlab = HAS_SWANLAB and swanlab_config.get("enabled", False)
        if self.use_swanlab and self.accelerator.is_main_process:
            # Generate run name if not specified or set to "auto"
            run_name = swanlab_config.get("name", None)
            if run_name is None or run_name == "auto":
                run_name = self._generate_swanlab_run_name(cfg)
            else:
                # Timestamp
                timestamp = datetime.now().strftime("%m%d_%H%M")
                run_name = f"{run_name}_{timestamp}"
            swanlab.init(
                project=swanlab_config.get("project", "cosmos-wam"),
                experiment_name=run_name,
                config=OmegaConf.to_container(cfg, resolve=True),
                logdir=self.output_dir,
            )
            logger.info("Swanlab initialized: project=%s, run_name=%s", 
                       swanlab_config.get("project", "cosmos-wam"), run_name)

    def _build_loader(self, dataset, worker_init_fn=None):
        sampler = ResumableEpochSampler(
            dataset=dataset,
            seed=self.seed,
            batch_size=self.batch_size,
            num_processes=self.accelerator.num_processes,
        )
        loader_kwargs = {
            "batch_size": self.batch_size,
            "shuffle": False,
            "sampler": sampler,
            "num_workers": self.num_workers,
            "pin_memory": torch.cuda.is_available(),
            "worker_init_fn": worker_init_fn,
            "drop_last": True,
        }
        if self.num_workers > 0:
            loader_kwargs["persistent_workers"] = self.persistent_workers
            if self.prefetch_factor is not None:
                loader_kwargs["prefetch_factor"] = int(self.prefetch_factor)

        return DataLoader(dataset, **loader_kwargs)

    def _generate_swanlab_run_name(self, cfg: DictConfig) -> str:
        """Generate a descriptive swanlab run name based on config."""
        
        
        # Extract dataset info from data config
        data_cfg = self.data_cfg.get("train", {})
        dataset_dirs = data_cfg.get("dataset_dirs", [])
        if dataset_dirs:
            # Extract dataset name from path (e.g., "libero_spatial" from ".../libero_spatial_no_noops_lerobot")
            dataset_names = []
            for d in dataset_dirs:
                ds_name = d.split("/")[-1].replace("_no_noops_lerobot", "").replace("_lerobot", "")
                dataset_names.append(ds_name)
            dataset_info = "+".join(dataset_names) if len(dataset_names) <= 2 else f"{len(dataset_names)}suites"
        else:
            dataset_info = "unknown"
        
        # Get num_cameras from data config
        processor = data_cfg.get("processor", {})
        num_cameras = processor.num_output_cameras

        # Get batch size
        batch_size = cfg.get("batch_size", cfg.get("trainer", {}).get("batch_size", "?"))
        
        # Timestamp
        timestamp = datetime.now().strftime("%m%d_%H%M")
        
        # Format: {dataset}_f{frames}_b{batch}_{timestamp}
        # e.g., "libero_f2_b16_0403_1430"
        run_name = f"{dataset_info}_f{num_cameras}_b{batch_size}_{timestamp}"
        return run_name
    
    def _estimate_total_steps(self):
        steps_per_epoch = len(self.train_loader) // self.gradient_accumulation_steps
        total = steps_per_epoch * self.num_epochs
        return total if self.max_steps is None else min(total, self.max_steps)

    def _build_scheduler(self, optimizer, total_steps, warmup_steps):
        from torch.optim.lr_scheduler import LambdaLR
        
        # Get scheduler type from config (cosine or constant)
        lr_schedule = self.cfg.get("lr_schedule", "cosine")
        min_lr_ratio = self.cfg.get("min_lr", 0.0) / self.learning_rate if self.learning_rate > 0 else 0.0
        
        if lr_schedule == "constant":
            # Constant LR with warmup: warmup for warmup_steps, then keep constant
            def lr_lambda(step):
                if step < warmup_steps:
                    return float(step) / float(max(1, warmup_steps))
                return 1.0  # Keep constant after warmup
        else:
            # Cosine annealing with warmup (default), matching starVLA's cosine_with_min_lr
            def lr_lambda(step):
                if step < warmup_steps:
                    return float(step) / float(max(1, warmup_steps))
                progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
                # cosine from 1.0 down to min_lr_ratio
                cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
                return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay

        return LambdaLR(optimizer, lr_lambda)

    def train(self):
        import time
        from tqdm import tqdm
        
        # Calculate total steps
        total_steps = self.max_steps
        logger.info(f"Starting training: total_steps={total_steps}, epochs={self.num_epochs}, batch_size={self.batch_size}")
        
        # Create progress bar (only on main process)
        pbar = None
        if self.accelerator.is_main_process:
            pbar = tqdm(
                total=total_steps,
                desc="Training",
                unit="step",
                bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
                ncols=100,
            )
        
        start_time = time.time()
        last_log_time = start_time
        
        while self.global_step < self.max_steps and self.epoch < self.num_epochs:
            # Set epoch for sampler to ensure different shuffling each epoch
            if hasattr(self.train_loader, 'sampler') and hasattr(self.train_loader.sampler, 'set_epoch'):
                self.train_loader.sampler.set_epoch(self.epoch)
            
            self.model.train()
            epoch_steps = 0
            for batch in self.train_loader:
                with self.accelerator.accumulate(self.model):
                    loss, loss_dict = self.model.training_loss(batch)
                    self.accelerator.backward(loss)
                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad(set_to_none=True)

                if self.accelerator.sync_gradients:
                    self.global_step += 1
                    epoch_steps += 1
                    
                    # Update progress bar
                    if pbar is not None:
                        pbar.update(1)
                        
                        # Calculate ETA
                        elapsed = time.time() - start_time
                        steps_per_sec = self.global_step / elapsed if elapsed > 0 else 0
                        eta_seconds = (total_steps - self.global_step) / steps_per_sec if steps_per_sec > 0 else 0
                        
                        # Format ETA
                        eta_str = self._format_time(eta_seconds)
                        
                        # Update progress bar postfix
                        pbar.set_postfix({
                            "loss": f"{loss_dict['loss_total']:.3f}",
                            "action": f"{loss_dict['loss_action']:.3f}",
                            "lr": f"{self.scheduler.get_last_lr()[0]:.2e}",
                            "ETA": eta_str,
                        })
                    
                    # Log to file/console
                    if self.accelerator.is_main_process and self.global_step % self.log_every == 0:
                        lr = self.scheduler.get_last_lr()[0]
                        current_time = time.time()
                        elapsed_since_last = current_time - last_log_time
                        last_log_time = current_time
                        
                        steps_per_sec = self.log_every / elapsed_since_last if elapsed_since_last > 0 else 0
                        samples_per_sec = steps_per_sec * self.batch_size * self.accelerator.num_processes
                        
                        logger.info(
                            "step=%d/%d epoch=%d loss=%.4f action=%.4f lr=%.6f speed=%.2fsteps/s %.2fsamples/s",
                            self.global_step,
                            total_steps,
                            self.epoch,
                            loss_dict["loss_total"],
                            loss_dict["loss_action"],
                            lr,
                            steps_per_sec,
                            samples_per_sec,
                        )
                        
                        # Log to swanlab
                        if self.use_swanlab:
                            swanlab.log({
                                "train/loss_total": loss_dict["loss_total"].item(),
                                "train/loss_action": loss_dict["loss_action"].item(),
                                "train/learning_rate": lr,
                                "train/steps_per_sec": steps_per_sec,
                                "train/samples_per_sec": samples_per_sec,
                                "train/epoch": self.epoch,
                            }, step=self.global_step)

                    if self.global_step % self.save_every == 0:
                        self._save_checkpoint()

                    if self.global_step >= self.max_steps:
                        break

            logger.info(f"Epoch {self.epoch} finished: epoch_steps={epoch_steps}, global_step={self.global_step}")
            self.epoch += 1

        if pbar is not None:
            pbar.close()
            
        self._save_checkpoint(is_final=True)
        total_time = time.time() - start_time
        logger.info("Training finished. Total steps: %d, Total time: %s", 
                    self.global_step, self._format_time(total_time))
        
        # Close swanlab
        if self.use_swanlab and self.accelerator.is_main_process:
            swanlab.finish()
            logger.info("Swanlab run finished")
    
    def _format_time(self, seconds: float) -> str:
        """Format seconds to human readable string."""
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            return f"{seconds/60:.0f}m"
        elif seconds < 86400:
            return f"{seconds/3600:.1f}h"
        else:
            return f"{seconds/86400:.1f}d"

    def _save_checkpoint(self, is_final: bool = False):
        if not self.accelerator.is_main_process:
            return
        
        tag = "final" if is_final else f"step_{self.global_step:07d}"
        path = os.path.join(self.output_dir, "checkpoints", f"{tag}.pt")
        unwrapped = self.accelerator.unwrap_model(self.model)
        
        # Only save DiT and Action Head (VAE is frozen pre-trained, skip to save space)
        # Filter out VAE parameters (keys starting with "vae.")
        full_state = unwrapped.state_dict()
        filtered_state = {k: v for k, v in full_state.items() if not k.startswith("vae.")}
        
        # Convert to bf16 to save space (matching original Cosmos checkpoint format)
        filtered_state_bf16 = {k: v.to(torch.bfloat16) if v.dtype == torch.float32 else v 
                               for k, v in filtered_state.items()}
        
        state = {
            "model": filtered_state_bf16,
            "step": self.global_step,
            "epoch": self.epoch,
        }
        
        torch.save(state, path)
        
        # Log saved size
        size_mb = os.path.getsize(path) / (1024 * 1024)
        logger.info("Saved checkpoint to %s (%.1f MB) [epoch=%d, step=%d]", 
                    path, size_mb, self.epoch, self.global_step)
    
    def _load_checkpoint(self, ckpt_path: str, reset_step: bool = True):
        """Load checkpoint to resume training.
        
        Args:
            ckpt_path: Path to checkpoint file
            reset_step: If True, start from step=0, epoch=0 (hot start).
                       If False, restore step and epoch from checkpoint.
        """
        logger.info("Loading checkpoint from %s (reset_step=%s)", ckpt_path, reset_step)
        
        # Load on CPU first to avoid GPU memory issues
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        
        # Check checkpoint contents
        ckpt_keys = set(checkpoint["model"].keys())
        model_keys = set(self.model.state_dict().keys())
        
        # Check overlap
        overlapping = ckpt_keys & model_keys
        missing_in_ckpt = model_keys - ckpt_keys
        extra_in_ckpt = ckpt_keys - model_keys
        
        logger.info("Checkpoint analysis: ckpt_keys=%d, model_keys=%d, overlapping=%d", 
                    len(ckpt_keys), len(model_keys), len(overlapping))
        if len(overlapping) == 0:
            logger.error("WARNING: No overlapping keys! Checkpoint may have different structure.")
            logger.error("Checkpoint keys (first 10): %s", list(ckpt_keys)[:10])
            logger.error("Model keys (first 10): %s", list(model_keys)[:10])
        else:
            logger.info("Sample overlapping keys: %s", list(overlapping)[:5])
        
        # Load model state directly (called before accelerator.prepare(), so model is not wrapped yet)
        missing, unexpected = self.model.load_state_dict(checkpoint["model"], strict=False)
        
        logger.info("load_state_dict result: missing=%d, unexpected=%d", len(missing), len(unexpected))
        if missing:
            logger.warning("Missing keys when loading checkpoint: %s", list(missing)[:10])
        if unexpected:
            logger.warning("Unexpected keys when loading checkpoint: %s", list(unexpected)[:10])
        
        # Restore training state
        if reset_step:
            # Hot start: load weights only, start from step=0, epoch=0
            self.global_step = 0
            self.epoch = 0
            logger.info("Hot start: loaded model weights, starting from step=0, epoch=0")
        else:
            # Full resume: restore step and epoch
            self.global_step = checkpoint.get("step", 0)
            self.epoch = checkpoint.get("epoch", 0)
            logger.info("Resumed training from step %d, epoch %d", self.global_step, self.epoch)
