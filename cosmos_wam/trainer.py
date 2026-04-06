import os
import torch
from torch.utils.data import DataLoader
from accelerate import Accelerator
from accelerate.utils import DeepSpeedPlugin
from omegaconf import DictConfig, OmegaConf

from .utils.samplers import ResumableEpochSampler
from .utils.logging_config import get_logger

# Optional wandb import
try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False
    wandb = None

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
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.cfg = cfg
        self.output_dir = str(cfg.output_dir)
        self.learning_rate = float(cfg.learning_rate)
        self.weight_decay = float(cfg.weight_decay)
        self.batch_size = int(cfg.batch_size)
        self.num_workers = int(cfg.num_workers)
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

        self.optimizer = torch.optim.AdamW(
            param_groups,
            betas=(0.9, 0.95),
        )

        # Print dataset info
        dataset_size = len(self.train_dataset)
        logger.info("Dataset info: total_samples=%d, batch_size=%d, num_workers=%d", 
                    dataset_size, self.batch_size, self.num_workers)
        
        # Calculate total steps
        if self.max_steps is None:
            # Train by epochs: calculate steps from epochs
            steps_per_epoch = len(self.train_loader) // self.gradient_accumulation_steps
            total_steps = steps_per_epoch * self.num_epochs
            logger.info("Training by epochs: %d epochs * %d steps/epoch = %d total steps", 
                        self.num_epochs, steps_per_epoch, total_steps)
        else:
            # Train by fixed steps
            total_steps = self.max_steps
            logger.info("Training by steps: %d total steps", total_steps)
        
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

        self.model, self.optimizer, self.train_loader, self.scheduler = self.accelerator.prepare(
            self.model, self.optimizer, self.train_loader, self.scheduler
        )
        self.optimizer.zero_grad(set_to_none=True)
        
        # Initialize wandb if enabled (only on main process)
        self.use_wandb = HAS_WANDB and cfg.get("wandb", {}).get("enabled", False)
        if self.use_wandb and self.accelerator.is_main_process:
            wandb_config = cfg.get("wandb", {})
            wandb.init(
                project=wandb_config.get("project", "cosmos-wam"),
                name=wandb_config.get("name", None),
                config=OmegaConf.to_container(cfg, resolve=True),
                dir=self.output_dir,
            )
            logger.info("Wandb initialized: project=%s", wandb_config.get("project", "cosmos-wam"))

    def _build_loader(self, dataset, worker_init_fn=None):
        sampler = ResumableEpochSampler(
            dataset=dataset,
            seed=self.seed,
            batch_size=self.batch_size,
            num_processes=self.accelerator.num_processes,
        )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            worker_init_fn=worker_init_fn,
            drop_last=True,
        )

    def _estimate_total_steps(self):
        steps_per_epoch = len(self.train_loader) // self.gradient_accumulation_steps
        total = steps_per_epoch * self.num_epochs
        return total if self.max_steps is None else min(total, self.max_steps)

    def _build_scheduler(self, optimizer, total_steps, warmup_steps):
        from torch.optim.lr_scheduler import LambdaLR
        
        # Get scheduler type from config (cosine or constant)
        lr_schedule = self.cfg.get("lr_schedule", "cosine")
        
        if lr_schedule == "constant":
            # Constant LR with warmup: warmup for warmup_steps, then keep constant
            def lr_lambda(step):
                if step < warmup_steps:
                    return float(step) / float(max(1, warmup_steps))
                return 1.0  # Keep constant after warmup
        else:
            # Cosine annealing with warmup (default)
            def lr_lambda(step):
                if step < warmup_steps:
                    return float(step) / float(max(1, warmup_steps))
                progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
                return 0.5 * (1.0 + math.cos(math.pi * progress))

        return LambdaLR(optimizer, lr_lambda)

    def train(self):
        import math
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
            self.model.train()
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
                            "video": f"{loss_dict['loss_video']:.3f}",
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
                            "step=%d/%d epoch=%d loss=%.4f video=%.4f action=%.4f lr=%.6f speed=%.2fsteps/s %.2fsamples/s",
                            self.global_step,
                            total_steps,
                            self.epoch,
                            loss_dict["loss_total"],
                            loss_dict["loss_video"],
                            loss_dict["loss_action"],
                            lr,
                            steps_per_sec,
                            samples_per_sec,
                        )
                        
                        # Log to wandb
                        if self.use_wandb:
                            wandb.log({
                                "train/loss_total": loss_dict["loss_total"],
                                "train/loss_video": loss_dict["loss_video"],
                                "train/loss_action": loss_dict["loss_action"],
                                "train/learning_rate": lr,
                                "train/steps_per_sec": steps_per_sec,
                                "train/samples_per_sec": samples_per_sec,
                                "train/epoch": self.epoch,
                            }, step=self.global_step)

                    if self.global_step % self.save_every == 0:
                        self._save_checkpoint()

                    if self.global_step >= self.max_steps:
                        break

            self.epoch += 1

        if pbar is not None:
            pbar.close()
            
        self._save_checkpoint(is_final=True)
        total_time = time.time() - start_time
        logger.info("Training finished. Total steps: %d, Total time: %s", 
                    self.global_step, self._format_time(total_time))
        
        # Close wandb
        if self.use_wandb and self.accelerator.is_main_process:
            wandb.finish()
            logger.info("Wandb run finished")
    
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
        
        # Save complete model state (DiT + VAE + Action Head) for easy loading
        state = {
            "model": unwrapped.state_dict(),
            "step": self.global_step,
            "epoch": self.epoch,
        }
        
        torch.save(state, path)
        
        # Log saved size
        import os
        size_mb = os.path.getsize(path) / (1024 * 1024)
        logger.info("Saved checkpoint to %s (%.1f MB)", path, size_mb)
