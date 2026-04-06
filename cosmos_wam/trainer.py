import os
import torch
from torch.utils.data import DataLoader
from accelerate import Accelerator
from accelerate.utils import DeepSpeedPlugin
from omegaconf import DictConfig, OmegaConf

from .utils.samplers import ResumableEpochSampler
from .utils.logging_config import get_logger

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
            # Convert OmegaConf dict to DeepSpeedPlugin
            deepspeed_cfg = cfg.deepspeed
            if isinstance(deepspeed_cfg, DictConfig):
                deepspeed_cfg = OmegaConf.to_container(deepspeed_cfg, resolve=True)
            deepspeed_plugin = DeepSpeedPlugin(**deepspeed_cfg)
        
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

        total_steps = self._estimate_total_steps()
        self.max_steps = total_steps if self.max_steps is None else self.max_steps
        warmup_steps = int(total_steps * 0.05)
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

        def lr_lambda(step):
            if step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))
            progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        return LambdaLR(optimizer, lr_lambda)

    def train(self):
        import math
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
                    if self.accelerator.is_main_process and self.global_step % self.log_every == 0:
                        lr = self.scheduler.get_last_lr()[0]
                        logger.info(
                            "step=%d epoch=%d loss=%.4f video=%.4f action=%.4f lr=%.6f",
                            self.global_step,
                            self.epoch,
                            loss_dict["loss_total"],
                            loss_dict["loss_video"],
                            loss_dict["loss_action"],
                            lr,
                        )

                    if self.global_step % self.save_every == 0:
                        self._save_checkpoint()

                    if self.global_step >= self.max_steps:
                        break

            self.epoch += 1

        self._save_checkpoint(is_final=True)
        logger.info("Training finished. Total steps: %d", self.global_step)

    def _save_checkpoint(self, is_final: bool = False):
        if not self.accelerator.is_main_process:
            return
        tag = "final" if is_final else f"step_{self.global_step:07d}"
        path = os.path.join(self.output_dir, "checkpoints", f"{tag}.pt")
        unwrapped = self.accelerator.unwrap_model(self.model)
        state = {
            "model": unwrapped.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "step": self.global_step,
            "epoch": self.epoch,
        }
        torch.save(state, path)
        logger.info("Saved checkpoint to %s", path)
