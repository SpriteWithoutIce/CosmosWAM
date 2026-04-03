import hashlib
import os
from typing import Optional
import numpy as np
import torch
import torchvision.transforms.functional as transforms_F
from contextlib import contextmanager
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
from .base_lerobot_dataset import BaseLerobotDataset
from ..dataset_utils import ResizeSmallestSideAspectPreserving, CenterCrop, Normalize


DEFAULT_PROMPT = "A video recorded from a robot's point of view executing the following instruction: {task}"


class RobotVideoDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_dirs,
        shape_meta,
        num_frames=33,
        video_size=[384, 640],
        camera_key=None,
        processor=None,
        text_embedding_cache_dir=None,
        context_len=512,
        val_set_proportion=0.05,
        is_training_set=False,
        global_sample_stride=1,
        action_video_freq_ratio: int = 1,
        skip_padding_as_possible: bool = False,
        max_padding_retry: int = 3,
        concat_multi_camera: str = "horizontal",
        override_instruction: Optional[str] = None,
    ):
        self.lerobot_dataset = BaseLerobotDataset(
            dataset_dirs=dataset_dirs,
            shape_meta=OmegaConf.to_container(shape_meta, resolve=True),
            obs_size=num_frames,
            action_size=num_frames - 1,
            val_set_proportion=val_set_proportion,
            is_training_set=is_training_set,
            global_sample_stride=global_sample_stride,
        )

        self.num_frames = num_frames
        self.action_video_freq_ratio = action_video_freq_ratio
        assert (num_frames - 1) % self.action_video_freq_ratio == 0
        assert ((num_frames - 1) // self.action_video_freq_ratio) % 4 == 0
        self.video_sample_indices = list(range(0, num_frames, self.action_video_freq_ratio))

        self.camera_key = camera_key
        self.lerobot_dataset._set_return_images(True)

        self.video_size = video_size
        self.text_embedding_cache_dir = text_embedding_cache_dir
        self.context_len = context_len
        self.skip_padding_as_possible = skip_padding_as_possible
        self.max_padding_retry = max_padding_retry
        self.concat_multi_camera = concat_multi_camera
        self.override_instruction = override_instruction

        self.resize_transform = ResizeSmallestSideAspectPreserving(
            args={"img_w": self.video_size[1], "img_h": self.video_size[0]},
        )
        self.crop_transform = CenterCrop(
            args={"img_w": self.video_size[1], "img_h": self.video_size[0]},
        )
        self.normalize_transform = Normalize(args={"mean": 0.5, "std": 0.5})

        if processor is not None:
            if isinstance(processor, DictConfig):
                processor = instantiate(processor)
            processor.set_normalizer_from_stats(None)  # Will compute on the fly or skip for now
            self.lerobot_dataset.set_processor(processor)

    def __len__(self):
        return len(self.lerobot_dataset)

    def _get(self, idx):
        sample_idx = idx
        sample = None
        for attempt in range(self.max_padding_retry + 1):
            sample = self.lerobot_dataset[sample_idx]
            if not self.skip_padding_as_possible:
                break
            has_pad = False
            if sample.get("action_is_pad") is not None and sample["action_is_pad"].any():
                has_pad = True
            if sample.get("image_is_pad") is not None and sample["image_is_pad"].any():
                has_pad = True
            if not has_pad or attempt >= self.max_padding_retry:
                break
            sample_idx = np.random.randint(len(self.lerobot_dataset))

        video = sample["pixel_values"]  # [T, C, H, W] or [num_cameras, T, C, H, W]
        num_cameras = 1
        if video.ndim == 5:
            video = video[:, self.video_sample_indices, :, :, :]
            num_cameras, T_video, C, H, W = video.shape
        else:
            assert video.ndim == 4
            video = video[self.video_sample_indices, :, :, :]
            T_video, C, H, W = video.shape

        video = video.view(num_cameras, T_video, C, H, W)
        if num_cameras > 1:
            if self.concat_multi_camera == "horizontal":
                video = torch.cat([video[i] for i in range(num_cameras)], dim=-1)
            elif self.concat_multi_camera == "vertical":
                video = torch.cat([video[i] for i in range(num_cameras)], dim=-2)
            else:
                raise ValueError(f"Invalid concat_multi_camera: {self.concat_multi_camera}")
        else:
            video = video.squeeze(0)

        video = self.resize_transform(video)
        video = self.crop_transform(video)
        video = self.normalize_transform(video)
        video = video.permute(1, 0, 2, 3)  # [C, T_video, H, W]

        action = sample["action"]
        proprio = sample["state"][:, :-1, :] if sample["state"].shape[1] > 1 else sample["state"]

        task = sample["task"]
        if self.override_instruction is not None:
            task = self.override_instruction
        instruction = DEFAULT_PROMPT.format(task=task)

        context, context_mask = self._get_cached_text_context(instruction)
        context[~context_mask] = 0.0
        context_mask = torch.ones_like(context_mask)

        return {
            "video": video,
            "action": action,
            "proprio": proprio,
            "prompt": instruction,
            "context": context,
            "context_mask": context_mask,
            "image_is_pad": sample.get("image_is_pad", torch.zeros(video.shape[1], dtype=torch.bool)),
            "action_is_pad": sample.get("action_is_pad", torch.zeros(action.shape[0], dtype=torch.bool)),
            "proprio_is_pad": sample.get("state_is_pad", torch.zeros(proprio.shape[0], dtype=torch.bool)),
        }

    def _get_cached_text_context(self, prompt: str):
        if self.text_embedding_cache_dir is None:
            raise ValueError("text_embedding_cache_dir is not set.")
        cache_dir = self.text_embedding_cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        hashed = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        cache_path = os.path.join(cache_dir, f"{hashed}.t5_len{self.context_len}.pt")
        if not os.path.exists(cache_path):
            raise FileNotFoundError(f"Missing text embedding cache: {cache_path}")
        payload = torch.load(cache_path, map_location="cpu")
        context = payload["context"]
        context_mask = payload["mask"].bool()
        return context, context_mask

    def __getitem__(self, idx):
        try:
            return self._get(idx)
        except Exception as e:
            print(f"Error processing sample {idx}: {e}. Returning random sample.")
            random_idx = np.random.randint(len(self))
            return self._get(random_idx)
