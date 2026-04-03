import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
from torch.utils.data import Dataset

from cosmos_predict2._src.predict2.datasets.lerobot.lerobot_dataset import (
    LeRobotDatasetMetadata,
    MultiLeRobotDataset,
)


class BaseLerobotDataset(Dataset):
    def __init__(
        self,
        dataset_dirs: List[str],
        shape_meta: Dict[str, Any],
        action_size: int = 1,
        obs_size: int = 1,
        val_set_proportion: float = 0.05,
        is_training_set: bool = False,
        seed: int = 42,
        global_sample_stride: int = 1,
    ):
        self.dataset_dirs = dataset_dirs
        self.shape_meta = shape_meta
        self.action_size = action_size
        self.obs_size = obs_size
        self.val_set_proportion = val_set_proportion
        self.is_training_set = is_training_set
        self.seed = seed

        metas = []
        for ds_dir in dataset_dirs:
            meta = LeRobotDatasetMetadata(repo_id=ds_dir, root=Path(ds_dir))
            metas.append(meta)

        fps_list = [m.fps for m in metas]
        assert len(set(fps_list)) == 1, f"All datasets must have same fps, got {fps_list}"
        fps = fps_list[0]

        delta_timestamps = {}
        for meta_key in ["images", "state"]:
            if meta_key in shape_meta:
                for m in shape_meta[meta_key]:
                    key = m["key"]
                    lerobot_key = f"observation.{meta_key}.{key}" if key != "default" else f"observation.{meta_key}"
                    m["lerobot_key"] = lerobot_key
                    delta_timestamps[lerobot_key] = [(t * global_sample_stride) / fps for t in range(obs_size)]

        for m in shape_meta.get("action", []):
            key = m["key"]
            lerobot_key = f"action.{key}" if key != "default" else "action"
            m["lerobot_key"] = lerobot_key
            delta_timestamps[lerobot_key] = [(t * global_sample_stride) / fps for t in range(action_size)]

        episodes = {}
        if val_set_proportion < 1e-6:
            for meta in metas:
                episodes[meta.repo_id] = list(range(meta.total_episodes))
        else:
            for meta in metas:
                split_idx = int(meta.total_episodes * (1 - val_set_proportion))
                rng = np.random.default_rng(seed)
                indices = list(range(meta.total_episodes))
                rng.shuffle(indices)
                if is_training_set:
                    episodes[meta.repo_id] = indices[:split_idx]
                else:
                    episodes[meta.repo_id] = indices[split_idx:]

        self.multi_dataset = MultiLeRobotDataset(
            dataset_dirs=dataset_dirs,
            episodes=episodes,
            delta_timestamps=delta_timestamps,
        )

        self.image_meta = shape_meta.get("images", [])
        self.state_meta = shape_meta.get("state", [])
        self.action_meta = shape_meta.get("action", [])

    def __len__(self):
        return self.multi_dataset.num_frames

    def _get_image(self, meta, lerobot_sample):
        key, raw_shape = meta["key"], meta.get("raw_shape", [3, 224, 224])
        lerobot_key = meta["lerobot_key"]
        image = lerobot_sample[lerobot_key]
        if image.ndim == 3:
            image = image.unsqueeze(0)
        image = (image * 255).to(torch.uint8)
        return image

    def _get_state(self, meta, lerobot_sample):
        lerobot_key = meta["lerobot_key"]
        state = lerobot_sample[lerobot_key]
        if state.ndim == 1:
            state = state.unsqueeze(-1)
        return state

    def _get_action(self, meta, lerobot_sample):
        lerobot_key = meta["lerobot_key"]
        action = lerobot_sample[lerobot_key]
        if action.ndim == 1:
            action = action.unsqueeze(-1)
        return action

    def __getitem__(self, idx):
        if idx >= len(self):
            raise IndexError(f"Index {idx} out of bounds")
        lerobot_sample = self.multi_dataset[idx]
        sample = {
            "idx": idx,
            "task": lerobot_sample.get("task", ""),
            "images": {},
            "state": {},
            "action": {},
        }
        for meta in self.image_meta:
            sample["images"][meta["key"]] = self._get_image(meta, lerobot_sample)
        for meta in self.state_meta:
            sample["state"][meta["key"]] = self._get_state(meta, lerobot_sample)
        for meta in self.action_meta:
            sample["action"][meta["key"]] = self._get_action(meta, lerobot_sample)
        sample["action_is_pad"] = lerobot_sample.get(f"{self.action_meta[0]['lerobot_key']}_is_pad", None)
        sample["state_is_pad"] = lerobot_sample.get(f"{self.state_meta[0]['lerobot_key']}_is_pad", None)
        sample["image_is_pad"] = lerobot_sample.get(f"{self.image_meta[0]['lerobot_key']}_is_pad", None)
        return sample
