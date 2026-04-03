"""Base LeRobot dataset using lerobot library."""

import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
from torch.utils.data import Dataset

try:
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    HAS_LEROBOT = True
except ImportError:
    HAS_LEROBOT = False
    print("Warning: lerobot not installed")


class BaseLerobotDataset(Dataset):
    """Simple LeRobot dataset wrapper."""
    
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
        if not HAS_LEROBOT:
            raise ImportError("lerobot library is required. Install with: pip install lerobot")
        
        self.dataset_dirs = dataset_dirs
        self.shape_meta = shape_meta
        self.action_size = action_size
        self.obs_size = obs_size
        
        # Load datasets
        self.datasets = []
        for ds_dir in dataset_dirs:
            ds = LeRobotDataset(repo_id=ds_dir, root=Path(ds_dir))
            self.datasets.append(ds)
        
        # For simplicity, use the first dataset
        # TODO: Support multiple datasets properly
        self.dataset = self.datasets[0]
        
        self.image_meta = shape_meta.get("images", [])
        self.state_meta = shape_meta.get("state", [])
        self.action_meta = shape_meta.get("action", [])
        
        self.processor = None
        self.return_images = False
    
    def _set_return_images(self, value: bool):
        self.return_images = value
    
    def set_processor(self, processor):
        self.processor = processor
    
    def get_dataset_stats(self, processor):
        """Calculate dataset statistics."""
        return {}
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        
        # Extract data based on shape_meta
        sample = {
            "idx": idx,
            "task": item.get("task", ""),
            "images": {},
            "state": {},
            "action": {},
        }
        
        # Get images
        for meta in self.image_meta:
            key = meta["key"]
            # Try different key formats
            img_key = f"observation.images.{key}"
            if img_key in item:
                sample["images"][key] = item[img_key]
            elif key in item:
                sample["images"][key] = item[key]
        
        # Get state
        for meta in self.state_meta:
            key = meta["key"]
            state_key = "observation.state" if key == "default" else f"observation.{key}"
            if state_key in item:
                sample["state"][key] = item[state_key]
            elif key in item:
                sample["state"][key] = item[key]
        
        # Get action
        for meta in self.action_meta:
            key = meta["key"]
            action_key = "action" if key == "default" else f"action.{key}"
            if action_key in item:
                sample["action"][key] = item[action_key]
            elif key in item:
                sample["action"][key] = item[key]
        
        # Padding flags (default to no padding)
        sample["action_is_pad"] = torch.zeros(self.action_size, dtype=torch.bool)
        sample["state_is_pad"] = torch.zeros(self.obs_size, dtype=torch.bool)
        sample["image_is_pad"] = torch.zeros(self.obs_size, dtype=torch.bool)
        
        # Pixel values (for video)
        if self.return_images and "pixel_values" not in sample:
            # Try to get images as pixel values
            if sample["images"]:
                sample["pixel_values"] = list(sample["images"].values())[0]
        
        return sample
