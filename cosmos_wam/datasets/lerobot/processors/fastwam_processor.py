from typing import Dict, Any, Optional
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from .base_processor import BaseProcessor


class FastWAMProcessor(BaseProcessor):
    def __init__(
        self,
        shape_meta: Dict[str, Any],
        num_obs_steps: int,
        num_output_cameras: int,
        action_output_dim: int,
        proprio_output_dim: int,
        delta_action_dim_mask: Optional[Dict[str, list]] = None,
        action_state_transforms=None,
        use_stepwise_action_norm: bool = False,
        norm_default_mode: str = "min/max",
        norm_exception_mode=None,
        action_state_merger=None,
        train_transforms=None,
        val_transforms=None,
    ):
        self.shape_meta = shape_meta
        self.num_obs_steps = num_obs_steps
        self.num_output_cameras = num_output_cameras
        self.action_output_dim = action_output_dim
        self.proprio_output_dim = proprio_output_dim
        self.delta_action_dim_mask = delta_action_dim_mask
        self.action_state_merger = action_state_merger
        if self.action_state_merger is not None:
            self.action_state_merger.set_shape_meta(shape_meta)
        self.train_transforms = train_transforms if train_transforms is not None else []
        self.val_transforms = val_transforms if val_transforms is not None else []
        self._is_train = True

    def train(self):
        self._is_train = True
        return self

    def eval(self):
        self._is_train = False
        return self

    def preprocess(self, data: Dict[str, Any]) -> Dict[str, Any]:
        # Apply transforms to images
        transforms = self.train_transforms if self._is_train else self.val_transforms
        for key in data.get("images", {}):
            image = data["images"][key]
            for trans in transforms:
                image = trans(image)
            data["images"][key] = image

        # Merge action/state
        if self.action_state_merger is not None:
            data = self.action_state_merger.forward(data)

        # Create pixel_values tensor from images
        images_list = []
        for meta in self.shape_meta.get("images", []):
            key = meta["key"]
            if key in data["images"]:
                images_list.append(data["images"][key])
        if images_list:
            data["pixel_values"] = torch.stack(images_list, dim=0)

        # Rename state to proprio for consistency
        if "state" in data:
            data["proprio"] = data.pop("state")

        return data

    def set_normalizer_from_stats(self, stats):
        pass
