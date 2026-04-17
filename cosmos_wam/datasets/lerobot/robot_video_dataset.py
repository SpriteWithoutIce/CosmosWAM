import hashlib
import os
from typing import Optional
import time
import numpy as np
import traceback
import torch
import torchvision.transforms.functional as transforms_F
from contextlib import contextmanager

from omegaconf import DictConfig, OmegaConf

from hydra.utils import instantiate
from .base_lerobot_dataset import BaseLerobotDataset
from .utils.normalizer import save_dataset_stats_to_json, load_dataset_stats_from_json
from ..dataset_utils import ResizeSmallestSideAspectPreserving, CenterCrop, Normalize
from cosmos_wam.utils.logging_config import get_logger
from cosmos_wam.utils import misc, pytorch_utils
from accelerate import PartialState
from cosmos_wam.utils.projection import project_world_to_pixel, compute_pose_keypoints, generate_gaussian_heatmap
logger = get_logger(__name__)


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
        context_len=128,
        pretrained_norm_stats=None,
        val_set_proportion=0.05,
        is_training_set=False,
        global_sample_stride=1,
        action_video_freq_ratio: int = 1,
        skip_padding_as_possible: bool = False,
        max_padding_retry: int = 3,
        concat_multi_camera: str = "horizontal", # "horizontal", "vertical", "robotwin", or None
        override_instruction: Optional[str] = None, # whether to hardcode a specific instruction for all samples, for debugging
        use_text_prompt_template: bool = True, # whether to use DEFAULT_PROMPT template for text embedding hash
        camera_params_path: Optional[str] = None,
        depth_map_dir: Optional[str] = None,
    ):
        self.lerobot_dataset = BaseLerobotDataset(
            dataset_dirs=dataset_dirs,
            shape_meta=OmegaConf.to_container(shape_meta, resolve=True) if OmegaConf.is_config(shape_meta) else shape_meta,
            obs_size=num_frames,
            action_size=num_frames - 1,
            val_set_proportion=val_set_proportion,
            is_training_set=is_training_set,
            global_sample_stride=global_sample_stride,
        )
    
        self.num_frames = num_frames
        self.action_video_freq_ratio = action_video_freq_ratio
        
        assert (num_frames - 1) % self.action_video_freq_ratio == 0, \
            f"num_frames-1 must be divisible by action_video_freq_ratio, got {num_frames - 1} and {self.action_video_freq_ratio}"
        assert ((num_frames - 1) // self.action_video_freq_ratio) % 4 == 0, \
            f"video frames must be divisible by 4 for tokenization, got {(num_frames - 1) // self.action_video_freq_ratio}"
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
        self.use_text_prompt_template = use_text_prompt_template
        self.camera_params_path = camera_params_path
        self.depth_map_dir = depth_map_dir

        # Load camera params for trajectory projection
        if camera_params_path is not None:
            import json
            with open(camera_params_path, "r") as f:
                params = json.load(f)
            self._intrinsic = np.array(params["intrinsic"], dtype=np.float32)
            self._extrinsic = np.array(params["extrinsic"], dtype=np.float32)
            self._render_h = params.get("image_height", 256)
            self._render_w = params.get("image_width", 256)
        else:
            self._intrinsic = None
            self._extrinsic = None
            self._render_h = 256
            self._render_w = 256

        self.resize_transform = ResizeSmallestSideAspectPreserving(
            args={"img_w": self.video_size[1], "img_h": self.video_size[0]},
        )
        self.crop_transform = CenterCrop(
            args={"img_w": self.video_size[1], "img_h": self.video_size[0]},
        )
        self.normalize_transform = Normalize(
            args={"mean": 0.5, "std": 0.5},
        )
        if processor is not None:
            if isinstance(processor, DictConfig):
                processor = instantiate(processor)
            if not pretrained_norm_stats:
                if not is_training_set:
                    # Try to load from default location (train set should have saved it)
                    work_dir = misc.get_work_dir()
                    default_stats_path = os.path.join(work_dir, "dataset_stats.json")
                    if os.path.exists(default_stats_path):
                        logger.info(f"Loading dataset stats from default location: {default_stats_path}")
                        dataset_stats = load_dataset_stats_from_json(default_stats_path)
                    else:
                        raise ValueError(
                            f"pretrained_norm_stats must be provided for validation/test sets. "
                            f"Could not find stats at default location: {default_stats_path}. "
                            f"Please run training set first to generate stats, or provide pretrained_norm_stats path."
                        )
                else:
                    if PartialState().is_main_process:
                        logger.info("Calculating dataset stats for normalization...")
                        dataset_stats = self.lerobot_dataset.get_dataset_stats(processor)
                        work_dir = misc.get_work_dir()
                        save_dataset_stats_to_json(dataset_stats, os.path.join(work_dir, "dataset_stats.json"))
                    else:
                        dataset_stats = None
                    if torch.distributed.is_available() and torch.distributed.is_initialized():
                        obj_list = [dataset_stats]
                        torch.distributed.broadcast_object_list(obj_list, src=0)
                        dataset_stats = obj_list[0]
            else:
                dataset_stats = load_dataset_stats_from_json(pretrained_norm_stats)
                logger.info(f"Using dataset stats: {pretrained_norm_stats}")
                if PartialState().is_main_process:
                    work_dir = misc.get_work_dir()
                    save_dataset_stats_to_json(dataset_stats, os.path.join(work_dir, "dataset_stats.json"))

            processor.set_normalizer_from_stats(dataset_stats)
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

            action_is_pad = sample["action_is_pad"]
            image_is_pad = sample["image_is_pad"]
            proprio_is_pad = sample["proprio_is_pad"]
            has_pad = False
            if bool(action_is_pad.any().item()):
                has_pad = True
            if bool(image_is_pad.any().item()):
                has_pad = True
            if bool(proprio_is_pad.any().item()):
                has_pad = True

            if not has_pad or attempt >= self.max_padding_retry:
                break

            sample_idx = np.random.randint(len(self.lerobot_dataset))
        
        image_is_pad = sample["image_is_pad"]

        video = sample["pixel_values"]  # [T, C, H, W] or [num_cameras, T, C, H, W]
        num_cameras = 1
        if video.ndim == 5:
            video = video[:, self.video_sample_indices, :, :, :] # [num_cameras, T_video, C, H, W]
            num_cameras, T_video, C, H, W = video.shape
        else:
            assert video.ndim == 4, f"Expected video to have shape [T, C, H, W], but got {video.shape}"
            video = video[self.video_sample_indices, :, :, :] # [T_video, C, H, W]
            T_video, C, H, W = video.shape
        image_is_pad = image_is_pad[self.video_sample_indices]

        video = video.view(num_cameras, T_video, C, H, W)  # [num_cameras, T_video, C, H, W]
        if self.concat_multi_camera == "robotwin":
            if num_cameras != 3:
                raise ValueError(
                    f"`concat_multi_camera='robotwin'` requires exactly 3 cameras, got {num_cameras}"
                )
            cam_top = transforms_F.resize(
                video[0],
                size=[256, 320],
                interpolation=transforms_F.InterpolationMode.BILINEAR,
                antialias=True,
            )  # [T_video, C, 256, 320]
            cam_left = transforms_F.resize(
                video[1],
                size=[128, 160],
                interpolation=transforms_F.InterpolationMode.BILINEAR,
                antialias=True,
            )  # [T_video, C, 128, 160]
            cam_right = transforms_F.resize(
                video[2],
                size=[128, 160],
                interpolation=transforms_F.InterpolationMode.BILINEAR,
                antialias=True,
            )  # [T_video, C, 128, 160]
            bottom = torch.cat([cam_left, cam_right], dim=-1)  # [T_video, C, 128, 320]
            video = torch.cat([cam_top, bottom], dim=-2)  # [T_video, C, 384, 320]
        elif num_cameras > 1:
            if self.concat_multi_camera == "horizontal":
                video = torch.cat([video[i] for i in range(num_cameras)], dim=-1)  # [T_video, C, H, num_cameras*W]
            elif self.concat_multi_camera == "vertical":
                video = torch.cat([video[i] for i in range(num_cameras)], dim=-2)  # [T_video, C, num_cameras*H, W]
            else:
                raise ValueError(
                    f"Invalid concat_multi_camera: {self.concat_multi_camera}. "
                    "Expected one of: horizontal, vertical, robotwin."
                )
        else:
            video = video.squeeze(0)  # [T_video, C, H, W]

        # final resize and normalization
        video = self.resize_transform(video)
        video = self.crop_transform(video)
        video = self.normalize_transform(video)  # [T_video, C, H, W]

        video = video.permute(1, 0, 2, 3) # [C, T_video, H, W], range [-1, 1]

        # Proxy (from lerobot): 
        #   action: [num_frames-1, action_dim] # start from t0, except the last frame
        #   proprio: [num_frames, proprio_dim] # start from t0 to the last frame, aligned with video frames
        action = sample["action"] # [T-1, action_dim]
        proprio = sample["proprio"][:-1, :] # [T-1, state_dim]， to align with action
        if video.shape[1] <= 1:
            raise ValueError(f"`video` must have at least 2 frames, got shape {tuple(video.shape)}")
        if action.shape[0] % (video.shape[1] - 1) != 0:
            raise ValueError(
                f"`action` horizon must be divisible by `video` transitions, got {action.shape[0]} and {video.shape[1] - 1}"
            )

        task = sample["instruction"]
        
        # FIXME
        if self.override_instruction is not None:
            task = self.override_instruction
        
        # Use prompt template or raw task for hash lookup
        if self.use_text_prompt_template:
            instruction = DEFAULT_PROMPT.format(task=task)
            hash_key = instruction
        else:
            # Use raw task description for hash (e.g., LIBERO)
            hash_key = task
            instruction = DEFAULT_PROMPT.format(task=task)  # Still use template for prompt field

        context, context_mask = self._get_cached_text_context(hash_key)
        # NOTE: to keep consistent with wan2.2's behavior
        context[~context_mask] = 0.0
        context_mask = torch.ones_like(context_mask)
        
        # Additional fields for new architecture
        state = proprio[0, :] if proprio is not None else None  # [state_dim]
        eef_pos = proprio[0, :3] if proprio is not None else None  # [3]

        # Target trajectory: 16x16 heatmaps for 4 pose keypoints (origin + XYZ axes)
        target_trajectory = None
        if self._intrinsic is not None and proprio is not None:
            future_proprio = proprio.cpu().numpy()  # [T, 8]
            future_pos = future_proprio[:, :3]   # [T, 3]
            future_axis_angle = future_proprio[:, 3:6]  # [T, 3]  (axis-angle, matching training state format)

            # 1) Compute 4 keypoints per frame
            points_3d = compute_pose_keypoints(future_pos, future_axis_angle, axis_length=0.1)  # [T, 4, 3]

            # 2) Project to 2D pixel coordinates
            points_2d = project_world_to_pixel(
                points_3d.reshape(-1, 3),
                self._intrinsic,
                self._extrinsic,
                self._render_h,
                self._render_w,
            )  # [T*4, 2] numpy
            points_2d = points_2d.reshape(-1, 4, 2)

            # 3) Scale to 16x16 heatmap resolution
            heatmap_size = 16
            scale_x = heatmap_size / self._render_w
            scale_y = heatmap_size / self._render_h
            points_2d[:, :, 0] *= scale_x
            points_2d[:, :, 1] *= scale_y

            # 4) Generate 4-channel Gaussian heatmaps
            T = future_proprio.shape[0]
            heatmaps = np.zeros((T, 4, heatmap_size, heatmap_size), dtype=np.float32)
            sigma = 1.0
            for t in range(T):
                for c in range(4):
                    heatmaps[t, c] = generate_gaussian_heatmap(
                        points_2d[t, c, 0],
                        points_2d[t, c, 1],
                        heatmap_size,
                        heatmap_size,
                        sigma,
                    )

            target_trajectory = torch.from_numpy(heatmaps).float()

        # Depth map for condition frame (frame_idx)
        depth_map = None
        if self.depth_map_dir is not None:
            depth_map = self._get_depth_map(sample["idx"])
            if depth_map is not None:
                depth_map = torch.from_numpy(depth_map).unsqueeze(0).float()

        data = {
            "video": video,
            "action": action,
            "proprio": proprio,
            "state": state,
            "eef_pos": eef_pos,
            "target_trajectory": target_trajectory,
            "depth_map": depth_map,
            "prompt": instruction,
            "context": context,
            "context_mask": context_mask,
            "image_is_pad": image_is_pad,
            "action_is_pad": sample["action_is_pad"],
            "proprio_is_pad": sample["proprio_is_pad"],
        }
        return data

    def _get_depth_map(self, frame_idx: int):
        """Load precomputed depth map for a given global frame index."""
        if self.depth_map_dir is None:
            return None

        # Find global episode index
        ep_data = self.lerobot_dataset.episode_data_index
        ep_mask = (frame_idx >= ep_data["from"]) & (frame_idx < ep_data["to"])
        ep_indices = ep_mask.nonzero(as_tuple=True)[0]
        if len(ep_indices) == 0:
            return None
        ep_idx = ep_indices[0].item()

        # Find dataset index and local episode index
        local_ep_idx = ep_idx
        ds_name = None
        for ds_idx, ds in enumerate(self.lerobot_dataset.multi_dataset._datasets):
            if local_ep_idx < ds.num_episodes:
                ds_name = self.lerobot_dataset.multi_dataset.ds_names[ds_idx]
                break
            local_ep_idx -= ds.num_episodes
        if ds_name is None:
            return None

        filename = f"{ds_name}_episode_{local_ep_idx:06d}_depth.npy"
        depth_path = os.path.join(self.depth_map_dir, filename)
        if not os.path.exists(depth_path):
            # also support per-dataset subfolder structure
            depth_path = os.path.join(self.depth_map_dir, ds_name, filename)
            if not os.path.exists(depth_path):
                return None

        depths = np.load(depth_path)
        local_frame_idx = (frame_idx - ep_data["from"][ep_idx]).item()
        if local_frame_idx >= len(depths):
            return None
        depth = depths[local_frame_idx]
        # Resize to 224x224 if needed
        if depth.shape != (224, 224):
            import cv2
            depth = cv2.resize(depth, (224, 224), interpolation=cv2.INTER_LINEAR)
        return depth.astype(np.float32)

    def _get_cached_text_context(self, prompt: str):
        if self.text_embedding_cache_dir is None:
            raise ValueError("text_embedding_cache_dir is not set.")
        cache_dir = self.text_embedding_cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        hashed = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        cache_filename = f"{hashed}.t5_len{self.context_len}.pt"
        
        # Try flat structure first
        cache_path = os.path.join(cache_dir, cache_filename)
        
        # If not found, search recursively in subdirectories
        if not os.path.exists(cache_path):
            found_path = None
            for root, dirs, files in os.walk(cache_dir):
                if cache_filename in files:
                    found_path = os.path.join(root, cache_filename)
                    break
            if found_path:
                cache_path = found_path
            else:
                raise FileNotFoundError(
                    f"Missing text embedding cache: {cache_filename} in {cache_dir} or its subdirectories. Prompt is: {prompt}"
                    "Run scripts/precompute_text_embeds.py first."
                )
        payload = torch.load(cache_path, map_location="cpu")
        context = payload["context"]
        context_mask = payload["mask"].bool()
        if context.ndim != 2:
            raise ValueError(
                f"Cached `context` must be 2D [L, D], got shape {tuple(context.shape)} in {cache_path}"
            )
        if context_mask.ndim != 1:
            raise ValueError(
                f"Cached `mask` must be 1D [L], got shape {tuple(context_mask.shape)} in {cache_path}"
            )
        if context.shape[0] != self.context_len:
            raise ValueError(
                f"Cached context_len mismatch: expected {self.context_len}, got {context.shape[0]} in {cache_path}"
            )
        if context_mask.shape[0] != self.context_len:
            raise ValueError(
                f"Cached mask_len mismatch: expected {self.context_len}, got {context_mask.shape[0]} in {cache_path}"
            )

        return context, context_mask

    def __getitem__(self, idx):
        try:
            data = self._get(idx)
        except Exception as e:
            print(f"Error processing sample idx {idx}: {e}. Returning a random sample instead.")
            # trace back
            print(traceback.format_exc())
            random_idx = np.random.randint(len(self))
            data = self._get(random_idx)
        return data
