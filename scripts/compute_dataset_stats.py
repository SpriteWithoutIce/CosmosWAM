#!/usr/bin/env python3
"""Compute and save dataset normalization stats."""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from omegaconf import OmegaConf
from hydra.utils import instantiate
from cosmos_wam.utils import misc
from cosmos_wam.datasets.lerobot.processors.fastwam_processor import FastWAMProcessor
from cosmos_wam.datasets.lerobot.utils.normalizer import save_dataset_stats_to_json

def main():
    # Config matching train_cosmos_2b_robotwin.yaml
    dataset_dirs = [
        "/home/jwhe/linyihan/datasets/lerobot_robotwin_eef_test/adjust_bottle-demo_clean_collect_200-50",
        "/home/jwhe/linyihan/datasets/lerobot_robotwin_eef_test/beat_block_hammer-demo_clean_collect_200-50",
        "/home/jwhe/linyihan/datasets/lerobot_robotwin_eef_test/blocks_ranking_rgb-demo_clean_collect_200-50",
        "/home/jwhe/linyihan/datasets/lerobot_robotwin_eef_test/blocks_ranking_size-demo_clean_collect_200-50",
        "/home/jwhe/linyihan/datasets/lerobot_robotwin_eef_test/click_alarmclock-demo_clean_collect_200-50",
        "/home/jwhe/linyihan/datasets/lerobot_robotwin_eef_test/click_bell-demo_clean_collect_200-50",
        "/home/jwhe/linyihan/datasets/lerobot_robotwin_eef_test/dump_bin_bigbin-demo_clean_collect_200-50",
        "/home/jwhe/linyihan/datasets/lerobot_robotwin_eef_test/grab_roller-demo_clean_collect_200-50",
        "/home/jwhe/linyihan/datasets/lerobot_robotwin_eef_test/handover_block-demo_clean_collect_200-50",
        "/home/jwhe/linyihan/datasets/lerobot_robotwin_eef_test/handover_mic-demo_clean_collect_200-50",
    ]
    
    shape_meta = {
        "images": [{"key": "cam_high", "raw_shape": [3, 480, 640], "shape": [3, 240, 320]}],
        "action": [{"key": "default", "raw_shape": 16, "shape": 16}],
        "state": [{"key": "default", "raw_shape": 16, "shape": 16}],
    }
    
    output_path = sys.argv[1] if len(sys.argv) > 1 else "./dataset_stats.json"
    
    print(f"Computing dataset stats for {len(dataset_dirs)} datasets...")
    print(f"Output will be saved to: {output_path}")
    
    # Create processor
    from cosmos_wam.datasets.lerobot.transforms.action_state_merger import ConcatLeftAlign
    processor = FastWAMProcessor(
        shape_meta=shape_meta,
        num_obs_steps=33,
        num_output_cameras=1,
        action_output_dim=16,
        proprio_output_dim=16,
        delta_action_dim_mask={"default": [True, True, True, True, True, True, True, False, True, True, True, True, True, True, True, False]},
        action_state_transforms=None,
        use_stepwise_action_norm=False,
        norm_default_mode="min/max",
        norm_exception_mode=None,
        action_state_merger=ConcatLeftAlign(),
        train_transforms=[],
        val_transforms=[],
    )
    
    # Import here to avoid circular import
    from cosmos_wam.datasets.lerobot.base_lerobot_dataset import BaseLerobotDataset
    
    # Create dataset
    dataset = BaseLerobotDataset(
        dataset_dirs=dataset_dirs,
        shape_meta=shape_meta,
        obs_size=33,
        action_size=32,
        val_set_proportion=0.05,
        is_training_set=True,
        global_sample_stride=1,
    )
    
    # Compute stats
    print("Computing stats...")
    stats = dataset.get_dataset_stats(processor)
    
    # Save stats
    save_dataset_stats_to_json(stats, output_path)
    print(f"Stats saved to: {output_path}")
    
    # Print summary
    print("\nStats summary:")
    for key, value in stats.items():
        if isinstance(value, dict):
            print(f"  {key}:")
            for k, v in value.items():
                if hasattr(v, 'shape'):
                    print(f"    {k}: shape={v.shape}")
                else:
                    print(f"    {k}: {v}")

if __name__ == "__main__":
    main()
