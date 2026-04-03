#!/usr/bin/env python3
"""Diagnose import issues for cosmos_wam."""

import sys
import os

print("=" * 60)
print("Python Import Diagnostics")
print("=" * 60)

# 1. Check Python path
print("\n1. Python sys.path:")
for i, p in enumerate(sys.path):
    print(f"  [{i}] {p}")

# 2. Check cosmos_wam location
print("\n2. Looking for cosmos_wam:")
try:
    import cosmos_wam
    print(f"   Found at: {cosmos_wam.__file__}")
    print(f"   Package contents: {dir(cosmos_wam)}")
except Exception as e:
    print(f"   Import error: {e}")

# 3. Try importing datasets
print("\n3. Looking for datasets:")
try:
    import cosmos_wam.datasets
    print(f"   datasets at: {cosmos_wam.datasets.__file__}")
    print(f"   contents: {dir(cosmos_wam.datasets)}")
except Exception as e:
    print(f"   Import error: {e}")

# 4. Try importing lerobot
print("\n4. Looking for lerobot:")
try:
    import cosmos_wam.datasets.lerobot
    print(f"   lerobot at: {cosmos_wam.datasets.lerobot.__file__}")
    print(f"   contents: {dir(cosmos_wam.datasets.lerobot)}")
except Exception as e:
    print(f"   Import error: {e}")
    import traceback
    traceback.print_exc()

# 5. Try importing RobotVideoDataset
print("\n5. Looking for RobotVideoDataset:")
try:
    from cosmos_wam.datasets.lerobot import RobotVideoDataset
    print(f"   Success! {RobotVideoDataset}")
except Exception as e:
    print(f"   Import error: {e}")
    import traceback
    traceback.print_exc()

# 6. Check file existence
print("\n6. Checking file existence:")
files_to_check = [
    "cosmos_wam/__init__.py",
    "cosmos_wam/datasets/__init__.py",
    "cosmos_wam/datasets/lerobot/__init__.py",
    "cosmos_wam/datasets/lerobot/robot_video_dataset.py",
]
for f in files_to_check:
    exists = os.path.exists(f)
    print(f"   {f}: {'EXISTS' if exists else 'MISSING'}")

print("\n" + "=" * 60)
