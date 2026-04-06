#!/usr/bin/env python3
"""List all RoboTwin dataset directories for config generation."""

from pathlib import Path
import sys

def list_dataset_dirs(base_path):
    """List all subdirectories that contain LeRobot meta/info.json."""
    base = Path(base_path)
    if not base.exists():
        print(f"Error: {base} does not exist", file=sys.stderr)
        sys.exit(1)
    
    dataset_dirs = []
    for subdir in sorted(base.iterdir()):
        if subdir.is_dir():
            # Check if it looks like a LeRobot dataset
            meta_dir = subdir / "meta"
            if meta_dir.exists():
                info_file = meta_dir / "info.json"
                if info_file.exists():
                    dataset_dirs.append(str(subdir))
    
    return dataset_dirs

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("base_path", help="Base path containing dataset subdirectories")
    parser.add_argument("--format", choices=["yaml", "raw"], default="yaml")
    args = parser.parse_args()
    
    dirs = list_dataset_dirs(args.base_path)
    
    if args.format == "yaml":
        print("dataset_dirs:")
        for d in dirs:
            print(f"  - {d}")
    else:
        for d in dirs:
            print(d)
    
    print(f"\n# Found {len(dirs)} datasets", file=sys.stderr)
