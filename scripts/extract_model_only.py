"""
Extract model from checkpoint and convert to bf16 to save space.

Usage:
    python scripts/extract_model_only.py <input_ckpt_path> [output_ckpt_path]

Example:
    python scripts/extract_model_only.py \
        ./outputs/cosmos_2b_robotwin_20250406_174530/checkpoints/step_0005000.pt \
        ./outputs/cosmos_2b_robotwin_20250406_174530/checkpoints/step_0005000_small.pt
"""

import sys
import torch
from pathlib import Path


def extract_model_only(input_path: str, output_path: str = None):
    """Load checkpoint, convert to bf16, save smaller checkpoint."""
    input_path = Path(input_path)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)
    
    print(f"Loading checkpoint from {input_path}...")
    ckpt = torch.load(input_path, map_location="cpu")
    
    # Get original size
    original_size = input_path.stat().st_size / (1024**3)
    print(f"Original checkpoint size: {original_size:.2f} GB")
    
    # Extract model state dict
    if isinstance(ckpt, dict) and "model" in ckpt:
        state_dict = ckpt["model"]
    else:
        state_dict = ckpt
    
    # Filter out VAE parameters and convert to bf16
    filtered_state = {}
    vae_keys = []
    for k, v in state_dict.items():
        if k.startswith("vae."):
            vae_keys.append(k)
        else:
            # Convert fp32 to bf16 to save space
            if v.dtype == torch.float32:
                filtered_state[k] = v.to(torch.bfloat16)
            else:
                filtered_state[k] = v
    
    # Count parameters
    total_params = sum(v.numel() for v in filtered_state.values())
    
    print(f"Total keys: {len(state_dict)}")
    print(f"VAE keys removed: {len(vae_keys)}")
    print(f"Remaining keys: {len(filtered_state)}")
    print(f"Parameters: {total_params:,}")
    print(f"Dtype: bf16 (2 bytes per param)")
    
    # Build new checkpoint
    new_ckpt = {
        "model": filtered_state,
        "step": ckpt.get("step", 0),
        "epoch": ckpt.get("epoch", 0),
    }
    
    # Determine output path
    if output_path is None:
        output_path = input_path.parent / f"{input_path.stem}_bf16{input_path.suffix}"
    else:
        output_path = Path(output_path)
    
    # Save
    print(f"\nSaving bf16 checkpoint to {output_path}...")
    torch.save(new_ckpt, output_path)
    
    # Get new size
    new_size = output_path.stat().st_size / (1024**3)
    print(f"New checkpoint size: {new_size:.2f} GB")
    print(f"Size reduction: {original_size - new_size:.2f} GB ({(1 - new_size/original_size)*100:.1f}%)")
    
    return output_path


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    
    input_ckpt = sys.argv[1]
    output_ckpt = sys.argv[2] if len(sys.argv) > 2 else None
    
    extract_model_only(input_ckpt, output_ckpt)
