"""Fix VAE checkpoint format for cosmos-predict2.5.

If your VAE checkpoint has nested structure (e.g., {'model': {...}}),
this script extracts the state_dict.
"""

import argparse
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input VAE checkpoint")
    parser.add_argument("--output", required=True, help="Output fixed checkpoint")
    args = parser.parse_args()

    print(f"Loading checkpoint from {args.input}")
    ckpt = torch.load(args.input, map_location="cpu")
    
    print(f"Checkpoint type: {type(ckpt)}")
    
    if isinstance(ckpt, dict):
        print(f"Keys: {list(ckpt.keys())}")
        
        # Check if it's a nested checkpoint
        if "model" in ckpt:
            print("Found nested 'model' key, extracting...")
            state_dict = ckpt["model"]
        elif "state_dict" in ckpt:
            print("Found nested 'state_dict' key, extracting...")
            state_dict = ckpt["state_dict"]
        else:
            print("Using checkpoint as-is (already a state_dict)")
            state_dict = ckpt
        
        # Print sample keys
        print(f"\nState dict has {len(state_dict)} keys")
        print("Sample keys:")
        for k in list(state_dict.keys())[:10]:
            print(f"  {k}")
        
        # Check for expected VAE keys
        has_encoder = any("encoder" in k for k in state_dict.keys())
        has_decoder = any("decoder" in k for k in state_dict.keys())
        print(f"\nHas encoder keys: {has_encoder}")
        print(f"Has decoder keys: {has_decoder}")
        
        # Save fixed checkpoint
        torch.save(state_dict, args.output)
        print(f"\nSaved fixed checkpoint to {args.output}")
    else:
        print(f"Warning: Checkpoint is not a dict, type: {type(ckpt)}")
        torch.save(ckpt, args.output)
        print(f"Saved to {args.output} (unchanged)")


if __name__ == "__main__":
    main()
