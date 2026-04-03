"""Analyze Cosmos checkpoint structure."""

import argparse
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="Path to checkpoint")
    args = parser.parse_args()

    print(f"Loading checkpoint: {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location="cpu")
    
    print(f"\n=== Checkpoint Type: {type(ckpt)} ===")
    
    if isinstance(ckpt, dict):
        print(f"\n=== Top-level keys ({len(ckpt.keys())}) ===")
        for k in sorted(ckpt.keys()):
            v = ckpt[k]
            if isinstance(v, dict):
                print(f"  {k}: dict with {len(v)} keys")
            elif isinstance(v, torch.Tensor):
                print(f"  {k}: tensor {tuple(v.shape)}")
            else:
                print(f"  {k}: {type(v)}")
        
        # Check for model/state_dict
        if "model" in ckpt:
            state_dict = ckpt["model"]
            print(f"\n=== Model keys (sample, first 30) ===")
            for k in list(state_dict.keys())[:30]:
                v = state_dict[k]
                if isinstance(v, torch.Tensor):
                    print(f"  {k}: {tuple(v.shape)}")
                else:
                    print(f"  {k}: {type(v)}")
            
            # Analyze key prefixes
            print(f"\n=== Key prefix analysis ===")
            prefixes = {}
            for k in state_dict.keys():
                prefix = k.split('.')[0]
                prefixes[prefix] = prefixes.get(prefix, 0) + 1
            for prefix, count in sorted(prefixes.items(), key=lambda x: -x[1]):
                print(f"  {prefix}: {count} keys")
                
            # Check for specific patterns
            print(f"\n=== Specific patterns ===")
            adaln_keys = [k for k in state_dict.keys() if 'adaln' in k.lower()]
            print(f"  AdaLN keys: {len(adaln_keys)}")
            if adaln_keys:
                print(f"    Sample: {adaln_keys[:5]}")
            
            lora_keys = [k for k in state_dict.keys() if 'lora' in k.lower()]
            print(f"  LoRA keys: {len(lora_keys)}")
            if lora_keys:
                print(f"    Sample: {lora_keys[:5]}")
            
            block_keys = [k for k in state_dict.keys() if k.startswith('net.blocks.') or k.startswith('blocks.')]
            print(f"  Block keys: {len(block_keys)}")
            
            # Find blocks.0 keys
            block0_keys = [k for k in state_dict.keys() if '.blocks.0.' in k or k.startswith('blocks.0.')]
            print(f"\n=== blocks.0 keys ===")
            for k in sorted(block0_keys):
                v = state_dict[k]
                if isinstance(v, torch.Tensor):
                    print(f"  {k}: {tuple(v.shape)}")
    
    elif isinstance(ckpt, torch.Tensor):
        print(f"Checkpoint is a single tensor: {tuple(ckpt.shape)}")
    else:
        print(f"Unknown checkpoint format: {type(ckpt)}")


if __name__ == "__main__":
    main()
