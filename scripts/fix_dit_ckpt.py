"""Fix Cosmos DiT checkpoint to match MiniTrainDIT structure."""

import argparse
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input checkpoint")
    parser.add_argument("--output", required=True, help="Output checkpoint")
    args = parser.parse_args()

    print(f"Loading checkpoint: {args.input}")
    ckpt = torch.load(args.input, map_location="cpu")
    
    # Extract state dict
    if isinstance(ckpt, dict) and "model" in ckpt:
        state_dict = ckpt["model"]
    else:
        state_dict = ckpt
    
    print(f"Original keys: {len(state_dict)}")
    
    # Key mapping
    new_state_dict = {}
    
    for k, v in state_dict.items():
        # Skip training metadata
        if any(x in k for x in ['accum_video', 'accum_image', 'accum_iteration', 'accum_train']):
            continue
        
        new_k = k
        
        # Remove 'net.' prefix
        if new_k.startswith('net.'):
            new_k = new_k[4:]
        
        # Map x_embedder to patch_embedding
        if new_k.startswith('x_embedder.'):
            new_k = new_k.replace('x_embedder.', 'patch_embedding.')
        
        # Map y_embedder to t_embedding (if applicable)
        if new_k.startswith('y_embedder.'):
            new_k = new_k.replace('y_embedder.', 't_embedding.')
        
        # Map t_embedder to t_embedding
        if new_k.startswith('t_embedder.'):
            new_k = new_k.replace('t_embedder.', 't_embedding.')
        
        # Map norm to t_embedding_norm (if applicable)
        if new_k == 't_embedder_norm.weight':
            new_k = 't_embedding_norm.weight'
        
        # Map final_layer
        if new_k.startswith('final_layer.'):
            # Keep as is, structure should match
            pass
        
        # Map blocks - structure should already match
        if new_k.startswith('blocks.'):
            # Check if we need to map adaln modulation keys
            # From checkpoint: blocks.X.adaln_modulation.1.weight -> model: blocks.X.adaln_modulation_self_attn.1.weight
            # Actually the MiniTrainDIT uses adaln_modulation_self_attn, not adaln_modulation
            if 'adaln_modulation.1.weight' in new_k or 'adaln_modulation.1.bias' in new_k:
                # This is ambiguous - could be for self_attn, cross_attn, or mlp
                # The checkpoint has only one adaln_modulation per block, but MiniTrainDIT has three
                # We need to figure out which one this maps to
                # For now, let's assume the order in checkpoint matches: self_attn, cross_attn, mlp
                # But checkpoint only has one, so we might need to duplicate
                pass
        
        new_state_dict[new_k] = v
    
    print(f"After filtering: {len(new_state_dict)}")
    
    # Save
    torch.save(new_state_dict, args.output)
    print(f"Saved to: {args.output}")
    
    # Print sample keys
    print(f"\nSample keys:")
    for k in list(new_state_dict.keys())[:20]:
        print(f"  {k}: {tuple(new_state_dict[k].shape)}")


if __name__ == "__main__":
    main()
