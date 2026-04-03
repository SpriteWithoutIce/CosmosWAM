"""Convert Cosmos-Predict2.5 checkpoint to MiniTrainDIT format."""

import argparse
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input Cosmos checkpoint")
    parser.add_argument("--output", required=True, help="Output checkpoint")
    args = parser.parse_args()

    print(f"Loading checkpoint: {args.input}")
    ckpt = torch.load(args.input, map_location="cpu")
    
    if isinstance(ckpt, dict) and "model" in ckpt:
        state_dict = ckpt["model"]
    else:
        state_dict = ckpt
    
    print(f"Original keys: {len(state_dict)}")
    
    new_state_dict = {}
    
    for k, v in state_dict.items():
        # Skip training metadata
        if any(x in k for x in ['accum_video', 'accum_image', 'accum_iteration', 'accum_train']):
            continue
        
        # Remove 'net.' prefix
        if k.startswith('net.'):
            new_k = k[4:]
        else:
            new_k = k
        
        # Map x_embedder to patch_embedding
        if new_k == 'x_embedder.proj.1.weight':
            new_k = 'patch_embedding.proj.0.weight'
            # Initialize bias to zeros
            new_state_dict[new_k] = v
            new_state_dict['patch_embedding.proj.0.bias'] = torch.zeros(v.shape[0])
            continue
        
        # Map t_embedder to t_embedding (it has nested structure)
        if new_k.startswith('t_embedder.1.'):
            new_k = new_k.replace('t_embedder.1.', 't_embedding.')
        elif new_k.startswith('t_embedder.'):
            new_k = new_k.replace('t_embedder.', 't_embedding.')
        
        # Map t_embedding_norm
        if new_k == 't_embedding_norm.weight':
            pass  # already correct
        
        # Map final_layer - checkpoint has .1 and .2 (Sequential), model uses different naming
        if new_k.startswith('final_layer.adaln_modulation.'):
            # final_layer.adaln_modulation.1.weight -> final_layer.adaln_modulation.1.weight (keep)
            # final_layer.adaln_modulation.2.weight -> final_layer.adaln_lora.weight
            if '.2.weight' in new_k:
                new_k = new_k.replace('.2.weight', '.adaln_lora.weight')
        
        # Map blocks
        if new_k.startswith('blocks.'):
            # Map mlp.layer1/layer2 to mlp.0/2
            new_k = new_k.replace('.mlp.layer1.', '.mlp.0.')
            new_k = new_k.replace('.mlp.layer2.', '.mlp.2.')
            
            # Map adaln_modulation_X.2.weight to adaln_lora_X.weight
            for mod_type in ['self_attn', 'cross_attn', 'mlp']:
                old_pattern = f'adaln_modulation_{mod_type}.2.'
                new_pattern = f'adaln_lora_{mod_type}.'
                if old_pattern in new_k:
                    new_k = new_k.replace(old_pattern, new_pattern)
        
        # Map crossattn_proj - checkpoint uses Sequential .0, model uses linear directly
        if new_k.startswith('crossattn_proj.0.'):
            new_k = new_k.replace('crossattn_proj.0.', 'crossattn_proj.')
        elif new_k.startswith('crossattn_proj.1.'):
            # Skip the activation layer
            continue
        
        # Skip pos_embedder (not used in rope3d mode)
        if new_k.startswith('pos_embedder.'):
            continue
        
        new_state_dict[new_k] = v
    
    print(f"After conversion: {len(new_state_dict)}")
    
    # Save
    torch.save(new_state_dict, args.output)
    print(f"Saved to: {args.output}")
    
    # Print sample keys
    print(f"\nSample keys:")
    for k in sorted(new_state_dict.keys())[:20]:
        print(f"  {k}: {tuple(new_state_dict[k].shape)}")


if __name__ == "__main__":
    main()
