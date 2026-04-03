import torch
import torch.nn as nn


def load_dit_from_checkpoint(dit_model: nn.Module, ckpt_path: str, strict: bool = False) -> None:
    """Load DIT checkpoint, handling key mapping from Cosmos format."""
    state_dict = torch.load(ckpt_path, map_location="cpu")
    
    # If checkpoint has nested 'model' key, extract it
    if isinstance(state_dict, dict) and "model" in state_dict:
        state_dict = state_dict["model"]
    
    # Key mapping from Cosmos checkpoint format to MiniTrainDIT format
    new_state_dict = {}
    for k, v in state_dict.items():
        # Skip training metadata
        if any(x in k for x in ['accum_video', 'accum_image', 'accum_iteration', 'accum_train', 'pos_embedder']):
            continue
        
        new_k = k
        
        # Remove 'net.' prefix
        if new_k.startswith('net.'):
            new_k = new_k[4:]
        
        # Map x_embedder to patch_embedding
        if new_k.startswith('x_embedder.'):
            new_k = new_k.replace('x_embedder.', 'patch_embedding.')
        
        # Map t_embedder to t_embedding (handle nested structure)
        if new_k.startswith('t_embedder.1.'):
            new_k = new_k.replace('t_embedder.1.', 't_embedding.')
        elif new_k.startswith('t_embedder.'):
            new_k = new_k.replace('t_embedder.', 't_embedding.')
        
        # Map t_embedding_norm
        if new_k == 't_embedding_norm.weight' or new_k == 't_embedder_norm.weight':
            pass  # keep as is
        
        # Map final_layer adaln_modulation indices: .1 -> [1], .2 -> [2]
        # Checkpoint: final_layer.adaln_modulation.1.weight -> Model: final_layer.adaln_modulation.1.weight
        # The Sequential indices already match (1, 2)
        
        # Map crossattn_proj: .0 -> [0]
        if new_k.startswith('crossattn_proj.0.'):
            new_k = new_k.replace('crossattn_proj.0.', 'crossattn_proj.0.')
        elif new_k.startswith('crossattn_proj.1.'):
            # Skip activation layer
            continue
        
        # Map blocks mlp: layer1 -> layer1, layer2 -> layer2
        if '.mlp.layer1.' in new_k:
            new_k = new_k.replace('.mlp.layer1.', ".mlp.layer1.")
        if '.mlp.layer2.' in new_k:
            new_k = new_k.replace('.mlp.layer2.', ".mlp.layer2.")
        
        # Skip _extra_state keys (RMSNorm internal states)
        if '_extra_state' in new_k:
            continue
        
        new_state_dict[new_k] = v
    
    # Load with shape checking
    model_state = dit_model.state_dict()
    filtered_state_dict = {}
    incompatible = []
    
    for k, v in new_state_dict.items():
        if k in model_state:
            if model_state[k].shape == v.shape:
                filtered_state_dict[k] = v
            else:
                incompatible.append(f"{k}: ckpt={tuple(v.shape)}, model={tuple(model_state[k].shape)}")
        else:
            incompatible.append(f"{k}: not in model")
    
    print(f"[ckpt] Loading DIT from {ckpt_path}")
    print(f"[ckpt] Loading {len(filtered_state_dict)}/{len(new_state_dict)} keys")
    
    if incompatible:
        print(f"[ckpt] {len(incompatible)} incompatible keys (showing first 10):")
        for msg in incompatible[:10]:
            print(f"  {msg}")
    
    missing, unexpected = dit_model.load_state_dict(filtered_state_dict, strict=False)
    
    if missing:
        print(f"[ckpt] Missing keys ({len(missing)}): {list(missing)[:5]}{'...' if len(missing) > 5 else ''}")
    if unexpected:
        print(f"[ckpt] Unexpected keys ({len(unexpected)}): {list(unexpected)[:5]}{'...' if len(unexpected) > 5 else ''}")


def load_vae_from_checkpoint(vae_model: nn.Module, ckpt_path: str, strict: bool = False) -> None:
    state_dict = torch.load(ckpt_path, map_location="cpu")
    missing, unexpected = vae_model.load_state_dict(state_dict, strict=strict)
    print(f"[ckpt] Loaded VAE from {ckpt_path}")
    if missing:
        print(f"[ckpt] VAE missing keys ({len(missing)}): {missing[:5]}{'...' if len(missing) > 5 else ''}")
    if unexpected:
        print(f"[ckpt] VAE unexpected keys ({len(unexpected)}): {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")
