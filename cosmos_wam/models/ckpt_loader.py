import torch
import torch.nn as nn


def load_dit_from_checkpoint(dit_model: nn.Module, ckpt_path: str, strict: bool = False) -> None:
    state_dict = torch.load(ckpt_path, map_location="cpu")
    missing, unexpected = dit_model.load_state_dict(state_dict, strict=strict)
    print(f"[ckpt] Loaded DIT from {ckpt_path}")
    if missing:
        print(f"[ckpt] DIT missing keys ({len(missing)}): {missing[:5]}{'...' if len(missing) > 5 else ''}")
    if unexpected:
        print(f"[ckpt] DIT unexpected keys ({len(unexpected)}): {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")


def load_vae_from_checkpoint(vae_model: nn.Module, ckpt_path: str, strict: bool = False) -> None:
    state_dict = torch.load(ckpt_path, map_location="cpu")
    missing, unexpected = vae_model.load_state_dict(state_dict, strict=strict)
    print(f"[ckpt] Loaded VAE from {ckpt_path}")
    if missing:
        print(f"[ckpt] VAE missing keys ({len(missing)}): {missing[:5]}{'...' if len(missing) > 5 else ''}")
    if unexpected:
        print(f"[ckpt] VAE unexpected keys ({len(unexpected)}): {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")
