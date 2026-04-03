from typing import Optional
import torch
import torch.nn as nn


class VideoTokenizerInterface:
    @property
    def spatial_compression_factor(self) -> int:
        raise NotImplementedError

    @property
    def temporal_compression_factor(self) -> int:
        raise NotImplementedError

    @property
    def latent_ch(self) -> int:
        raise NotImplementedError

    def encode(self, state: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def get_latent_num_frames(self, num_pixel_frames: int) -> int:
        return (num_pixel_frames - 1) // self.temporal_compression_factor + 1


class Wan2pt1VAEInterface(VideoTokenizerInterface, nn.Module):
    def __init__(self, vae_pth: str, temporal_window: int = 16, auto_load: bool = True):
        """
        Args:
            vae_pth: Path to VAE checkpoint
            temporal_window: Temporal window size
            auto_load: If True, let cosmos-predict2.5 load VAE in __init__.
                      If False, VAE will be loaded later via load_vae_from_checkpoint.
        """
        super().__init__()
        # Import from local cosmos checkout
        from cosmos_predict2._src.predict2.tokenizers.wan2pt1 import (
            Wan2pt1VAEInterface as _OrigInterface,
        )

        if auto_load:
            # Let cosmos-predict2.5 load VAE automatically
            self._impl = _OrigInterface(vae_pth=vae_pth, temporal_window=temporal_window)
        else:
            # Create interface without loading weights
            self._impl = _OrigInterface(vae_pth=None, temporal_window=temporal_window)
            # Store path for later manual loading
            self._vae_pth = vae_pth

    def load_state_dict(self, state_dict, strict: bool = True):
        """Load VAE weights manually."""
        missing, unexpected = self._impl.model.load_state_dict(state_dict, strict=strict)
        print(f"[VAE] Loaded weights from checkpoint")
        if missing:
            print(f"[VAE] Missing keys ({len(missing)}): {missing[:5]}{'...' if len(missing) > 5 else ''}")
        if unexpected:
            print(f"[VAE] Unexpected keys ({len(unexpected)}): {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")
        return missing, unexpected

    @property
    def spatial_compression_factor(self) -> int:
        return self._impl.spatial_compression_factor

    @property
    def temporal_compression_factor(self) -> int:
        return self._impl.temporal_compression_factor

    @property
    def latent_ch(self) -> int:
        return self._impl.latent_ch

    def encode(self, state: torch.Tensor) -> torch.Tensor:
        return self._impl.encode(state)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        return self._impl.decode(latent)
