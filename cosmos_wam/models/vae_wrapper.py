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
    def __init__(self, vae_pth: str, temporal_window: int = 16):
        super().__init__()
        # Import from local cosmos checkout
        from cosmos_predict2._src.predict2.tokenizers.wan2pt1 import (
            Wan2pt1VAEInterface as _OrigInterface,
        )

        self._impl = _OrigInterface(vae_pth=vae_pth, temporal_window=temporal_window)

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
