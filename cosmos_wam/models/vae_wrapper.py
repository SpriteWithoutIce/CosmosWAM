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
                      If False, VAE will be loaded later via load_vae_weights.
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
            # Create interface without loading weights (pass empty string or None)
            # cosmos-predict2.5 will create the model structure but without weights
            self._impl = _OrigInterface(vae_pth=None, temporal_window=temporal_window)
            self._vae_pth = vae_pth
    
    def to(self, *args, **kwargs):
        """Override to() to also move the internal implementation."""
        super().to(*args, **kwargs)
        if hasattr(self, '_impl') and self._impl is not None:
            self._impl = self._impl.to(*args, **kwargs)
        return self

    def load_vae_weights(self, ckpt_path: str):
        """Load VAE weights from checkpoint."""
        import torch
        state_dict = torch.load(ckpt_path, map_location="cpu")
        
        # cosmos-predict2.5's WanVAE uses assign=True for loading
        # We need to load into self._impl.model
        if hasattr(self._impl, 'model'):
            # Try to load directly - WanVAE in cosmos-predict2.5 uses a custom load
            missing_keys, unexpected_keys = [], []
            
            # Get the model state dict
            model_state = self._impl.model.state_dict() if hasattr(self._impl.model, 'state_dict') else None
            
            if model_state is None:
                # WanVAE might not have standard state_dict, try direct assignment
                # The checkpoint might have keys that need to be mapped
                for k, v in state_dict.items():
                    if hasattr(self._impl.model, k):
                        setattr(self._impl.model, k, v)
                    else:
                        # Try to find matching attribute
                        missing_keys.append(k)
                
                if missing_keys:
                    print(f"[VAE] Warning: {len(missing_keys)} keys not found in model")
            else:
                # Standard state dict loading
                missing_keys, unexpected_keys = self._impl.model.load_state_dict(state_dict, strict=False)
            
            print(f"[VAE] Loaded weights from {ckpt_path}")
            if missing_keys:
                print(f"[VAE] Missing keys ({len(missing_keys)}): {missing_keys[:5]}{'...' if len(missing_keys) > 5 else ''}")
            if unexpected_keys:
                print(f"[VAE] Unexpected keys ({len(unexpected_keys)}): {unexpected_keys[:5]}{'...' if len(unexpected_keys) > 5 else ''}")
        else:
            raise AttributeError("VAE implementation does not have 'model' attribute")

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
        # VAE is frozen and should stay on the device it was initialized on
        # Just ensure input is on the same device as VAE
        vae_device = next(self._impl.model.parameters()).device if hasattr(self._impl.model, 'parameters') else state.device
        state = state.to(vae_device)
        return self._impl.encode(state)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        # VAE is frozen and should stay on the device it was initialized on
        vae_device = next(self._impl.model.parameters()).device if hasattr(self._impl.model, 'parameters') else latent.device
        latent = latent.to(vae_device)
        return self._impl.decode(latent)
