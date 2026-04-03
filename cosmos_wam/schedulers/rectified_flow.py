import math
import torch
import torch.nn.functional as F


class RectifiedFlowScheduler:
    def __init__(self, num_train_timesteps: int = 1000, shift: float = 1.0):
        self.num_train_timesteps = num_train_timesteps
        self.shift = shift

    def sample_train_time(self, batch_size: int, device: str = "cuda") -> torch.Tensor:
        """Sample continuous timestep in [0, 1]."""
        return torch.rand(batch_size, device=device, dtype=torch.float32)

    def get_sigmas(self, t: torch.Tensor) -> torch.Tensor:
        """Map [0,1] timestep to sigma for flow matching."""
        # Standard rectified flow: sigma = t
        return t

    def get_interpolation(self, noise: torch.Tensor, x0: torch.Tensor, sigmas: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute x_t = (1 - sigma) * noise + sigma * x0 and target velocity."""
        # Expand sigma to match noise/x0 shape
        while sigmas.ndim < noise.ndim:
            sigmas = sigmas.unsqueeze(-1)
        xt = (1.0 - sigmas) * noise + sigmas * x0
        vt = x0 - noise
        return xt, vt

    def training_target(self, x0: torch.Tensor, noise: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Velocity target for flow matching."""
        return x0 - noise

    def add_noise(self, x0: torch.Tensor, noise: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Forward diffusion: x_t = (1-t) * noise + t * x0."""
        while t.ndim < x0.ndim:
            t = t.unsqueeze(-1)
        return (1.0 - t) * noise + t * x0

    def step(self, pred_velocity: torch.Tensor, delta_t: torch.Tensor, xt: torch.Tensor) -> torch.Tensor:
        """Euler step for inference."""
        while delta_t.ndim < xt.ndim:
            delta_t = delta_t.unsqueeze(-1)
        return xt + delta_t * pred_velocity

    def train_time_weight(self, t: torch.Tensor, tensor_kwargs: dict) -> torch.Tensor:
        """Optional weighting for training loss (e.g., min-SNR can be applied externally)."""
        return torch.ones(t.shape[0], **tensor_kwargs)
