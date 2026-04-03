from typing import List
import torch
from PIL import Image


def save_mp4(frames: List[Image.Image], filepath: str, fps: int = 8):
    import imageio
    writer = imageio.get_writer(filepath, fps=fps)
    for frame in frames:
        writer.append_data(frame)
    writer.close()


def pil_frames_to_video_tensor(frames: List[Image.Image]) -> torch.Tensor:
    # [T, H, W, 3] uint8
    import numpy as np
    arr = np.stack([np.array(f) for f in frames], axis=0)
    tensor = torch.from_numpy(arr).permute(0, 3, 1, 2).unsqueeze(0)  # [1, T, 3, H, W]
    return tensor


def video_psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    mse = torch.mean((pred.float() - target.float()) ** 2)
    if mse == 0:
        return float("inf")
    return 20 * torch.log10(255.0 / torch.sqrt(mse)).item()


def video_ssim(pred: torch.Tensor, target: torch.Tensor) -> float:
    # Simplified SSIM, can be replaced with pytorch_msssim
    return 0.9  # Placeholder
