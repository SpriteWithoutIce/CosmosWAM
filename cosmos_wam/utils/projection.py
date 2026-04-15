import json
import numpy as np
import torch


def euler_to_rotation_matrix(roll, pitch, yaw):
    """欧拉角 (roll, pitch, yaw) → 旋转矩阵，ZYX顺序。"""
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    R = np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp,     cp * sr,                cp * cr               ]
    ])
    return R


def euler_to_rotation_matrix_batch(rpy):
    """批量欧拉角 → 旋转矩阵。
    Args:
        rpy: [N, 3]
    Returns:
        R: [N, 3, 3]
    """
    N = rpy.shape[0]
    R = np.zeros((N, 3, 3), dtype=rpy.dtype)
    for i in range(N):
        R[i] = euler_to_rotation_matrix(rpy[i, 0], rpy[i, 1], rpy[i, 2])
    return R


def compute_pose_keypoints(eef_pos, eef_rpy, axis_length=0.1):
    """计算夹爪坐标系的 4 个关键点。
    Args:
        eef_pos: [N, 3] 或 [3]
        eef_rpy: [N, 3] 或 [3]
        axis_length: 坐标轴长度（米）
    Returns:
        points: [N, 4, 3] 或 [4, 3]
            - 点 0: 原点
            - 点 1: X 轴端点
            - 点 2: Y 轴端点
            - 点 3: Z 轴端点
    """
    if eef_pos.ndim == 1:
        R = euler_to_rotation_matrix(eef_rpy[0], eef_rpy[1], eef_rpy[2])
        x_axis = R[:, 0]
        y_axis = R[:, 1]
        z_axis = R[:, 2]
        p_origin = eef_pos
        p_x = eef_pos + axis_length * x_axis
        p_y = eef_pos + axis_length * y_axis
        p_z = eef_pos + axis_length * z_axis
        return np.stack([p_origin, p_x, p_y, p_z], axis=0)  # [4, 3]
    else:
        N = eef_pos.shape[0]
        R = euler_to_rotation_matrix_batch(eef_rpy)  # [N, 3, 3]
        x_axis = R[:, :, 0]  # [N, 3]
        y_axis = R[:, :, 1]
        z_axis = R[:, :, 2]
        p_origin = eef_pos
        p_x = eef_pos + axis_length * x_axis
        p_y = eef_pos + axis_length * y_axis
        p_z = eef_pos + axis_length * z_axis
        return np.stack([p_origin, p_x, p_y, p_z], axis=1)  # [N, 4, 3]


def project_world_to_pixel(point_3d, intrinsic, extrinsic, image_height, image_width):
    """Project 3D world points to 2D pixel coordinates.
    
    Args:
        point_3d: [..., 3] tensor or array of 3D points
        intrinsic: [3, 3] camera intrinsic matrix
        extrinsic: [4, 4] camera-to-world transformation
        image_height: int
        image_width: int
    
    Returns:
        pixel_uv: [..., 2] tensor of (u, v) pixel coordinates
    """
    is_numpy = not torch.is_tensor(point_3d)
    if is_numpy:
        device = torch.device("cpu")
        dtype = torch.float32
        point_3d = torch.as_tensor(point_3d, device=device, dtype=dtype)
        intrinsic = torch.as_tensor(intrinsic, device=device, dtype=dtype)
        extrinsic = torch.as_tensor(extrinsic, device=device, dtype=dtype)
    else:
        device = point_3d.device
        dtype = point_3d.dtype
        intrinsic = torch.as_tensor(intrinsic, device=device, dtype=torch.float32)
        extrinsic = torch.as_tensor(extrinsic, device=device, dtype=torch.float32)

    orig_shape = point_3d.shape
    point_3d = point_3d.reshape(-1, 3).float()
    B = point_3d.shape[0]

    world_to_camera = torch.inverse(extrinsic)
    point_homo = torch.cat([point_3d, torch.ones(B, 1, device=device)], dim=1)
    point_camera = (world_to_camera @ point_homo.unsqueeze(-1)).squeeze(-1)

    x_c, y_c, z_c = point_camera[:, 0], point_camera[:, 1], point_camera[:, 2]
    z_abs = z_c.abs().clamp(min=1e-6)
    x_norm = x_c / z_abs
    y_norm = y_c / z_abs

    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    cx, cy = intrinsic[0, 2], intrinsic[1, 2]

    u = fx * x_norm + cx
    v = fy * y_norm + cy

    # Horizontal flip fix from libero_heatmap_projection.py
    u = image_width - 1 - u

    pixel_uv = torch.stack([u, v], dim=1).to(dtype=dtype).reshape(*orig_shape[:-1], 2)
    if is_numpy:
        pixel_uv = pixel_uv.cpu().numpy()
    return pixel_uv


def generate_gaussian_heatmap(u, v, height, width, sigma):
    """Generate 2D Gaussian heatmap.
    
    Args:
        u, v: scalar or tensor of center coordinates
        height, width: int
        sigma: float
    
    Returns:
        heatmap: [B, 1, height, width] tensor or [1, height, width] array
    """
    is_numpy = not (torch.is_tensor(u) or torch.is_tensor(v))
    if is_numpy:
        u = float(u)
        v = float(v)
        y_grid, x_grid = np.mgrid[0:height, 0:width].astype(np.float32)
        dist_sq = (x_grid - u) ** 2 + (y_grid - v) ** 2
        heatmap = np.exp(-dist_sq / (2 * sigma * sigma))
        return heatmap.astype(np.float32)
    else:
        if not torch.is_tensor(u):
            u = torch.as_tensor(u)
        if not torch.is_tensor(v):
            v = torch.as_tensor(v)
        device = u.device
        y_grid, x_grid = torch.meshgrid(
            torch.arange(height, device=device, dtype=torch.float32),
            torch.arange(width, device=device, dtype=torch.float32),
            indexing="ij",
        )
        if u.dim() == 0:
            u = u.unsqueeze(0)
            v = v.unsqueeze(0)
        B = u.shape[0]
        dist_sq = (x_grid[None, :, :] - u[:, None, None]) ** 2 + (
            y_grid[None, :, :] - v[:, None, None]) ** 2
        heatmap = torch.exp(-dist_sq / (2 * sigma * sigma))
        return heatmap.unsqueeze(1)
