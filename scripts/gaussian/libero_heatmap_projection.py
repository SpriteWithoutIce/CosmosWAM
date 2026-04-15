#!/usr/bin/env python3
"""
LIBERO End-Effector Heatmap Generation Pipeline (6DoF Pose Version)
====================================================================

支持两种模式：
1. 单点模式：只用 xyz 位置，生成 1 通道 heatmap
2. 4点坐标系模式：用 xyz + rpy，生成 4 通道 heatmap（原点 + 3 个轴端点）

用法：
  # Step 1: 提取相机参数（只需运行一次）
  python libero_heatmap_projection_pose.py extract_camera_params \
      --task_suite libero_spatial --output camera_params.json
  
  # Step 2: 对所有数据生成 4 点 heatmap
  python libero_heatmap_projection_pose.py generate_heatmaps \
      --dataset_dir /path/to/libero_spatial_no_noops_lerobot \
      --camera_params camera_params.json \
      --output_dir /path/to/heatmap_output \
      --mode pose  # 或 --mode position（单点模式）
  
  # Step 3: 可视化验证
  python libero_heatmap_projection_pose.py visualize_single \
      --parquet_path /path/to/episode_000000.parquet \
      --video_path /path/to/episode_000000.mp4 \
      --camera_params camera_params.json \
      --output visualization.mp4 \
      --mode pose
"""

import json
import math
import numpy as np
import torch
from pathlib import Path
from typing import Tuple, Optional, Dict, List
import argparse


# ============================================================================
# Part 1: 从LIBERO环境提取相机参数
# ============================================================================

def extract_camera_params_from_libero(
    task_suite_name: str = "libero_spatial",
    task_id: int = 0,
    camera_name: str = "agentview",
    image_height: int = 256,
    image_width: int = 256,
) -> Dict:
    """
    启动一个LIBERO环境，提取相机内外参。
    """
    from robosuite.utils import camera_utils as CU
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    import pathlib

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[task_suite_name]()
    task = task_suite.get_task(task_id)
    
    task_bddl_file = (
        pathlib.Path(get_libero_path("bddl_files"))
        / task.problem_folder
        / task.bddl_file
    )
    
    env_args = {
        "bddl_file_name": str(task_bddl_file),
        "camera_heights": image_height,
        "camera_widths": image_width,
    }
    env = OffScreenRenderEnv(**env_args)
    env.seed(42)
    env.reset()
    
    K = CU.get_camera_intrinsic_matrix(
        env.sim, camera_name, image_height, image_width
    )
    extrinsic = CU.get_camera_extrinsic_matrix(env.sim, camera_name)
    cam_id = env.sim.model.camera_name2id(camera_name)
    fovy = float(env.sim.model.cam_fovy[cam_id])
    
    print(f"\n{'='*60}")
    print(f"Camera: {camera_name}")
    print(f"Image size: {image_width}x{image_height}")
    print(f"FoVY: {fovy}")
    print(f"\nIntrinsic K:\n{K}")
    print(f"\nExtrinsic (camera-to-world, 4x4):\n{extrinsic}")
    print(f"{'='*60}\n")
    
    env.close()
    
    return {
        "intrinsic": K.tolist(),
        "extrinsic": extrinsic.tolist(),
        "fovy": fovy,
        "image_height": image_height,
        "image_width": image_width,
        "camera_name": camera_name,
    }


# ============================================================================
# Part 2: 欧拉角 → 旋转矩阵
# ============================================================================

def euler_to_rotation_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """
    欧拉角 (roll, pitch, yaw) → 旋转矩阵
    使用 ZYX 顺序（yaw → pitch → roll）
    
    Args:
        roll, pitch, yaw: 弧度
    
    Returns:
        R: [3, 3] 旋转矩阵
    """
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    
    # R = Rz(yaw) @ Ry(pitch) @ Rx(roll)
    R = np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp,     cp * sr,                cp * cr               ]
    ])
    
    return R


def euler_to_rotation_matrix_batch(rpy: np.ndarray) -> np.ndarray:
    """
    批量欧拉角 → 旋转矩阵
    
    Args:
        rpy: [N, 3] 或 [3]
    
    Returns:
        R: [N, 3, 3] 或 [3, 3]
    """
    if rpy.ndim == 1:
        return euler_to_rotation_matrix(rpy[0], rpy[1], rpy[2])
    
    N = rpy.shape[0]
    R = np.zeros((N, 3, 3))
    for i in range(N):
        R[i] = euler_to_rotation_matrix(rpy[i, 0], rpy[i, 1], rpy[i, 2])
    return R


# ============================================================================
# Part 3: 计算坐标系关键点
# ============================================================================

def compute_pose_keypoints(
    eef_pos: np.ndarray,
    eef_rpy: np.ndarray,
    axis_length: float = 0.1,
) -> np.ndarray:
    """
    计算夹爪坐标系的 4 个关键点
    
    Args:
        eef_pos: [3] 或 [N, 3] 末端执行器位置
        eef_rpy: [3] 或 [N, 3] 末端执行器姿态 (roll, pitch, yaw)
        axis_length: 坐标轴长度（米）
    
    Returns:
        points: [4, 3] 或 [N, 4, 3]
            - 点 0: 原点 (eef_pos)
            - 点 1: X 轴端点 (前方)
            - 点 2: Y 轴端点 (左侧)
            - 点 3: Z 轴端点 (上方)
    """
    if eef_pos.ndim == 1:
        # 单个点
        R = euler_to_rotation_matrix(eef_rpy[0], eef_rpy[1], eef_rpy[2])
        
        x_axis = R[:, 0]  # 前方
        y_axis = R[:, 1]  # 左侧
        z_axis = R[:, 2]  # 上方
        
        p_origin = eef_pos
        p_x = eef_pos + axis_length * x_axis
        p_y = eef_pos + axis_length * y_axis
        p_z = eef_pos + axis_length * z_axis
        
        return np.stack([p_origin, p_x, p_y, p_z], axis=0)  # [4, 3]
    else:
        # 批量
        N = eef_pos.shape[0]
        R = euler_to_rotation_matrix_batch(eef_rpy)  # [N, 3, 3]
        
        x_axis = R[:, :, 0]  # [N, 3]
        y_axis = R[:, :, 1]  # [N, 3]
        z_axis = R[:, :, 2]  # [N, 3]
        
        p_origin = eef_pos
        p_x = eef_pos + axis_length * x_axis
        p_y = eef_pos + axis_length * y_axis
        p_z = eef_pos + axis_length * z_axis
        
        return np.stack([p_origin, p_x, p_y, p_z], axis=1)  # [N, 4, 3]


# ============================================================================
# Part 4: 3D → 2D 投影
# ============================================================================

def project_world_to_pixel(
    point_3d: np.ndarray,
    intrinsic: np.ndarray,
    extrinsic: np.ndarray,
    image_height: int,
    image_width: int,
    verbose: bool = False,
) -> Tuple[float, float, bool]:
    """
    将单个 3D 点投影到 2D 像素坐标
    """
    world_to_camera = np.linalg.inv(extrinsic)
    
    point_homo = np.array([point_3d[0], point_3d[1], point_3d[2], 1.0])
    point_camera = world_to_camera @ point_homo
    
    x_c, y_c, z_c = point_camera[0], point_camera[1], point_camera[2]
    
    z_abs = abs(z_c)
    if z_abs < 1e-6:
        return 0.0, 0.0, False
    
    x_norm = x_c / z_abs
    y_norm = y_c / z_abs
    
    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    cx, cy = intrinsic[0, 2], intrinsic[1, 2]
    
    u = fx * x_norm + cx
    v = fy * y_norm + cy
    
    # 水平翻转（LIBERO 特有）
    u = image_width - 1 - u
    
    is_in_image = (0 <= u < image_width) and (0 <= v < image_height)
    
    return float(u), float(v), is_in_image


def project_points_batch(
    points_3d: np.ndarray,
    intrinsic: np.ndarray,
    extrinsic: np.ndarray,
    image_height: int,
    image_width: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    批量投影多个 3D 点到 2D
    
    Args:
        points_3d: [N, 3] 或 [N, M, 3]
        
    Returns:
        pixels: [N, 2] 或 [N, M, 2]
        visible: [N] 或 [N, M] bool
    """
    original_shape = points_3d.shape
    
    if points_3d.ndim == 3:
        N, M, _ = points_3d.shape
        points_flat = points_3d.reshape(-1, 3)
    else:
        points_flat = points_3d
        N = points_flat.shape[0]
        M = None
    
    world_to_camera = np.linalg.inv(extrinsic)
    
    # 齐次坐标
    ones = np.ones((points_flat.shape[0], 1))
    points_homo = np.concatenate([points_flat, ones], axis=1)  # [N*M, 4]
    
    # 变换到相机坐标系
    points_camera = (world_to_camera @ points_homo.T).T  # [N*M, 4]
    
    x_c = points_camera[:, 0]
    y_c = points_camera[:, 1]
    z_c = points_camera[:, 2]
    
    z_abs = np.abs(z_c)
    z_abs = np.maximum(z_abs, 1e-6)
    
    x_norm = x_c / z_abs
    y_norm = y_c / z_abs
    
    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    cx, cy = intrinsic[0, 2], intrinsic[1, 2]
    
    u = fx * x_norm + cx
    v = fy * y_norm + cy
    
    # 水平翻转
    u = image_width - 1 - u
    
    # 可见性
    visible = (u >= 0) & (u < image_width) & (v >= 0) & (v < image_height)
    
    pixels = np.stack([u, v], axis=1)
    
    if M is not None:
        pixels = pixels.reshape(N, M, 2)
        visible = visible.reshape(N, M)
    
    return pixels, visible


# ============================================================================
# Part 5: 生成高斯 Heatmap
# ============================================================================

def generate_gaussian_heatmap(
    u: float,
    v: float,
    height: int,
    width: int,
    sigma: float = 5.0,
) -> np.ndarray:
    """
    在 (u, v) 位置生成 2D Gaussian heatmap
    """
    y_grid, x_grid = np.mgrid[0:height, 0:width].astype(np.float32)
    dist_sq = (x_grid - u) ** 2 + (y_grid - v) ** 2
    heatmap = np.exp(-dist_sq / (2 * sigma * sigma))
    return heatmap.astype(np.float32)


def generate_multi_point_heatmap(
    pixels: np.ndarray,
    height: int,
    width: int,
    sigma: float = 5.0,
) -> np.ndarray:
    """
    为多个点生成多通道 heatmap
    
    Args:
        pixels: [N, 2] N 个点的 (u, v) 坐标
        
    Returns:
        heatmap: [N, height, width]
    """
    N = pixels.shape[0]
    heatmaps = np.zeros((N, height, width), dtype=np.float32)
    
    for i in range(N):
        heatmaps[i] = generate_gaussian_heatmap(
            pixels[i, 0], pixels[i, 1], height, width, sigma
        )
    
    return heatmaps


# ============================================================================
# Part 6: 处理 LeRobot 格式数据
# ============================================================================

def process_lerobot_episode_pose(
    parquet_path: str,
    intrinsic: np.ndarray,
    extrinsic: np.ndarray,
    image_height: int = 256,
    image_width: int = 256,
    heatmap_height: int = 224,
    heatmap_width: int = 224,
    sigma: float = 5.0,
    axis_length: float = 0.05,
    mode: str = "pose",  # "pose" 或 "position"
) -> List[Dict]:
    """
    处理一个 episode，生成 4 点投影信息
    
    Args:
        mode: 
            - "pose": 使用 xyz + rpy，生成 4 个点
            - "position": 只使用 xyz，生成 1 个点
    
    Returns:
        list of dicts，每帧包含:
            - frame_idx
            - state: [8] 完整 state
            - points_3d: [4, 3] 或 [1, 3] 关键点 3D 坐标
            - points_2d: [4, 2] 或 [1, 2] 关键点 2D 像素坐标
            - points_2d_heatmap: [4, 2] 或 [1, 2] heatmap 分辨率下的坐标
            - visible: [4] 或 [1] 可见性
    """
    import pandas as pd
    
    df = pd.read_parquet(parquet_path)
    
    # 找 state 列
    state_col = None
    for col in df.columns:
        if "state" in col.lower():
            state_col = col
            break
    
    if state_col is None:
        raise ValueError(f"No state column found. Columns: {df.columns.tolist()}")
    
    results = []
    
    for idx, row in df.iterrows():
        state = np.array(row[state_col])
        
        eef_pos = state[0:3]  # xyz
        eef_rpy = state[3:6]  # roll, pitch, yaw
        
        if mode == "pose":
            # 4 点坐标系
            points_3d = compute_pose_keypoints(eef_pos, eef_rpy, axis_length)  # [4, 3]
        else:
            # 单点
            points_3d = eef_pos.reshape(1, 3)  # [1, 3]
        
        # 投影到 2D
        points_2d, visible = project_points_batch(
            points_3d, intrinsic, extrinsic, image_height, image_width
        )
        
        # 缩放到 heatmap 分辨率
        scale_x = heatmap_width / image_width
        scale_y = heatmap_height / image_height
        points_2d_heatmap = points_2d.copy()
        points_2d_heatmap[:, 0] *= scale_x
        points_2d_heatmap[:, 1] *= scale_y
        
        results.append({
            "frame_idx": int(row.get("frame_index", idx)),
            "state": state.tolist(),
            "points_3d": points_3d.tolist(),
            "points_2d": points_2d.tolist(),
            "points_2d_heatmap": points_2d_heatmap.tolist(),
            "visible": visible.tolist(),
        })
    
    # 打印统计
    all_visible = np.array([r["visible"] for r in results])
    if mode == "pose":
        print(f"  Processed {len(results)} frames (4-point pose mode)")
        print(f"  Visibility per point:")
        point_names = ["Origin", "X-axis", "Y-axis", "Z-axis"]
        for i, name in enumerate(point_names):
            vis_count = all_visible[:, i].sum()
            print(f"    {name}: {vis_count}/{len(results)} ({100*vis_count/len(results):.1f}%)")
    else:
        print(f"  Processed {len(results)} frames (single-point position mode)")
        vis_count = all_visible.sum()
        print(f"  Visibility: {vis_count}/{len(results)} ({100*vis_count/len(results):.1f}%)")
    
    return results


def generate_heatmap_video_pose(
    projections: List[Dict],
    heatmap_height: int = 224,
    heatmap_width: int = 224,
    sigma: float = 5.0,
    mode: str = "pose",
    low_res: int = 16
) -> np.ndarray:
    """
    为整个 episode 生成 heatmap 视频
    
    Returns:
        heatmaps: [T, C, H, W] 其中 C=4 (pose) 或 C=1 (position)
    """
    T = len(projections)
    C = 4 if mode == "pose" else 1
    
    heatmaps = np.zeros((T, C, heatmap_height, heatmap_width), dtype=np.float32)
    
    for t, proj in enumerate(projections):
        points = np.array(proj["points_2d_heatmap"])  # [C, 2]
        visible = np.array(proj["visible"])  # [C]
        
        for c in range(C):
            if visible[c]:
                heatmaps[t, c] = generate_gaussian_heatmap(
                    points[c, 0], points[c, 1],
                    heatmap_height, heatmap_width,
                    sigma=sigma,
                )
    # 额外生成低分辨率版本
    heatmaps_low = np.zeros((T, C, low_res, low_res), dtype=np.float32)
    
    for t, proj in enumerate(projections):
        points = np.array(proj["points_2d_heatmap"])  # [C, 2]
        visible = np.array(proj["visible"])
        
        # 缩放坐标到 16×16
        scale = low_res / heatmap_width  # 16 / 224
        points_low = points * scale
        
        # sigma 也要缩放
        sigma_low = sigma * scale  # 8.0 * (16/224) ≈ 0.57
        
        for c in range(C):
            if visible[c]:
                heatmaps_low[t, c] = generate_gaussian_heatmap(
                    points_low[c, 0], points_low[c, 1],
                    low_res, low_res,
                    sigma=sigma_low,
                )
                
    return heatmaps, heatmaps_low


# ============================================================================
# Part 7: 可视化
# ============================================================================

def visualize_single_episode_pose(
    parquet_path: str,
    video_path: str,
    camera_params_path: str,
    output_path: str,
    heatmap_height: int = 224,
    heatmap_width: int = 224,
    sigma: float = 8.0,
    alpha: float = 0.5,
    axis_length: float = 0.05,
    mode: str = "pose",
):
    """
    可视化单个 episode 的 4 点投影
    """
    import cv2
    import torchvision
    from tqdm import tqdm
    
    # 加载相机参数
    with open(camera_params_path, "r") as f:
        params = json.load(f)
    
    intrinsic = np.array(params["intrinsic"])
    extrinsic = np.array(params["extrinsic"])
    render_h = params["image_height"]
    render_w = params["image_width"]
    
    # 处理投影
    print(f"Processing parquet: {parquet_path}")
    projections = process_lerobot_episode_pose(
        parquet_path, intrinsic, extrinsic,
        render_h, render_w,
        heatmap_height, heatmap_width,
        sigma, axis_length, mode
    )
    
    # 读取视频
    print(f"Opening video: {video_path}")
    torchvision.set_video_backend("pyav")
    reader = torchvision.io.VideoReader(video_path, "video")
    
    fps = reader.get_metadata()["video"]["fps"][0]
    video_info = reader.get_metadata()["video"]
    video_width = video_info.get("width", [512])[0]
    video_height = video_info.get("height", [512])[0]
    
    print(f"Video: {video_width}x{video_height} @ {fps}fps")
    
    # 颜色定义（BGR）
    colors = {
        "origin": (0, 255, 255),    # 黄色 - 原点
        "x_axis": (0, 0, 255),      # 红色 - X轴（前方）
        "y_axis": (0, 255, 0),      # 绿色 - Y轴（左侧）
        "z_axis": (255, 0, 0),      # 蓝色 - Z轴（上方）
    }
    point_names = ["origin", "x_axis", "y_axis", "z_axis"]
    
    frames_to_save = []
    frame_idx = 0
    pbar = tqdm(total=len(projections), desc="Generating video")
    
    for frame in reader:
        if frame_idx >= len(projections):
            break
        
        frame_tensor = frame["data"]
        frame_rgb = frame_tensor.permute(1, 2, 0).numpy()
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        
        proj = projections[frame_idx]
        points_2d = np.array(proj["points_2d"])  # [4, 2] 或 [1, 2]
        visible = np.array(proj["visible"])
        
        # 绘制点和连线
        if mode == "pose" and visible.all():
            # 缩放到视频分辨率
            scale_x = video_width / render_w
            scale_y = video_height / render_h
            pts = points_2d.copy()
            pts[:, 0] *= scale_x
            pts[:, 1] *= scale_y
            pts = pts.astype(int)
            
            origin = tuple(pts[0])
            p_x = tuple(pts[1])
            p_y = tuple(pts[2])
            p_z = tuple(pts[3])
            
            # 画坐标轴线
            cv2.line(frame_bgr, origin, p_x, colors["x_axis"], 2)  # X轴 红
            cv2.line(frame_bgr, origin, p_y, colors["y_axis"], 2)  # Y轴 绿
            cv2.line(frame_bgr, origin, p_z, colors["z_axis"], 2)  # Z轴 蓝
            
            # 画点
            cv2.circle(frame_bgr, origin, 6, colors["origin"], -1)
            cv2.circle(frame_bgr, p_x, 4, colors["x_axis"], -1)
            cv2.circle(frame_bgr, p_y, 4, colors["y_axis"], -1)
            cv2.circle(frame_bgr, p_z, 4, colors["z_axis"], -1)
            
            # 标签
            cv2.putText(frame_bgr, "O", (origin[0]+5, origin[1]-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, colors["origin"], 1)
            cv2.putText(frame_bgr, "X", (p_x[0]+5, p_x[1]-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, colors["x_axis"], 1)
            cv2.putText(frame_bgr, "Y", (p_y[0]+5, p_y[1]-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, colors["y_axis"], 1)
            cv2.putText(frame_bgr, "Z", (p_z[0]+5, p_z[1]-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, colors["z_axis"], 1)
        
        elif mode == "position" and visible[0]:
            scale_x = video_width / render_w
            scale_y = video_height / render_h
            pt = points_2d[0].copy()
            pt[0] *= scale_x
            pt[1] *= scale_y
            pt = pt.astype(int)
            cv2.circle(frame_bgr, tuple(pt), 6, colors["origin"], -1)
        
        # 显示 state 信息
        state = proj["state"]
        info_text = f"xyz: ({state[0]:.3f}, {state[1]:.3f}, {state[2]:.3f})"
        cv2.putText(frame_bgr, info_text, (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        if mode == "pose":
            rpy_text = f"rpy: ({state[3]:.2f}, {state[4]:.2f}, {state[5]:.2f})"
            cv2.putText(frame_bgr, rpy_text, (10, 45),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        frames_to_save.append(frame_bgr)
        frame_idx += 1
        pbar.update(1)

        # 在 frame_bgr 上叠加一个放大后的 16×16 heatmap 预览

        # 生成当前帧的 16×16 heatmap
        points_low = np.array(proj["points_2d_heatmap"]) * (16 / 224)
        sigma_low = sigma * (16 / 224)

        # 4 个点各用一个颜色通道
        colors_rgb = [
            [255, 255, 0],   # 黄色 - 原点
            [255, 0, 0],     # 红色 - X轴
            [0, 255, 0],     # 绿色 - Y轴
            [0, 0, 255],     # 蓝色 - Z轴
        ]

        heatmap_color = np.zeros((16, 16, 3), dtype=np.float32)
        for c in range(4):
            if visible[c]:
                h = generate_gaussian_heatmap(points_low[c, 0], points_low[c, 1], 16, 16, sigma_low)
                for ch in range(3):
                    heatmap_color[:, :, ch] += h * colors_rgb[c][ch] / 255.0

        heatmap_color = np.clip(heatmap_color, 0, 1)
        heatmap_vis = (heatmap_color * 255).astype(np.uint8)
        heatmap_vis = cv2.resize(heatmap_vis, (128, 128), interpolation=cv2.INTER_NEAREST)

        # 注意：colors_rgb 是 RGB 顺序，OpenCV 需要 BGR
        heatmap_vis_bgr = cv2.cvtColor(heatmap_vis, cv2.COLOR_RGB2BGR)

        # 贴到画面右下角
        frame_bgr[-138:-10, -138:-10] = heatmap_vis_bgr
    
    pbar.close()
    reader.container.close()
    
    # 保存视频
    print(f"\nSaving video to: {output_path}")
    
    try:
        import torchvision.io as io
        frames_array = np.stack(frames_to_save)
        frames_rgb = frames_array[..., ::-1].copy()
        frames_tensor = torch.from_numpy(frames_rgb)
        io.write_video(output_path, frames_tensor, fps, video_codec='libx264')
        print("Saved using torchvision")
    except Exception as e:
        print(f"torchvision failed: {e}, trying imageio...")
        import imageio
        writer = imageio.get_writer(output_path, fps=fps, codec='libx264', quality=8)
        for frame in frames_to_save:
            writer.append_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        writer.close()
        print("Saved using imageio")
    
    print(f"Total frames: {frame_idx}")


# ============================================================================
# Part 8: 预计算所有 heatmap
# ============================================================================

def precompute_all_heatmaps_pose(
    dataset_dir: str,
    camera_params_path: str,
    output_dir: str,
    heatmap_height: int = 224,
    heatmap_width: int = 224,
    sigma: float = 8.0,
    axis_length: float = 0.05,
    mode: str = "pose",
):
    """
    为整个数据集预计算 heatmap
    
    输出:
        - episode_XXXXXX_heatmaps.npy: [T, C, H, W] 其中 C=4 (pose) 或 C=1 (position)
        - episode_XXXXXX_projections.json: 投影元数据
    """
    with open(camera_params_path, "r") as f:
        params = json.load(f)
    
    intrinsic = np.array(params["intrinsic"])
    extrinsic = np.array(params["extrinsic"])
    render_h = params["image_height"]
    render_w = params["image_width"]
    
    dataset_path = Path(dataset_dir)
    parquet_files = sorted(dataset_path.glob("data/chunk-*/episode_*.parquet"))
    
    if not parquet_files:
        print(f"No parquet files found in {dataset_dir}/data/chunk-*/")
        return
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Processing {len(parquet_files)} episodes...")
    print(f"Mode: {mode} ({'4 points' if mode == 'pose' else '1 point'})")
    
    for pq_file in parquet_files:
        episode_name = pq_file.stem
        print(f"\nProcessing {episode_name}...")
        
        # 投影
        projections = process_lerobot_episode_pose(
            str(pq_file), intrinsic, extrinsic,
            render_h, render_w,
            heatmap_height, heatmap_width,
            sigma, axis_length, mode
        )
        
        # 生成 heatmap
        heatmaps = generate_heatmap_video_pose(
            projections, heatmap_height, heatmap_width, sigma, mode
        )
        
        # 保存
        np.save(str(output_path / f"{episode_name}_heatmaps.npy"), heatmaps)
        
        with open(str(output_path / f"{episode_name}_projections.json"), "w") as f:
            json.dump(projections, f)
        
        print(f"  Saved: {heatmaps.shape}")
    
    print(f"\nDone! Output: {output_dir}")


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LIBERO Heatmap Generation (6DoF Pose)")
    subparsers = parser.add_subparsers(dest="command")
    
    # extract_camera_params
    extract_parser = subparsers.add_parser("extract_camera_params")
    extract_parser.add_argument("--task_suite", default="libero_spatial")
    extract_parser.add_argument("--task_id", type=int, default=0)
    extract_parser.add_argument("--camera_name", default="agentview")
    extract_parser.add_argument("--resolution", type=int, default=256)
    extract_parser.add_argument("--output", default="camera_params.json")
    
    # generate_heatmaps
    gen_parser = subparsers.add_parser("generate_heatmaps")
    gen_parser.add_argument("--dataset_dir", required=True)
    gen_parser.add_argument("--camera_params", required=True)
    gen_parser.add_argument("--output_dir", required=True)
    gen_parser.add_argument("--heatmap_height", type=int, default=224)
    gen_parser.add_argument("--heatmap_width", type=int, default=224)
    gen_parser.add_argument("--sigma", type=float, default=30.0)
    gen_parser.add_argument("--axis_length", type=float, default=0.15)
    gen_parser.add_argument("--mode", choices=["pose", "position"], default="pose")
    gen_parser.add_argument("--low_res", type=int, default=16, help="低分辨率 heatmap 尺寸")
    
    # visualize_single
    vis_parser = subparsers.add_parser("visualize_single")
    vis_parser.add_argument("--parquet_path", required=True)
    vis_parser.add_argument("--video_path", required=True)
    vis_parser.add_argument("--camera_params", required=True)
    vis_parser.add_argument("--output", required=True)
    vis_parser.add_argument("--heatmap_height", type=int, default=224)
    vis_parser.add_argument("--heatmap_width", type=int, default=224)
    vis_parser.add_argument("--sigma", type=float, default=30.0)
    vis_parser.add_argument("--alpha", type=float, default=0.5)
    vis_parser.add_argument("--axis_length", type=float, default=0.15)
    vis_parser.add_argument("--mode", choices=["pose", "position"], default="pose")
    
    args = parser.parse_args()
    
    if args.command == "extract_camera_params":
        params = extract_camera_params_from_libero(
            task_suite_name=args.task_suite,
            task_id=args.task_id,
            camera_name=args.camera_name,
            image_height=args.resolution,
            image_width=args.resolution,
        )
        with open(args.output, "w") as f:
            json.dump(params, f, indent=2)
        print(f"Saved to {args.output}")
    
    elif args.command == "generate_heatmaps":
        precompute_all_heatmaps_pose(
            dataset_dir=args.dataset_dir,
            camera_params_path=args.camera_params,
            output_dir=args.output_dir,
            heatmap_height=args.heatmap_height,
            heatmap_width=args.heatmap_width,
            sigma=args.sigma,
            axis_length=args.axis_length,
            mode=args.mode,
        )
    
    elif args.command == "visualize_single":
        visualize_single_episode_pose(
            parquet_path=args.parquet_path,
            video_path=args.video_path,
            camera_params_path=args.camera_params,
            output_path=args.output,
            heatmap_height=args.heatmap_height,
            heatmap_width=args.heatmap_width,
            sigma=args.sigma,
            alpha=args.alpha,
            axis_length=args.axis_length,
            mode=args.mode,
        )
    
    else:
        parser.print_help()