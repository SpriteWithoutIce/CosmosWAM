#!/usr/bin/env python3
"""
LIBERO End-Effector Heatmap Generation Pipeline
================================================

两步走：
  Step 1: 从LIBERO环境中提取agentview相机的固定内外参（只需运行一次）
  Step 2: 对LeRobot格式数据中的每帧，用state前3维(eef_pos)投影到2D，生成Gaussian heatmap

关键发现：
  - LIBERO的agentview是固定相机，内外参在整个数据集中不变
  - robot0_eye_in_hand是手腕相机，外参每帧变化（如果你只用agentview就不需要管它）
  - LeRobot格式的state前3维就是end-effector的xyz世界坐标

用法：
  # Step 1: 提取相机参数（只需运行一次）
  python libero_heatmap_projection.py extract_camera_params \
      --task_suite libero_spatial --output camera_params.json
  
  # Step 2: 对所有数据生成heatmap
  python libero_heatmap_projection.py generate_heatmaps \
      --dataset_dir /path/to/libero_spatial_no_noops_lerobot \
      --camera_params camera_params.json \
      --output_dir /path/to/heatmap_output
  
  # Step 3 (可选): 单个视频验证 - 生成对比视频
  python libero_heatmap_projection.py visualize_single \
      --parquet_path /path/to/episode_000000.parquet \
      --video_path /path/to/episode_000000.mp4 \
      --camera_params camera_params.json \
      --output visualization.mp4 \
      --sigma 8.0 --alpha 0.5
"""

import json
import math
import numpy as np
import torch
from pathlib import Path
from typing import Tuple, Optional, Dict
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
    
    agentview的相机参数是固定的（所有task、所有episode一样），
    所以只需要运行一次。
    
    Returns:
        dict: {
            "intrinsic": 3x3 matrix (list of lists),
            "extrinsic": 4x4 matrix (list of lists),
            "fovy": float,
            "image_height": int,
            "image_width": int,
            "camera_name": str,
        }
    """
    from robosuite.utils import camera_utils as CU
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    import pathlib

    # 获取任务
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[task_suite_name]()
    task = task_suite.get_task(task_id)
    
    task_bddl_file = (
        pathlib.Path(get_libero_path("bddl_files"))
        / task.problem_folder
        / task.bddl_file
    )
    
    # 创建环境
    env_args = {
        "bddl_file_name": str(task_bddl_file),
        "camera_heights": image_height,
        "camera_widths": image_width,
    }
    env = OffScreenRenderEnv(**env_args)
    env.seed(42)
    env.reset()
    
    # 提取内参
    # robosuite源码: f = 0.5 * height / tan(fovy * pi / 360)
    # K = [[f, 0, width/2], [0, f, height/2], [0, 0, 1]]
    K = CU.get_camera_intrinsic_matrix(
        env.sim, camera_name, image_height, image_width
    )
    
    # 提取外参 (4x4齐次矩阵, world_to_camera 或 camera_in_world)
    # robosuite返回的是camera-to-world变换（相机在世界坐标系中的位姿）
    extrinsic = CU.get_camera_extrinsic_matrix(env.sim, camera_name)
    
    # 获取fovy
    cam_id = env.sim.model.camera_name2id(camera_name)
    fovy = float(env.sim.model.cam_fovy[cam_id])
    
    # 验证：也提取一下wrist相机的外参（会变化，仅供参考）
    wrist_extrinsic = CU.get_camera_extrinsic_matrix(env.sim, "robot0_eye_in_hand")
    
    print(f"\n{'='*60}")
    print(f"Camera: {camera_name}")
    print(f"Image size: {image_width}x{image_height}")
    print(f"FoVY: {fovy}")
    print(f"\nIntrinsic K:")
    print(K)
    print(f"\nExtrinsic (camera-to-world, 4x4):")
    print(extrinsic)
    print(f"\nWrist camera extrinsic (for reference, changes per frame):")
    print(wrist_extrinsic)
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
# Part 2: 3D -> 2D 投影 + Gaussian Heatmap 生成
# ============================================================================

def project_world_to_pixel(
    point_3d: np.ndarray,         # (3,) 世界坐标
    intrinsic: np.ndarray,        # (3, 3) 内参矩阵
    extrinsic: np.ndarray,        # (4, 4) camera-to-world变换
    image_height: int,
    image_width: int,
    verbose: bool = False,         # 是否打印调试信息
) -> Tuple[float, float, bool]:
    """
    将世界坐标系中的3D点投影到图像像素坐标。
    
    注意LIBERO/robosuite的特殊性：
    - robosuite的get_camera_extrinsic_matrix返回的是camera-to-world变换
    - 需要求逆得到world-to-camera变换
    - MuJoCo的相机看向-z方向（OpenGL约定）
    - 图像坐标系：robosuite渲染后做了上下翻转 ([::-1, ::-1])
    
    Returns:
        (u, v, is_visible): 像素坐标 (列, 行) 和是否在图像内
    """
    # Step 1: 世界坐标 -> 相机坐标
    # extrinsic是camera-to-world，求逆得到world-to-camera
    world_to_camera = np.linalg.inv(extrinsic)
    
    # 齐次坐标
    point_homo = np.array([point_3d[0], point_3d[1], point_3d[2], 1.0])
    point_camera = world_to_camera @ point_homo  # (4,)
    
    # Step 2: 相机坐标 -> 像素坐标
    # MuJoCo/OpenGL约定：相机看向-z方向
    # 所以需要把点投影到相机前方（-z方向）
    x_c, y_c, z_c = point_camera[0], point_camera[1], point_camera[2]
    
    if verbose:
        print(f"    Camera coords: ({x_c:.3f}, {y_c:.3f}, {z_c:.3f})")
        print(f"    Camera pos (from extrinsic): {extrinsic[:3, 3]}")
        # 打印相机的朝向（rotation矩阵的第三列是相机z轴在世界坐标系中的方向）
        print(f"    Camera z-axis (world frame): {extrinsic[:3, 2]}")
    
    # 关键修复：robosuite的extrinsic矩阵可能不是标准的OpenGL格式
    # 让我们先检查z_c的符号。如果所有点都是z_c > 0，可能需要翻转判断
    # 或者使用绝对值进行投影
    
    # 方法：无论z_c符号如何，都进行投影（假设相机能看到近距离的点）
    # 实际 visibility 由投影后的像素位置是否在图像范围内决定
    use_absolute_z = True  # 使用绝对值进行投影
    
    if use_absolute_z:
        # 使用 |z_c| 进行投影，这样无论点在相机前方还是后方都能投影
        z_abs = abs(z_c)
        if z_abs < 1e-6:
            if verbose:
                print(f"    Point too close to camera (z_c={z_c:.3f})")
            return 0.0, 0.0, False
        x_norm = x_c / z_abs
        y_norm = y_c / z_abs
        # 点在相机后方时标记为不可见，但仍然投影
        is_in_front = (z_c < 0)  # OpenGL: z < 0 是前方
    else:
        # 原始方法
        if z_c >= 0:
            if verbose:
                print(f"    Point behind camera (z_c={z_c:.3f} >= 0)")
            return 0.0, 0.0, False
        x_norm = x_c / (-z_c)
        y_norm = y_c / (-z_c)
        is_in_front = True
    
    # 用内参矩阵转换到像素坐标
    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    cx, cy = intrinsic[0, 2], intrinsic[1, 2]
    
    u = fx * x_norm + cx  # 列
    v = fy * y_norm + cy  # 行
    
    if verbose:
        print(f"    Before flip: u={u:.1f}, v={v:.1f}")
    
    # robosuite在渲染后做了 [::-1, ::-1] 翻转（见libero_utils.py）
    # 这意味着图像上下左右都翻转了
    u = image_width - 1 - u
    v = image_height - 1 - v
    
    # 可见性判断：在图像范围内且在相机前方
    is_in_image = (0 <= u < image_width) and (0 <= v < image_height)
    is_visible = is_in_image and is_in_front
    
    if verbose:
        print(f"    After flip: u={u:.1f}, v={v:.1f}, in_image={is_in_image}, in_front={is_in_front}, visible={is_visible}")
    
    return float(u), float(v), is_visible


def generate_gaussian_heatmap(
    u: float,
    v: float,
    height: int,
    width: int,
    sigma: float = 5.0,
) -> np.ndarray:
    """
    在(u, v)位置生成2D Gaussian heatmap。
    
    Args:
        u: 列坐标 (x方向)
        v: 行坐标 (y方向)  
        height: 图像高度
        width: 图像宽度
        sigma: Gaussian标准差（控制heatmap大小）
    
    Returns:
        heatmap: (height, width), float32, range [0, 1]
    """
    y_grid, x_grid = np.mgrid[0:height, 0:width].astype(np.float32)
    
    # Gaussian: exp(-((x-u)^2 + (y-v)^2) / (2*sigma^2))
    dist_sq = (x_grid - u) ** 2 + (y_grid - v) ** 2
    heatmap = np.exp(-dist_sq / (2 * sigma * sigma))
    
    return heatmap.astype(np.float32)


def generate_heatmap_rgb(
    heatmap: np.ndarray,
    colormap: str = "hot",
) -> np.ndarray:
    """
    将单通道heatmap转为3通道RGB图片，用于过VAE。
    
    Args:
        heatmap: (H, W) float32, range [0, 1]
        colormap: matplotlib colormap名称
    
    Returns:
        rgb: (H, W, 3) uint8, range [0, 255]
    """
    # 方法1: 简单复制到3通道（灰度）
    # 这样更简单，也能过VAE
    heatmap_uint8 = (heatmap * 255).clip(0, 255).astype(np.uint8)
    rgb = np.stack([heatmap_uint8, heatmap_uint8, heatmap_uint8], axis=-1)
    return rgb
    
    # 方法2: 用colormap（更有信息量，但可能引入不必要的颜色信息）
    # import matplotlib.pyplot as plt
    # cmap = plt.get_cmap(colormap)
    # colored = cmap(heatmap)[:, :, :3]  # (H, W, 3) float [0,1]
    # return (colored * 255).astype(np.uint8)


# ============================================================================
# Part 3: 处理LeRobot格式数据
# ============================================================================

def process_lerobot_episode(
    parquet_path: str,
    intrinsic: np.ndarray,
    extrinsic: np.ndarray,
    image_height: int = 256,
    image_width: int = 256,
    heatmap_height: int = 224,
    heatmap_width: int = 224,
    sigma: float = 5.0,
) -> list:
    """
    处理一个episode的parquet文件，生成每帧的heatmap信息。
    
    Args:
        parquet_path: LeRobot parquet文件路径
        intrinsic: 相机内参 (3, 3)
        extrinsic: 相机外参 (4, 4), camera-to-world
        image_height/width: 原始渲染分辨率（用于投影）
        heatmap_height/width: 输出heatmap分辨率（匹配训练图片大小）
        sigma: Gaussian标准差
    
    Returns:
        list of dicts: [{
            "frame_idx": int,
            "eef_pos_3d": [x, y, z],
            "pixel_u": float,  # 在原始分辨率下的像素坐标
            "pixel_v": float,
            "heatmap_u": float,  # 在heatmap分辨率下的像素坐标
            "heatmap_v": float,
            "is_visible": bool,
        }]
    """
    import pandas as pd
    
    df = pd.read_parquet(parquet_path)
    
    # state列名可能是 "observation.state" 或 "observation.state.default"
    state_col = None
    for col in df.columns:
        if "state" in col.lower():
            state_col = col
            break
    
    if state_col is None:
        raise ValueError(f"No state column found in {parquet_path}. Columns: {df.columns.tolist()}")
    
    # 首先打印相机参数和统计信息
    print(f"  Camera params:")
    print(f"    Intrinsic:\n{intrinsic}")
    print(f"    Extrinsic (camera-to-world):\n{extrinsic}")
    print(f"    Camera pos in world: {extrinsic[:3, 3]}")
    print(f"    Image size: {image_width}x{image_height}")
    print(f"    Heatmap size: {heatmap_width}x{heatmap_width}")
    
    results = []
    debug_count = 0
    for idx, row in df.iterrows():
        state = np.array(row[state_col])
        eef_pos = state[:3]  # 前3维是end-effector position (x, y, z)
        
        # 投影到原始渲染分辨率
        # 前3帧打印详细调试信息
        verbose = (idx < 3)
        u, v, is_visible = project_world_to_pixel(
            eef_pos, intrinsic, extrinsic, image_height, image_width, verbose=verbose
        )
        
        # 调试输出前5帧
        if debug_count < 5:
            print(f"  Frame {idx}: EEF=({eef_pos[0]:.3f}, {eef_pos[1]:.3f}, {eef_pos[2]:.3f}), "
                  f"Pixel=({u:.1f}, {v:.1f}), Visible={is_visible}")
            debug_count += 1
        
        # 缩放到heatmap分辨率
        scale_x = heatmap_width / image_width
        scale_y = heatmap_height / image_height
        heatmap_u = u * scale_x
        heatmap_v = v * scale_y
        
        results.append({
            "frame_idx": int(row.get("frame_index", idx)),
            "eef_pos_3d": eef_pos.tolist(),
            "pixel_u": u,
            "pixel_v": v,
            "heatmap_u": heatmap_u,
            "heatmap_v": heatmap_v,
            "is_visible": is_visible,
        })
    
    # 统计
    visible_frames = sum(1 for r in results if r["is_visible"])
    print(f"  Summary: {visible_frames}/{len(results)} frames visible")
    
    # 检查EEF位置范围
    all_eef = np.array([r["eef_pos_3d"] for r in results])
    print(f"  EEF position range:")
    print(f"    X: [{all_eef[:, 0].min():.3f}, {all_eef[:, 0].max():.3f}]")
    print(f"    Y: [{all_eef[:, 1].min():.3f}, {all_eef[:, 1].max():.3f}]")
    print(f"    Z: [{all_eef[:, 2].min():.3f}, {all_eef[:, 2].max():.3f}]")
    
    if visible_frames == 0:
        print(f"  WARNING: No frames are visible!")
        print(f"  This might be due to:")
        print(f"    1. Wrong camera parameters (check camera_params.json)")
        print(f"    2. EEF positions are truly behind the camera")
        print(f"    3. Coordinate system mismatch (check extrinsic matrix)")
    
    return results


def generate_heatmap_video_for_episode(
    projections: list,
    heatmap_height: int = 224,
    heatmap_width: int = 224,
    sigma: float = 5.0,
) -> np.ndarray:
    """
    为一个episode生成heatmap视频序列。
    
    Returns:
        heatmaps: (T, H, W) float32 array, range [0, 1]
    """
    T = len(projections)
    heatmaps = np.zeros((T, heatmap_height, heatmap_width), dtype=np.float32)
    
    for i, proj in enumerate(projections):
        if proj["is_visible"]:
            heatmaps[i] = generate_gaussian_heatmap(
                proj["heatmap_u"],
                proj["heatmap_v"],
                heatmap_height,
                heatmap_width,
                sigma=sigma,
            )
    
    return heatmaps


# ============================================================================
# Part 4: 集成到你的训练pipeline
# ============================================================================

def demo_integration_with_cosmos_wam():
    """
    演示如何在你的Cosmos-WAM训练pipeline中使用heatmap。
    
    核心思路：
    1. 预计算所有episode的heatmap，存成.npy或.mp4
    2. 在RobotVideoDataset的__getitem__中，和RGB视频一起加载heatmap
    3. heatmap过VAE encoder编码成latent
    4. 沿view维度和RGB latent拼接
    5. DiT同时denoise两者
    """
    
    print("""
    ============================================================
    集成方案概述
    ============================================================
    
    你的架构: Cosmos DiT (28层) + Action Head (14层)
    当前输入: RGB视频 [B, 3, T, H, W] -> VAE -> latent [B, 16, T', H', W']
    
    添加heatmap后:
    - RGB视频: [B, 3, T, H, W] -> VAE -> rgb_latent [B, 16, T', H', W']
    - Heatmap视频: [B, 3, T, H, W] -> VAE -> hm_latent [B, 16, T', H', W']
    - 拼接方式 (MV-VDP style): 沿view维度
      concat_latent = cat([rgb_latent, hm_latent], dim=?) 
    
    具体修改点:
    
    1. 数据层 (robot_video_dataset.py):
       - 在_get()中，额外加载对应帧的heatmap序列
       - heatmap也做resize/normalize到[-1,1]
       - 返回 data["heatmap_video"] = heatmap_tensor  # [3, T, H, W]
    
    2. 模型层 (cosmos_wam.py - build_inputs):
       - heatmap也过VAE encode
       - 和RGB latent拼接:
         方案A (view维度): 把heatmap当第2个"相机"
           latents = cat([rgb_latent, hm_latent], dim=3)  # 沿W维度拼
         方案B (直接相加): latents = rgb_latent + alpha * hm_latent
    
    3. Loss: 
       - 如果用view维度拼接，DiT输出也是拼接的，
         可以split后分别算video_loss和heatmap_loss
       - L = L_video + lambda_hm * L_heatmap + lambda_action * L_action
    
    4. 推理时:
       - 不需要heatmap（和MV-VDP一样，heatmap只在训练时使用）
       - 或者可以只用1步denoise就能得到heatmap预测
    ============================================================
    """)


# ============================================================================
# Part 5: 验证脚本（不需要LIBERO环境，用硬编码的参数测试投影逻辑）
# ============================================================================

def test_projection_without_libero():
    """
    用LIBERO agentview的典型参数测试投影逻辑。
    
    LIBERO agentview的典型参数（从MuJoCo XML中获取）:
    - fovy: 45度 (robosuite默认)
    - 分辨率: 256x256
    - 相机位置: 大约在桌子上方偏后
    """
    image_height = 256
    image_width = 256
    fovy = 45.0  # robosuite默认
    
    # 计算内参
    f = 0.5 * image_height / np.tan(fovy * np.pi / 360.0)
    K = np.array([
        [f, 0, image_width / 2.0],
        [0, f, image_height / 2.0],
        [0, 0, 1.0]
    ])
    
    print(f"Focal length: {f:.2f}")
    print(f"Intrinsic matrix K:\n{K}\n")
    
    # 模拟一个camera-to-world外参（agentview大约在桌子正前方上方）
    # 这是一个示意值，真实值需要从环境中提取
    # robosuite的agentview通常位于 (0, -0.6, 0.9) 看向 (0, 0, 0.8)
    extrinsic = np.eye(4)
    extrinsic[:3, 3] = [0.0, -0.6, 0.9]  # 相机位置
    # 简化的旋转矩阵（相机朝前下方看）
    pitch = -30 * np.pi / 180  # 向下看30度
    extrinsic[:3, :3] = np.array([
        [1, 0, 0],
        [0, np.cos(pitch), -np.sin(pitch)],
        [0, np.sin(pitch), np.cos(pitch)],
    ])
    
    # 测试几个典型的end-effector位置
    test_points = [
        np.array([0.0, 0.0, 0.85]),    # 桌面中心
        np.array([0.1, 0.1, 0.9]),     # 稍偏右上
        np.array([-0.1, -0.1, 0.85]),  # 稍偏左下
        np.array([0.0, 0.0, 1.0]),     # 抬高
    ]
    
    print("Test projections (示意值，真实值需从环境提取):")
    print("-" * 60)
    for pt in test_points:
        u, v, visible = project_world_to_pixel(pt, K, extrinsic, image_height, image_width)
        print(f"  3D: [{pt[0]:.2f}, {pt[1]:.2f}, {pt[2]:.2f}] -> "
              f"Pixel: ({u:.1f}, {v:.1f}), Visible: {visible}")
    
    # 生成一个示例heatmap
    u, v, _ = project_world_to_pixel(test_points[0], K, extrinsic, image_height, image_width)
    heatmap = generate_gaussian_heatmap(u, v, 224, 224, sigma=8.0)
    print(f"\nHeatmap shape: {heatmap.shape}")
    print(f"Heatmap max: {heatmap.max():.4f}, min: {heatmap.min():.4f}")
    print(f"Heatmap center at ({u:.1f}, {v:.1f})")
    
    return K, extrinsic


# ============================================================================
# Part 6: 单个视频验证 - 生成对比视频
# ============================================================================

def visualize_single_episode(
    parquet_path: str,
    video_path: str,
    camera_params_path: str,
    output_path: str,
    heatmap_height: int = 224,
    heatmap_width: int = 224,
    sigma: float = 8.0,
    alpha: float = 0.5,
):
    """
    为单个episode生成heatmap可视化视频。
    
    输出: 一个MP4视频，左边是原视频，右边是heatmap叠加
    
    Args:
        parquet_path: episode的parquet文件路径 (e.g., episode_000000.parquet)
        video_path: 对应的视频文件路径 (e.g., episode_000000.mp4)
        camera_params_path: 相机参数JSON文件路径
        output_path: 输出视频路径 (e.g., output.mp4)
        heatmap_height/width: heatmap分辨率
        sigma: Gaussian标准差
        alpha: heatmap叠加透明度 (0-1)
    """
    import cv2
    import torchvision
    from tqdm import tqdm
    from pathlib import Path
    
    # 加载相机参数
    with open(camera_params_path, "r") as f:
        params = json.load(f)
    
    intrinsic = np.array(params["intrinsic"])
    extrinsic = np.array(params["extrinsic"])
    render_h = params["image_height"]
    render_w = params["image_width"]
    
    # 处理parquet获取投影
    print(f"Processing parquet: {parquet_path}")
    projections = process_lerobot_episode(
        parquet_path,
        intrinsic,
        extrinsic,
        render_h,
        render_w,
        heatmap_height,
        heatmap_width,
        sigma,
    )
    
    # 使用torchvision读取视频 (和数据集相同的方法)
    print(f"Opening video: {video_path}")
    torchvision.set_video_backend("pyav")
    reader = torchvision.io.VideoReader(video_path, "video")
    
    # 获取视频信息
    fps = reader.get_metadata()["video"]["fps"][0]
    video_info = reader.get_metadata()["video"]
    video_width = video_info["width"][0] if "width" in video_info else 512
    video_height = video_info["height"][0] if "height" in video_info else 512
    
    print(f"Video: {video_width}x{video_height} @ {fps}fps")
    print(f"Projections: {len(projections)} frames")
    
    # 准备输出视频 (并排显示: 原视频 | heatmap)
    output_width = video_width * 2
    output_height = max(video_height, heatmap_height)
    
    # 准备colormap
    colormap = cv2.COLORMAP_JET
    
    # 收集所有帧，然后一次性写入（避免OpenCV编码器问题）
    frames_to_save = []
    frame_idx = 0
    pbar = tqdm(total=len(projections), desc="Generating video")
    
    # 读取所有帧
    for frame in reader:
        if frame_idx >= len(projections):
            break
        
        # 获取帧数据 [C, H, W] -> [H, W, C]
        frame_tensor = frame["data"]  # [3, H, W], float32 [0, 1]
        frame_rgb = (frame_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        
        # OpenCV需要BGR格式
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        
        # 生成heatmap
        proj = projections[frame_idx]
        if proj["is_visible"]:
            heatmap = generate_gaussian_heatmap(
                proj["heatmap_u"],
                proj["heatmap_v"],
                heatmap_height,
                heatmap_width,
                sigma=sigma,
            )
            
            # Resize heatmap to match video size
            heatmap_resized = cv2.resize(heatmap, (video_width, video_height))
            
            # 转换为colormap (0-255)
            heatmap_colored = cv2.applyColorMap((heatmap_resized * 255).astype(np.uint8), colormap)
            
            # 叠加到原视频 (heatmap_colored是BGR格式)
            frame_with_heatmap = (frame_rgb * (1 - alpha) + heatmap_colored * alpha).astype(np.uint8)
            
            # 添加文字信息 (OpenCV在RGB图像上绘制)
            info_text = f"EEF: ({proj['eef_pos_3d'][0]:.3f}, {proj['eef_pos_3d'][1]:.3f}, {proj['eef_pos_3d'][2]:.3f})"
            cv2.putText(frame_with_heatmap, info_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            pixel_text = f"Pixel: ({proj['pixel_u']:.1f}, {proj['pixel_v']:.1f})"
            cv2.putText(frame_with_heatmap, pixel_text, (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # 转回BGR用于输出
            frame_with_heatmap_bgr = cv2.cvtColor(frame_with_heatmap, cv2.COLOR_RGB2BGR)
        else:
            # EEF不在视野内
            cv2.putText(frame_bgr, "EEF NOT VISIBLE", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            frame_with_heatmap_bgr = frame_bgr.copy()
        
        # 并排拼接
        combined = np.concatenate([frame_bgr, frame_with_heatmap_bgr], axis=1)
        frames_to_save.append(combined)
        
        frame_idx += 1
        pbar.update(1)
    
    pbar.close()
    reader.container.close()  # 关闭视频reader
    
    print(f"\nCollected {len(frames_to_save)} frames, saving to video...")
    
    # 使用torchvision保存视频（更可靠）
    try:
        import torchvision.io as io
        # 转换回RGB并调整维度为 [T, H, W, C]
        frames_array = np.stack(frames_to_save)  # [T, H, W, 3] BGR
        # BGR -> RGB
        frames_rgb = frames_array[..., ::-1].copy()
        frames_tensor = torch.from_numpy(frames_rgb)  # [T, H, W, 3]
        
        # 使用torchvision保存
        io.write_video(output_path, frames_tensor, fps, video_codec='libx264')
        print(f"Saved using torchvision (H.264)")
    except Exception as e:
        print(f"torchvision save failed: {e}")
        print("Trying imageio...")
        try:
            import imageio
            writer = imageio.get_writer(output_path, fps=fps, codec='libx264', quality=8)
            for frame in frames_to_save:
                writer.append_data(frame)
            writer.close()
            print(f"Saved using imageio")
        except Exception as e2:
            print(f"imageio save failed: {e2}")
            print("Falling back to frame sequence...")
            # 保存为帧序列
            import os
            output_dir = output_path.replace('.mp4', '_frames')
            os.makedirs(output_dir, exist_ok=True)
            for i, frame in enumerate(frames_to_save):
                cv2.imwrite(f"{output_dir}/frame_{i:05d}.png", frame)
            print(f"Saved frame sequence to {output_dir}")
    
    print(f"\nSaved visualization to: {output_path}")
    print(f"Total frames processed: {frame_idx}")
    
    # 打印统计
    if frame_idx > 0:
        visible_count = sum(1 for p in projections[:frame_idx] if p["is_visible"])
        print(f"Visible frames: {visible_count}/{frame_idx} ({100*visible_count/frame_idx:.1f}%)")
    else:
        print("Warning: No frames were processed")


# ============================================================================
# Part 7: 完整的预处理脚本 - 为所有episode生成heatmap
# ============================================================================

def precompute_all_heatmaps(
    dataset_dir: str,
    camera_params_path: str,
    output_dir: str,
    heatmap_height: int = 224,
    heatmap_width: int = 224,
    sigma: float = 8.0,
):
    """
    为整个数据集预计算所有episode的heatmap。
    
    输出结构:
    output_dir/
    ├── episode_000000_heatmaps.npy  # (T, H, W) float32
    ├── episode_000001_heatmaps.npy
    ├── ...
    └── projections.json  # 所有投影的元数据
    """
    import glob
    
    # 加载相机参数
    with open(camera_params_path, "r") as f:
        params = json.load(f)
    
    intrinsic = np.array(params["intrinsic"])
    extrinsic = np.array(params["extrinsic"])
    render_h = params["image_height"]
    render_w = params["image_width"]
    
    # 找到所有parquet文件
    dataset_path = Path(dataset_dir)
    parquet_files = sorted(dataset_path.glob("data/chunk-*/episode_*.parquet"))
    
    if not parquet_files:
        print(f"No parquet files found in {dataset_dir}/data/chunk-*/")
        return
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    all_projections = {}
    
    for pq_file in parquet_files:
        episode_name = pq_file.stem  # e.g., "episode_000000"
        print(f"Processing {episode_name}...")
        
        # 投影
        projections = process_lerobot_episode(
            str(pq_file),
            intrinsic,
            extrinsic,
            render_h,
            render_w,
            heatmap_height,
            heatmap_width,
            sigma,
        )
        
        # 生成heatmap视频
        heatmaps = generate_heatmap_video_for_episode(
            projections,
            heatmap_height,
            heatmap_width,
            sigma,
        )
        
        # 保存
        np.save(str(output_path / f"{episode_name}_heatmaps.npy"), heatmaps)
        all_projections[episode_name] = projections
        
        # 统计
        visible_count = sum(1 for p in projections if p["is_visible"])
        print(f"  Frames: {len(projections)}, Visible: {visible_count}/{len(projections)}")
    
    # 保存元数据
    with open(str(output_path / "projections.json"), "w") as f:
        json.dump(all_projections, f, indent=2)
    
    print(f"\nDone! Saved {len(parquet_files)} episodes to {output_dir}")


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LIBERO Heatmap Generation")
    subparsers = parser.add_subparsers(dest="command")
    
    # Command: extract_camera_params
    extract_parser = subparsers.add_parser(
        "extract_camera_params",
        help="Extract camera intrinsic/extrinsic from LIBERO environment"
    )
    extract_parser.add_argument("--task_suite", default="libero_spatial")
    extract_parser.add_argument("--task_id", type=int, default=0)
    extract_parser.add_argument("--camera_name", default="agentview")
    extract_parser.add_argument("--resolution", type=int, default=256)
    extract_parser.add_argument("--output", default="camera_params.json")
    
    # Command: generate_heatmaps
    gen_parser = subparsers.add_parser(
        "generate_heatmaps",
        help="Generate heatmaps for all episodes in a dataset"
    )
    gen_parser.add_argument("--dataset_dir", required=True)
    gen_parser.add_argument("--camera_params", required=True)
    gen_parser.add_argument("--output_dir", required=True)
    gen_parser.add_argument("--heatmap_height", type=int, default=224)
    gen_parser.add_argument("--heatmap_width", type=int, default=224)
    gen_parser.add_argument("--sigma", type=float, default=8.0)
    
    # Command: test (不需要LIBERO环境)
    test_parser = subparsers.add_parser(
        "test",
        help="Test projection logic without LIBERO environment"
    )
    
    # Command: demo
    demo_parser = subparsers.add_parser(
        "demo",
        help="Show integration plan for Cosmos-WAM"
    )
    
    # Command: visualize_single (新增)
    single_parser = subparsers.add_parser(
        "visualize_single",
        help="Generate heatmap visualization for a single episode (for verification)"
    )
    single_parser.add_argument("--parquet_path", required=True, 
                               help="Path to episode parquet file (e.g., episode_000000.parquet)")
    single_parser.add_argument("--video_path", required=True,
                               help="Path to corresponding video file (e.g., episode_000000.mp4)")
    single_parser.add_argument("--camera_params", required=True,
                               help="Path to camera_params.json")
    single_parser.add_argument("--output", required=True,
                               help="Output video path (e.g., output.mp4)")
    single_parser.add_argument("--heatmap_height", type=int, default=224)
    single_parser.add_argument("--heatmap_width", type=int, default=224)
    single_parser.add_argument("--sigma", type=float, default=8.0,
                               help="Gaussian sigma for heatmap")
    single_parser.add_argument("--alpha", type=float, default=0.5,
                               help="Heatmap overlay transparency (0-1)")
    
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
        print(f"Saved camera params to {args.output}")
    
    elif args.command == "generate_heatmaps":
        precompute_all_heatmaps(
            dataset_dir=args.dataset_dir,
            camera_params_path=args.camera_params,
            output_dir=args.output_dir,
            heatmap_height=args.heatmap_height,
            heatmap_width=args.heatmap_width,
            sigma=args.sigma,
        )
    
    elif args.command == "test":
        test_projection_without_libero()
    
    elif args.command == "demo":
        demo_integration_with_cosmos_wam()
    
    elif args.command == "visualize_single":
        visualize_single_episode(
            parquet_path=args.parquet_path,
            video_path=args.video_path,
            camera_params_path=args.camera_params,
            output_path=args.output,
            heatmap_height=args.heatmap_height,
            heatmap_width=args.heatmap_width,
            sigma=args.sigma,
            alpha=args.alpha,
        )
    
    else:
        parser.print_help()
