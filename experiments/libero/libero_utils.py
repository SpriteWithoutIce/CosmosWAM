"""Utils for evaluating policies in LIBERO simulation environments."""

import math
import sys
import time
import pathlib

# Add LIBERO to path (can be overridden via environment variable)
import os
LIBERO_PATH = os.environ.get("LIBERO_PATH", "/home/jwhe/linyihan/LIBERO")
if LIBERO_PATH not in sys.path:
    sys.path.insert(0, LIBERO_PATH)

import imageio
from PIL import Image, ImageDraw
import numpy as np
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv, SubprocVectorEnv
from cosmos_wam.utils.projection import compute_pose_keypoints, project_world_to_pixel

DATE = time.strftime("%Y_%m_%d")
DATE_TIME = time.strftime("%Y_%m_%d-%H_%M_%S")
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data


def get_libero_env(task, resolution, seed, env_num=1):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = (
        pathlib.Path(get_libero_path("bddl_files"))
        / task.problem_folder
        / task.bddl_file
    )
    env_args = {
        "bddl_file_name": task_bddl_file,
        "camera_heights": resolution,
        "camera_widths": resolution,
    }
    if env_num > 1:
        env = SubprocVectorEnv([lambda: OffScreenRenderEnv(**env_args) for _ in range(env_num)])
    else:
        env = OffScreenRenderEnv(**env_args)
    env.seed(seed)
    return env, task_description


def get_libero_dummy_action():
    """Get dummy/no-op action, used to roll out the simulation while the robot does nothing."""
    return [0, 0, 0, 0, 0, 0, -1]


def get_libero_image(obs):
    """Extracts image from observations and preprocesses it."""
    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
    # IMPORTANT: rotate 180 degrees for LIBERO environment compatibility
    
    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
    # IMPORTANT: rotate 180 degrees for LIBERO environment compatibility
    
    return {
        "image": img,
        "wrist_image": wrist_img
    }


def save_rollout_video(rollout_dir, rollout_images, idx, success, task_description, fps=24):
    """Saves an MP4 replay of an episode."""
    processed_task_description = task_description.lower().replace(" ", "_").replace("\n", "_").replace(".", "_")[:50]
    mp4_path = f"{rollout_dir}/{DATE_TIME}--episode={idx}--success={success}--task={processed_task_description}.mp4"
    video_writer = imageio.get_writer(mp4_path, fps=fps)
    for img in rollout_images:
        if isinstance(img, dict):
            image = []
            for key, value in img.items():
                value_array = np.array(value) if isinstance(value, Image.Image) else value.copy()
                pil_img = Image.fromarray(value_array)
                draw = ImageDraw.Draw(pil_img)
                draw.text((10, 10), f"{key}", fill=(255, 255, 255))
                image.append(np.array(pil_img))
            frame = np.concatenate(image, axis=1)
        elif isinstance(img, Image.Image):
            frame = np.array(img.convert("RGB"))
        else:
            frame = np.array(img)
        video_writer.append_data(frame)
    video_writer.close()
    print(f"Saved rollout MP4 at path {mp4_path}")
    return mp4_path


def quat2axisangle(quat):
    """Converts quaternion to axis-angle format."""
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def invert_gripper_action(action):
    """Flips the sign of the gripper action."""
    action[..., -1] = action[..., -1] * -1.0
    return action


def draw_pose_keypoints_on_image(image_np, points_2d, radius=4, thickness=2):
    """Draw 4 pose keypoints (origin + XYZ axes) on a numpy image.
    
    Args:
        image_np: [H, W, 3] uint8 numpy array.
        points_2d: [4, 2] numpy array of (u, v) pixel coordinates.
                  Order: origin, X-axis, Y-axis, Z-axis.
    Returns:
        Annotated image as numpy array.
    """
    pil_img = Image.fromarray(image_np.copy())
    draw = ImageDraw.Draw(pil_img)
    
    origin = tuple(points_2d[0].astype(int))
    px = tuple(points_2d[1].astype(int))
    py = tuple(points_2d[2].astype(int))
    pz = tuple(points_2d[3].astype(int))
    
    # Draw lines from origin to axis endpoints
    draw.line([origin, px], fill=(255, 0, 0), width=thickness)   # X: red
    draw.line([origin, py], fill=(0, 255, 0), width=thickness)   # Y: green
    draw.line([origin, pz], fill=(0, 0, 255), width=thickness)   # Z: blue
    
    # Draw circles at endpoints
    for pt, color in [(origin, (255, 255, 255)), (px, (255, 0, 0)), (py, (0, 255, 0)), (pz, (0, 0, 255))]:
        x, y = pt
        draw.ellipse([x - radius, y - radius, x + radius, y + radius], fill=color, outline=(0, 0, 0))
    
    # Add text labels
    draw.text((origin[0] + 6, origin[1] - 6), "O", fill=(255, 255, 255))
    draw.text((px[0] + 6, px[1] - 6), "X", fill=(255, 0, 0))
    draw.text((py[0] + 6, py[1] - 6), "Y", fill=(0, 255, 0))
    draw.text((pz[0] + 6, pz[1] - 6), "Z", fill=(0, 0, 255))
    
    return np.array(pil_img, dtype=np.uint8)


def project_and_visualize_current_pose(obs, intrinsic, extrinsic, render_h, render_w):
    """Project current eef pose to 2D and draw keypoints on the primary image.
    
    Args:
        obs: LIBERO observation dict.
        intrinsic: [3, 3] camera intrinsic matrix.
        extrinsic: [4, 4] camera extrinsic matrix.
        render_h, render_w: render resolution.
    
    Returns:
        annotated_image: [H, W, 3] uint8 numpy array with keypoints drawn.
        points_2d: [4, 2] projected 2D points (on render resolution).
    """
    eef_pos = obs["robot0_eef_pos"].astype(np.float32)
    eef_quat = obs["robot0_eef_quat"].astype(np.float32)
    eef_axis_angle = quat2axisangle(eef_quat)
    
    points_3d = compute_pose_keypoints(eef_pos, eef_axis_angle, axis_length=0.1)  # [4, 3]
    points_2d = project_world_to_pixel(points_3d, intrinsic, extrinsic, render_h, render_w)  # [4, 2]
    
    img = get_libero_image(obs)["image"]  # [H, W, 3], already rotated 180
    annotated = draw_pose_keypoints_on_image(img, points_2d)
    return annotated, points_2d
