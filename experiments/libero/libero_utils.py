"""Utils for evaluating policies in LIBERO simulation environments."""

import math
import time
import pathlib

import imageio
from PIL import Image, ImageDraw
import numpy as np
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv, SubprocVectorEnv

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
    # IMPORTANT: rotate 180 degrees to match train preprocessing
    
    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
    # IMPORTANT: rotate 180 degrees to match train preprocessing
    
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
