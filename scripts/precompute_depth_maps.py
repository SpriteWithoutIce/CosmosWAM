#!/usr/bin/env python3
"""
Precompute depth maps for LIBERO wrist camera videos using Depth-Anything-V2.
"""

import argparse
import os
import sys
import glob
import cv2
import numpy as np
import torch
from tqdm import tqdm

try:
    import av
    HAS_AV = True
except ImportError:
    HAS_AV = False

try:
    from decord import VideoReader, cpu
    HAS_DECORD = True
except ImportError:
    HAS_DECORD = False


def read_video_av(video_path):
    """使用 PyAV 读取视频，支持 AV1 等编码"""
    container = av.open(video_path)
    frames = []
    stream = container.streams.video[0]
    for frame in container.decode(video=0):
        img = frame.to_ndarray(format='bgr24')
        frames.append(img)
    container.close()
    if len(frames) == 0:
        raise RuntimeError(f"无法读取视频: {video_path}")
    return frames


def read_video_decord(video_path):
    """使用 decord 读取视频"""
    vr = VideoReader(video_path, ctx=cpu(0))
    frames = vr.get_batch(range(len(vr))).asnumpy()
    # decord 默认返回 RGB，转成 BGR 以和 OpenCV 行为一致
    frames = [cv2.cvtColor(f, cv2.COLOR_RGB2BGR) for f in frames]
    return frames


def read_video_cv2_fallback(video_path):
    """使用 cv2 读取视频（备用方案）"""
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    if len(frames) == 0:
        raise RuntimeError(f"无法读取视频: {video_path}")
    return frames


def read_video_robust(video_path):
    """优先用 PyAV，其次 decord，最后 cv2 fallback"""
    if HAS_AV:
        try:
            return read_video_av(video_path)
        except Exception as e:
            print(f"PyAV 读取失败 ({e}), 尝试 decord...")
    if HAS_DECORD:
        try:
            return read_video_decord(video_path)
        except Exception as e:
            print(f"decord 读取失败 ({e}), 尝试 cv2...")
    return read_video_cv2_fallback(video_path)


def main():
    parser = argparse.ArgumentParser(description="Precompute depth maps for wrist videos")
    parser.add_argument("--dataset-dirs", nargs="+", required=True,
                        help="List of LeRobot dataset directories")
    parser.add_argument("--video-key", default="observation.images.wrist_image",
                        help="Video key subdir name")
    parser.add_argument("--outdir", required=True,
                        help="Output directory for .npy depth maps")
    parser.add_argument("--depth-repo", default="/Users/linyihan/Documents/Embodied_AI/code/Depth-Anything-V2",
                        help="Path to Depth-Anything-V2 repository")
    parser.add_argument("--ckpt", default="/home/jwhe/linyihan/CKPT/depth_anything_v2_vitb.pth",
                        help="Path to Depth-Anything-V2 checkpoint")
    parser.add_argument("--input-size", type=int, default=518,
                        help="Input size for depth inference")
    parser.add_argument("--resize-to", type=int, default=224,
                        help="Resize output depth maps to this resolution")
    args = parser.parse_args()

    # Add Depth-Anything-V2 repo to path
    if args.depth_repo not in sys.path:
        sys.path.insert(0, args.depth_repo)

    from depth_anything_v2.dpt import DepthAnythingV2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_configs = {
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    }
    model = DepthAnythingV2(**model_configs['vitb'])
    model.load_state_dict(torch.load(args.ckpt, map_location="cpu"))
    model = model.to(device).eval()

    os.makedirs(args.outdir, exist_ok=True)

    for dataset_dir in args.dataset_dirs:
        ds_name = os.path.basename(os.path.normpath(dataset_dir))
        video_pattern = os.path.join(dataset_dir, "videos", "chunk-*", args.video_key, "*.mp4")
        video_paths = sorted(glob.glob(video_pattern))

        print(f"Processing dataset: {ds_name} ({len(video_paths)} videos)")
        for video_path in tqdm(video_paths, desc=f"Depth ({ds_name})"):
            try:
                frames = read_video_robust(video_path)
            except Exception as e:
                print(f"Warning: failed to read video {video_path}: {e}")
                continue

            depths = []
            for frame in frames:
                d = model.infer_image(frame, args.input_size)
                if args.resize_to > 0 and d.shape != (args.resize_to, args.resize_to):
                    d = cv2.resize(d, (args.resize_to, args.resize_to), interpolation=cv2.INTER_LINEAR)
                depths.append(d)

            depths = np.stack(depths, axis=0).astype(np.float32)

            ep_name = os.path.splitext(os.path.basename(video_path))[0]
            try:
                ep_idx = int(ep_name)
            except ValueError:
                # fallback: strip non-digit prefix/suffix if possible
                ep_idx = int(''.join(filter(str.isdigit, ep_name)))
            out_name = f"{ds_name}_episode_{ep_idx:06d}_depth.npy"
            np.save(os.path.join(args.outdir, out_name), depths)

    print(f"Done! Saved depth maps to {args.outdir}")


if __name__ == "__main__":
    main()
