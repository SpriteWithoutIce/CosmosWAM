python scripts/gaussian/libero_heatmap_projection.py visualize_single \
      --parquet_path /home/jwhe/linyihan/datasets/libero_mujoco3.3.2/libero_goal_no_noops_lerobot/data/chunk-000/episode_000000.parquet \
      --video_path /home/jwhe/linyihan/datasets/libero_mujoco3.3.2/libero_goal_no_noops_lerobot/videos/chunk-000/observation.images.image/episode_000000.mp4 \
      --camera_params scripts/gaussian/camera_params.json \
      --output scripts/gaussian/heatmap_vis.mp4 \
      --sigma 8.0 \
      --alpha 0.5