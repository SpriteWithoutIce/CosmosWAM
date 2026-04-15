export CUDA_VISIBLE_DEVICES=3
python scripts/precompute_depth_maps.py \
      --dataset-dirs \
          /home/jwhe/linyihan/datasets/libero_mujoco3.3.2/libero_10_no_noops_lerobot \
      --video-key observation.images.wrist_image \
      --outdir /home/jwhe/linyihan/datasets/libero_depth_maps \
      --depth-repo /home/jwhe/linyihan/Depth-Anything-V2-main \
      --ckpt /home/jwhe/linyihan/CKPT/depth_anything_v2_vitb.pth \
      --input-size 518 \
      --resize-to 224