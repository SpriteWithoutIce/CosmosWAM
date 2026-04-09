#!/bin/bash
# 示例：从 checkpoint 恢复训练 + 新数据集

# ===== 1. 指定 GPU =====
# 单卡训练（使用第 0 张 GPU）
export CUDA_VISIBLE_DEVICES=0

# 或者多卡训练（使用第 0,1 张 GPU）
# export CUDA_VISIBLE_DEVICES=0,1

# ===== 2. 设置 LIBERO 路径 =====
export LIBERO_PATH=/home/jwhe/linyihan/LIBERO

# ===== 3. 运行训练 =====
# 修改 configs/train_cosmos_2b_libero.yaml 中的以下配置：
# 
# 1. 设置 resume_from_checkpoint 指向你的 checkpoint：
#    trainer:
#      resume_from_checkpoint: /home/jwhe/linyihan/CosmosWAM/outputs/cosmos_2b_libero_20260408_125315/checkpoints/final.pt
#
# 2. 修改数据集路径（如果换了新数据集）：
#    data:
#      train:
#        dataset_dirs:
#          - /path/to/new/dataset
#
# 3. 不要设置 pretrained_norm_stats，让代码自动计算新数据集的统计数据
#    （注释掉或删除这一行）

# 单卡训练
python scripts/train.py --config-name=train_cosmos_2b_libero

# 多卡训练（2卡）
# accelerate launch --num_processes=2 scripts/train.py --config-name=train_cosmos_2b_libero
