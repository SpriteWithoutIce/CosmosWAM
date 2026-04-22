#!/bin/bash
# Cosmos-WAM LIBERO Evaluation Script (支持多 GPU 并行 + Task 循环)

# 从命令行接收参数
GPU_ID=${1:-0}      # 第一个参数：GPU 编号 (0-3)，默认 0

START_TASK=0
END_TASK=9

# 根据 GPU_ID 设置环境变量
export CUDA_VISIBLE_DEVICES=$GPU_ID
export MUJOCO_EGL_DEVICE_ID=0  # EGL 始终用 0（因为 CUDA_VISIBLE_DEVICES 只暴露一块 GPU）

# 其他配置
CKPT="/home/jwhe/linyihan/CosmosWAM/outputs/cosmos_2b_libero_20260414_170339/checkpoints/step_0020000.pt"
DATASET_STATS="/home/jwhe/linyihan/CosmosWAM/outputs/dataset_stats.json"
export LIBERO_PATH="${LIBERO_PATH:-/home/jwhe/linyihan/LIBERO}"

# 固定配置
TASK_SUITE="libero_10"
NUM_TRIALS=10
NUM_INFERENCE_STEPS=10
REPLAN_STEPS=8
ACTION_HORIZON=32

# 文本编码器配置
USE_ONLINE_ENCODER=false
TEXT_EMBED_CACHE="/home/jwhe/linyihan/datasets/text_embeds_cache/libero"
TEXT_ENCODER_DEVICE="cuda:0"  # 程序内部只有一块 GPU，所以是 cuda:0

echo "=============================================="
echo "Cosmos-WAM LIBERO Evaluation"
echo "GPU: $GPU_ID (Physical) -> cuda:0 (Internal)"
echo "Task Suite: $TASK_SUITE"
echo "=============================================="

if [ "$USE_ONLINE_ENCODER" = true ]; then
    $(which python3) experiments/libero/eval_libero_single.py \
        ckpt="$CKPT" \
        EVALUATION.task_suite_name="$TASK_SUITE" \
        EVALUATION.num_trials=$NUM_TRIALS \
        EVALUATION.dataset_stats_path="$DATASET_STATS" \
        EVALUATION.num_inference_steps=$NUM_INFERENCE_STEPS \
        EVALUATION.replan_steps=$REPLAN_STEPS \
        EVALUATION.action_horizon=$ACTION_HORIZON \
        EVALUATION.gpu_id=0 \
        EVALUATION.use_online_text_encoder=true \
        EVALUATION.online_text_encoder_path="/home/jwhe/linyihan/CKPT/Cosmos-Reason1-7B" \
        EVALUATION.text_encoder_device="$TEXT_ENCODER_DEVICE" \
        mixed_precision=bf16
else
    $(which python3) experiments/libero/eval_libero_single.py \
        ckpt="$CKPT" \
        EVALUATION.task_suite_name="$TASK_SUITE" \
        EVALUATION.num_trials=$NUM_TRIALS \
        EVALUATION.dataset_stats_path="$DATASET_STATS" \
        EVALUATION.text_embedding_cache_dir="$TEXT_EMBED_CACHE" \
        EVALUATION.num_inference_steps=$NUM_INFERENCE_STEPS \
        EVALUATION.replan_steps=$REPLAN_STEPS \
        EVALUATION.action_horizon=$ACTION_HORIZON \
        EVALUATION.gpu_id=0 \
        mixed_precision=bf16
fi

echo ""
echo "=============================================="
echo "All Tasks ($START_TASK ~ $END_TASK) on GPU $GPU_ID Complete!"
echo "Results saved to: ./evaluate_results/libero/"
echo "=============================================="