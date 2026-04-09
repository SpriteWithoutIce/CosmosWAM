#!/bin/bash
# Cosmos-WAM LIBERO Evaluation Script

# Configuration
CKPT="/home/jwhe/linyihan/CosmosWAM/outputs/cosmos_2b_libero_20260408_125315/checkpoints/step_0004000.pt"
DATASET_STATS="/home/jwhe/linyihan/CosmosWAM/datasets_stats/libero_spatial_dataset_stats.json"

# LIBERO library path (can be overridden)
export LIBERO_PATH="${LIBERO_PATH:-/home/jwhe/linyihan/LIBERO}"

# Task suite and ID
TASK_SUITE="libero_spatial"  # Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
TASK_ID=0                    # 0-9 for most suites
NUM_TRIALS=50                # Number of episodes per task

# Evaluation settings
NUM_INFERENCE_STEPS=4        # Diffusion sampling steps
REPLAN_STEPS=5               # Execute this many actions before replanning
ACTION_HORIZON=32
GPU_ID=0

# Text embedding settings
# Option 1: Use precomputed cache (set USE_ONLINE_ENCODER=false and set CACHE_DIR)
# Option 2: Use online encoder (set USE_ONLINE_ENCODER=true and set ENCODER_PATH)
USE_ONLINE_ENCODER=true
TEXT_EMBED_CACHE=""  # Not used when online encoder is enabled
ONLINE_ENCODER_PATH="/mnt/data/linyihan/Cosmos-Reason1-7b"
TEXT_ENCODER_DEVICE="cuda:0"

echo "=============================================="
echo "Cosmos-WAM LIBERO Evaluation"
echo "=============================================="
echo "LIBERO Path: $LIBERO_PATH"
echo "Task Suite: $TASK_SUITE"
echo "Task ID: $TASK_ID"
echo "Num Trials: $NUM_TRIALS"
echo "Checkpoint: $CKPT"
echo "Text Encoder: $([ "$USE_ONLINE_ENCODER" = true ] && echo "Online ($ONLINE_ENCODER_PATH)" || echo "Cache ($TEXT_EMBED_CACHE)")"
echo "=============================================="

if [ "$USE_ONLINE_ENCODER" = true ]; then
    # Use online text encoder
    python experiments/libero/eval_libero_single.py \
        ckpt="$CKPT" \
        EVALUATION.task_suite_name="$TASK_SUITE" \
        EVALUATION.task_id=$TASK_ID \
        EVALUATION.num_trials=$NUM_TRIALS \
        EVALUATION.dataset_stats_path="$DATASET_STATS" \
        EVALUATION.num_inference_steps=$NUM_INFERENCE_STEPS \
        EVALUATION.replan_steps=$REPLAN_STEPS \
        EVALUATION.action_horizon=$ACTION_HORIZON \
        EVALUATION.gpu_id=$GPU_ID \
        EVALUATION.use_online_text_encoder=true \
        EVALUATION.online_text_encoder_path="$ONLINE_ENCODER_PATH" \
        EVALUATION.text_encoder_device="$TEXT_ENCODER_DEVICE" \
        mixed_precision=bf16
else
    # Use precomputed text embedding cache
    python experiments/libero/eval_libero_single.py \
        ckpt="$CKPT" \
        EVALUATION.task_suite_name="$TASK_SUITE" \
        EVALUATION.task_id=$TASK_ID \
        EVALUATION.num_trials=$NUM_TRIALS \
        EVALUATION.dataset_stats_path="$DATASET_STATS" \
        EVALUATION.text_embedding_cache_dir="$TEXT_EMBED_CACHE" \
        EVALUATION.num_inference_steps=$NUM_INFERENCE_STEPS \
        EVALUATION.replan_steps=$REPLAN_STEPS \
        EVALUATION.action_horizon=$ACTION_HORIZON \
        EVALUATION.gpu_id=$GPU_ID \
        mixed_precision=bf16
fi

echo ""
echo "=============================================="
echo "Evaluation Complete!"
echo "Results saved to: ./evaluate_results/libero/"
echo "=============================================="
