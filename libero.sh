#!/bin/bash
# Cosmos-WAM LIBERO Evaluation Script

# Configuration
CKPT="/home/jwhe/linyihan/CosmosWAM/outputs/cosmos_2b_libero_20260408_125315/checkpoints/step_0004000.pt"
DATASET_STATS="/home/jwhe/linyihan/CosmosWAM/datasets_stats/libero_spatial_dataset_stats.json"
TEXT_EMBED_CACHE="/home/jwhe/linyihan/datasets/text_embeds_cache/libero"

# Task suite and ID
TASK_SUITE="libero_spatial"  # Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
TASK_ID=0                    # 0-9 for most suites
NUM_TRIALS=50                # Number of episodes per task

# Evaluation settings
NUM_INFERENCE_STEPS=4        # Diffusion sampling steps
REPLAN_STEPS=5               # Execute this many actions before replanning
ACTION_HORIZON=32
GPU_ID=0

echo "=============================================="
echo "Cosmos-WAM LIBERO Evaluation"
echo "=============================================="
echo "Task Suite: $TASK_SUITE"
echo "Task ID: $TASK_ID"
echo "Num Trials: $NUM_TRIALS"
echo "Checkpoint: $CKPT"
echo "=============================================="

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

echo ""
echo "=============================================="
echo "Evaluation Complete!"
echo "Results saved to: ./evaluate_results/libero/"
echo "=============================================="
