#!/bin/bash
# Run a single RoboTwin task with specific GPU pair
# Usage: ./run_single_task.sh <task_name> <text_encoder_gpu> <model_gpu> <log_file>

TASK_NAME=$1
TEXT_ENCODER_GPU=$2
MODEL_GPU=$3
LOG_FILE=$4

# Configuration
CKPT="/mnt/data/linyihan/ckpt/step_0014000_bf16.pt"
ROBOTWIN_ROOT="/root/linyihan/RoboTwin"
DATASET_STATS="./dataset_stats.json"
ONLINE_TEXT_ENCODER_PATH="/mnt/data/linyihan/Cosmos-Reason1-7b"

NUM_EPISODES=50
NUM_INFERENCE_STEPS=20
REPLAN_STEPS=8
INSTRUCTION_TYPE="seen"
MIXED_PRECISION="bf16"

# Set GPUs for this process
export CUDA_VISIBLE_DEVICES="${TEXT_ENCODER_GPU},${MODEL_GPU}"
export PYTHONUNBUFFERED=1

echo "[$(date)] Starting task: $TASK_NAME on GPUs $TEXT_ENCODER_GPU,$MODEL_GPU"

python experiments/robotwin/eval_robotwin_single.py \
    ckpt="$CKPT" \
    EVALUATION.task_name="$TASK_NAME" \
    EVALUATION.eval_num_episodes="$NUM_EPISODES" \
    EVALUATION.robotwin_root="$ROBOTWIN_ROOT" \
    EVALUATION.dataset_stats_path="$DATASET_STATS" \
    EVALUATION.num_inference_steps="$NUM_INFERENCE_STEPS" \
    EVALUATION.replan_steps="$REPLAN_STEPS" \
    EVALUATION.instruction_type="$INSTRUCTION_TYPE" \
    mixed_precision="$MIXED_PRECISION" \
    gpu_id="$MODEL_GPU" \
    EVALUATION.use_online_text_encoder=true \
    EVALUATION.online_text_encoder_path="$ONLINE_TEXT_ENCODER_PATH" \
    EVALUATION.text_encoder_device="cuda:0" \
    EVALUATION.device="cuda:1" \
    2>&1 | tee "$LOG_FILE"

EXIT_CODE=${PIPESTATUS[0]}
echo "[$(date)] Task $TASK_NAME completed with exit code $EXIT_CODE"
exit $EXIT_CODE
