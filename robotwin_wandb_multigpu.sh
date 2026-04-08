#!/bin/bash
# Cosmos-WAM RoboTwin Multi-GPU Evaluation with WandB Logging
# Uses 8 GPUs to run 4 tasks in parallel (2 GPUs per task)

# Configuration
CKPT="/mnt/data/linyihan/ckpt/step_0014000_bf16.pt"
ROBOTWIN_ROOT="/root/linyihan/RoboTwin"
DATASET_STATS="./dataset_stats.json"
ONLINE_TEXT_ENCODER_PATH="/mnt/data/linyihan/Cosmos-Reason1-7b"

# WandB settings
WANDB_PROJECT="cosmos-wam-robotwin"
WANDB_ENTITY=""  # Leave empty for default, or set your username/team
WANDB_RUN_NAME="cosmos-wam-8gpu-$(date +%Y%m%d-%H%M%S)"

# Evaluation settings
NUM_EPISODES=50
NUM_INFERENCE_STEPS=20
REPLAN_STEPS=8
INSTRUCTION_TYPE="seen"
MIXED_PRECISION="bf16"

# Multi-GPU settings
# With 8 GPUs and NUM_WORKERS=4, each worker gets 2 GPUs:
# Worker 0: GPUs 0,1  (Reason1 on 0, Main model on 1)
# Worker 1: GPUs 2,3  (Reason1 on 2, Main model on 3)
# Worker 2: GPUs 4,5  (Reason1 on 4, Main model on 5)
# Worker 3: GPUs 6,7  (Reason1 on 6, Main model on 7)
NUM_WORKERS=4
GPU_START_ID=0

# 20 RoboTwin tasks to evaluate (can run 4 at a time)
TASKS=(
    "adjust_bottle"
    "beat_block_hammer"
    "click_alarmclock"
    "click_bell"
    "grab_roller"
    "handover_block"
    "handover_mic"
    "hanging_mug"
    "lift_pot"
    "move_can_pot"
    "move_pillbottle_pad"
    "move_playingcard_away"
    "move_stapler_pad"
    "open_laptop"
    "open_microwave"
    "pick_diverse_bottles"
    "pick_dual_bottles"
    "place_a2b_left"
    "place_a2b_right"
    "place_bread_basket"
)

# Check if wandb is installed
if ! python -c "import wandb" 2>/dev/null; then
    echo "Installing wandb..."
    pip install wandb
fi

# Login to wandb
wandb login 2>/dev/null || true

echo "=============================================="
echo "Cosmos-WAM RoboTwin Multi-GPU Evaluation"
echo "=============================================="
echo "Total tasks: ${#TASKS[@]}"
echo "Episodes per task: $NUM_EPISODES"
echo "Parallel workers: $NUM_WORKERS"
echo "GPUs per worker: 2 (1 for Reason1, 1 for Main model)"
echo "Total GPUs used: $((NUM_WORKERS * 2))"
echo "WandB Project: $WANDB_PROJECT"
echo "WandB Run: $WANDB_RUN_NAME"
echo "=============================================="

# Verify GPUs are available
echo ""
echo "Checking GPU availability..."
nvidia-smi -L | head -$((NUM_WORKERS * 2))

echo ""
echo "Starting evaluation..."
echo ""

# Run multi-GPU evaluation
python experiments/robotwin/eval_robotwin_wandb_multigpu.py \
    --ckpt "$CKPT" \
    --tasks "${TASKS[@]}" \
    --num_episodes $NUM_EPISODES \
    --num_workers $NUM_WORKERS \
    --gpu_start_id $GPU_START_ID \
    --wandb_project "$WANDB_PROJECT" \
    --wandb_run_name "$WANDB_RUN_NAME" \
    --wandb_tags "bf16" "step14000" "replan8" "multigpu" "8gpu" \
    --robotwin_root "$ROBOTWIN_ROOT" \
    --dataset_stats_path "$DATASET_STATS" \
    --num_inference_steps $NUM_INFERENCE_STEPS \
    --replan_steps $REPLAN_STEPS \
    --instruction_type $INSTRUCTION_TYPE \
    --mixed_precision $MIXED_PRECISION \
    --online_text_encoder_path "$ONLINE_TEXT_ENCODER_PATH"

echo ""
echo "=============================================="
echo "Evaluation Complete!"
echo "Check WandB dashboard for results."
echo "=============================================="
