#!/bin/bash
# Cosmos-WAM RoboTwin Evaluation with WandB Logging
# Evaluates 10 tasks, 50 episodes each, and logs cumulative success rates to WandB

# Configuration
CKPT="/mnt/data/linyihan/ckpt/step_0014000_bf16.pt"
ROBOTWIN_ROOT="/root/linyihan/RoboTwin"
DATASET_STATS="./dataset_stats.json"
ONLINE_TEXT_ENCODER_PATH="/mnt/data/linyihan/Cosmos-Reason1-7b"

# WandB settings
WANDB_PROJECT="cosmos-wam-robotwin"
WANDB_ENTITY=""  # Leave empty to use default entity, or set your wandb username/team
WANDB_RUN_NAME="cosmos-wam-10tasks-$(date +%Y%m%d-%H%M%S)"

# Evaluation settings
NUM_EPISODES=50
NUM_INFERENCE_STEPS=20
REPLAN_STEPS=8
INSTRUCTION_TYPE="seen"
MIXED_PRECISION="bf16"
GPU_ID=1

# 10 RoboTwin tasks to evaluate
# Available tasks in RoboTwin:
# - adjust_bottle
# - beat_block_hammer
# - blocks_ranking_rgb
# - blocks_ranking_size
# - click_alarmclock
# - click_bell
# - dump_bin_bigbin
# - grab_roller
# - handover_block
# - handover_mic
# - hanging_mug
# - lift_pot
# - move_can_pot
# - move_pillbottle_pad
# - move_playingcard_away
# - move_stapler_pad
# - open_laptop
# - open_microwave
# - pick_diverse_bottles
# - pick_dual_bottles
# - place_a2b_left
# - place_a2b_right
# - place_bread_basket
# - place_bread_skillet
# - place_burger_fries
# - place_can_basket
# - (and more...)

TASKS=(
    "adjust_bottle"
    "beat_block_hammer"
    # "blocks_ranking_rgb"
    # "blocks_ranking_size"
    # "click_alarmclock"
    # "click_bell"
    # "dump_bin_bigbin"
    # "grab_roller"
    # "handover_block"
    # "handover_mic"
)

# Check if wandb is installed
if ! python -c "import wandb" 2>/dev/null; then
    echo "Installing wandb..."
    pip install wandb
fi

# Login to wandb (optional - will prompt if not already logged in)
wandb login 2>/dev/null || true

echo "=============================================="
echo "Cosmos-WAM RoboTwin Evaluation with WandB"
echo "=============================================="
echo "Tasks: ${#TASKS[@]} tasks"
echo "Episodes per task: $NUM_EPISODES"
echo "WandB Project: $WANDB_PROJECT"
echo "WandB Run: $WANDB_RUN_NAME"
echo "=============================================="

# Run evaluation
python experiments/robotwin/eval_robotwin_wandb.py \
    --ckpt "$CKPT" \
    --tasks "${TASKS[@]}" \
    --num_episodes $NUM_EPISODES \
    --wandb_project "$WANDB_PROJECT" \
    --wandb_run_name "$WANDB_RUN_NAME" \
    --wandb_tags "bf16" "step14000" "replan8" \
    --gpu_id $GPU_ID \
    --robotwin_root "$ROBOTWIN_ROOT" \
    --dataset_stats_path "$DATASET_STATS" \
    --num_inference_steps $NUM_INFERENCE_STEPS \
    --replan_steps $REPLAN_STEPS \
    --instruction_type $INSTRUCTION_TYPE \
    --mixed_precision $MIXED_PRECISION \
    --use_online_text_encoder \
    --online_text_encoder_path "$ONLINE_TEXT_ENCODER_PATH" \
    --text_encoder_device "cuda:0" \
    --model_device "cuda:1"

echo ""
echo "=============================================="
echo "Evaluation Complete!"
echo "Syncing WandB data..."
wandb sync 2>/dev/null || true
echo "Check WandB dashboard for results."
echo "==============================================
