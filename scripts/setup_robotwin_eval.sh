#!/bin/bash
# Setup script for RoboTwin evaluation

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "Setting up Cosmos-WAM RoboTwin evaluation..."

# Check if RoboTwin is installed
ROBOTWIN_ROOT="/root/linyihan/RoboTwin"
if [ ! -d "$ROBOTWIN_ROOT" ]; then
    echo "Error: RoboTwin not found at ${ROBOTWIN_ROOT}"
    echo "Please clone RoboTwin repository first:"
    echo "  mkdir -p ${PROJECT_ROOT}/third_party"
    echo "  git clone https://github.com/RoboTwin-Platform/RoboTwin.git ${ROBOTWIN_ROOT}"
    exit 1
fi

# Create policy symlink
POLICY_SOURCE="${PROJECT_ROOT}/experiments/robotwin/cosmos_wam_policy"
POLICY_TARGET="${ROBOTWIN_ROOT}/policy/cosmos_wam_policy"

if [ -L "$POLICY_TARGET" ]; then
    echo "Policy symlink already exists: ${POLICY_TARGET}"
elif [ -e "$POLICY_TARGET" ]; then
    echo "Warning: ${POLICY_TARGET} exists but is not a symlink"
    echo "Please remove it manually and re-run this script"
    exit 1
else
    ln -sfn "$POLICY_SOURCE" "$POLICY_TARGET"
    echo "Created symlink: ${POLICY_TARGET} -> ${POLICY_SOURCE}"
fi

echo ""
echo "Setup complete!"
echo ""
echo "To evaluate on RoboTwin, first ensure you have:"
echo "  1. Text embedding cache at: /path/to/text_embeds_cache"
echo "  2. Dataset stats JSON from training"
echo "  3. Model checkpoint"
echo ""
echo "Then run evaluation:"
echo "  # Single task:"
echo "  python experiments/robotwin/eval_robotwin_single.py \\"
echo "    ckpt=./outputs/cosmos_2b_robotwin_TIMESTAMP/checkpoints/step_XXXXX_bf16.pt \\"
echo "    EVALUATION.task_name=click_alarmclock \\"
echo "    EVALUATION.dataset_stats_path=./dataset_stats.json \\"
echo "    EVALUATION.text_embedding_cache_dir=/path/to/text_embeds_cache"
echo ""
echo "  # All tasks (multi-GPU):"
echo "  python experiments/robotwin/run_robotwin_manager.py \\"
echo "    ckpt=./outputs/cosmos_2b_robotwin_TIMESTAMP/checkpoints/step_XXXXX_bf16.pt \\"
echo "    EVALUATION.dataset_stats_path=./dataset_stats.json \\"
echo "    EVALUATION.text_embedding_cache_dir=/path/to/text_embeds_cache \\"
echo "    MULTIRUN.num_gpus=4"
