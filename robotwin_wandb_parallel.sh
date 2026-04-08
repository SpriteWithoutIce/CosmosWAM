#!/bin/bash
# Cosmos-WAM RoboTwin Parallel Evaluation with WandB
# Uses 8 GPUs to run tasks in parallel (2 GPUs per task)

set -e

# Check if running in conda environment
if [ -z "$CONDA_DEFAULT_ENV" ]; then
    echo "Warning: Not running in a conda environment. Make sure wandb is installed."
fi

# Configuration
CKPT="/mnt/data/linyihan/ckpt/step_0014000_bf16.pt"
ROBOTWIN_ROOT="/root/linyihan/RoboTwin"
DATASET_STATS="./dataset_stats.json"
ONLINE_TEXT_ENCODER_PATH="/mnt/data/linyihan/Cosmos-Reason1-7b"

# WandB settings
WANDB_PROJECT="cosmos-wam-robotwin"
WANDB_ENTITY=""
WANDB_RUN_NAME="cosmos-wam-parallel-$(date +%Y%m%d-%H%M%S)"

# Evaluation settings
NUM_EPISODES=50
NUM_INFERENCE_STEPS=20
REPLAN_STEPS=8
INSTRUCTION_TYPE="seen"
MIXED_PRECISION="bf16"

# GPU Configuration
NUM_PARALLEL=4
GPU_PAIRS=("0,1" "2,3" "4,5" "6,7")

# Tasks to evaluate
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
)

# Create output directory
OUTPUT_DIR="./evaluate_results/parallel_${WANDB_RUN_NAME}"
mkdir -p "$OUTPUT_DIR"

# Check dependencies
if ! python -c "import wandb" 2>/dev/null; then
    echo "Error: wandb is not installed. Please install it first:"
    echo "  pip install wandb"
    echo "or if using conda:"
    echo "  conda install -c conda-forge wandb"
    exit 1
fi

# Login to wandb
wandb login 2>/dev/null || true

echo "=============================================="
echo "Cosmos-WAM RoboTwin Parallel Evaluation"
echo "=============================================="
echo "Total tasks: ${#TASKS[@]}"
echo "Parallel jobs: $NUM_PARALLEL"
echo "GPUs per job: 2"
echo "Total GPUs used: $((NUM_PARALLEL * 2))"
echo "Output dir: $OUTPUT_DIR"
echo "=============================================="

# Show GPU status
echo ""
echo "GPU Status:"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv | head -9

# Create a Python script for the WandB logger
cat > "$OUTPUT_DIR/wandb_logger.py" << 'EOF'
import sys
import re
import wandb
import os

def main():
    task_name = sys.argv[1]
    log_file = sys.argv[2]
    
    # Connect to wandb
    run_id = os.environ.get("WANDB_RUN_ID")
    wandb.init(project=os.environ.get("WANDB_PROJECT"), id=run_id, resume="must")
    
    with open(log_file, "r") as f:
        for line in f:
            match = re.search(r"Success rate:\s*(\d+)/(\d+)\s*=>\s*([\d.]+)%", line)
            if match:
                suc, test_num, rate = int(match.group(1)), int(match.group(2)), float(match.group(3))
                wandb.log({
                    f"{task_name}/cumulative_success_rate": rate,
                    f"{task_name}/success_count": suc,
                    f"{task_name}/episode": test_num,
                }, step=test_num)
    
    wandb.finish()

if __name__ == "__main__":
    main()
EOF

# Initialize WandB run
python3 << EOF
import wandb

wandb.init(
    project="$WANDB_PROJECT",
    entity="$WANDB_ENTITY" or None,
    name="$WANDB_RUN_NAME",
    tags=["bf16", "step14000", "replan8", "parallel", "8gpu"],
    config={
        "checkpoint": "$CKPT",
        "num_episodes": $NUM_EPISODES,
        "num_parallel": $NUM_PARALLEL,
        "num_inference_steps": $NUM_INFERENCE_STEPS,
        "replan_steps": $REPLAN_STEPS,
    },
)
print(f"WandB run initialized: {wandb.run.url}")
# Save run ID for child processes
with open("$OUTPUT_DIR/wandb_run_id.txt", "w") as f:
    f.write(wandb.run.id)
EOF

# Save WANDB_RUN_ID and PROJECT for child processes
export WANDB_RUN_ID=$(cat "$OUTPUT_DIR/wandb_run_id.txt")
export WANDB_PROJECT="$WANDB_PROJECT"

# Create task list file
TASK_FILE="$OUTPUT_DIR/task_list.txt"
printf '%s\n' "${TASKS[@]}" > "$TASK_FILE"

# Function to run a single task
run_task() {
    local TASK_NAME=$1
    local GPU_PAIR=$2
    local TASK_IDX=$3
    
    local TEXT_ENCODER_GPU=$(echo $GPU_PAIR | cut -d',' -f1)
    local MODEL_GPU=$(echo $GPU_PAIR | cut -d',' -f2)
    local LOG_FILE="$OUTPUT_DIR/${TASK_NAME}_gpu${TEXT_ENCODER_GPU}${MODEL_GPU}.log"
    
    echo "[$(date)] Starting task $TASK_IDX: $TASK_NAME on GPUs $GPU_PAIR"
    
    export CUDA_VISIBLE_DEVICES="$GPU_PAIR"
    export PYTHONUNBUFFERED=1
    
    # Run the task
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
        > "$LOG_FILE" 2>&1
    
    local EXIT_CODE=$?
    
    # Log to wandb
    WANDB_PROJECT="$WANDB_PROJECT" WANDB_RUN_ID="$WANDB_RUN_ID" \
        python3 "$OUTPUT_DIR/wandb_logger.py" "$TASK_NAME" "$LOG_FILE" &
    
    echo "[$(date)] Task $TASK_NAME completed with exit code $EXIT_CODE"
    return $EXIT_CODE
}

export -f run_task
export CKPT ROBOTWIN_ROOT DATASET_STATS ONLINE_TEXT_ENCODER_PATH
export NUM_EPISODES NUM_INFERENCE_STEPS REPLAN_STEPS INSTRUCTION_TYPE MIXED_PRECISION
export OUTPUT_DIR WANDB_PROJECT WANDB_RUN_ID

# Run tasks in parallel
echo ""
echo "Starting parallel evaluation..."
echo ""

idx=0
for task in "${TASKS[@]}"; do
    gpu_pair=${GPU_PAIRS[$((idx % NUM_PARALLEL))]}
    run_task "$task" "$gpu_pair" $((idx + 1)) &
    
    idx=$((idx + 1))
    
    # Wait if we've launched NUM_PARALLEL jobs
    if [ $((idx % NUM_PARALLEL)) -eq 0 ] && [ $idx -lt ${#TASKS[@]} ]; then
        echo "Waiting for batch to complete..."
        wait
    fi
done

# Wait for all jobs
echo "Waiting for all tasks to complete..."
wait

# Finalize WandB
python3 << EOF
import wandb
import os
import glob
import re

run_id = open("$OUTPUT_DIR/wandb_run_id.txt").read().strip()
wandb.init(project="$WANDB_PROJECT", id=run_id, resume="must")

# Compute overall statistics
results = []
for log_file in glob.glob("$OUTPUT_DIR/*.log"):
    with open(log_file) as f:
        content = f.read()
        matches = re.findall(r"Success rate:\s*(\d+)/(\d+)\s*=>\s*([\d.]+)%", content)
        if matches:
            suc, total, rate = matches[-1]
            results.append(float(rate))

if results:
    avg_rate = sum(results) / len(results)
    wandb.summary["overall/average_success_rate"] = avg_rate
    wandb.summary["overall/num_tasks"] = len(results)
    print(f"Average success rate: {avg_rate:.1f}%")

wandb.finish()
EOF

echo ""
echo "=============================================="
echo "All evaluations completed!"
echo "Results saved to: $OUTPUT_DIR"
echo "=============================================="
