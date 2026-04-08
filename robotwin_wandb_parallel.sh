#!/bin/bash
# Cosmos-WAM RoboTwin Parallel Evaluation with WandB
# Uses 8 GPUs to run tasks in parallel (2 GPUs per task)
# This version uses GNU parallel for better process management

set -e

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
# With 8 GPUs, we can run 4 tasks in parallel
# Each task uses: 1 GPU for Reason1, 1 GPU for Main model
NUM_PARALLEL=4
GPU_PAIRS=("0,1" "2,3" "4,5" "6,7")

# Tasks to evaluate (will be distributed across GPU pairs)
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

# Create output directory
OUTPUT_DIR="./evaluate_results/parallel_${WANDB_RUN_NAME}"
mkdir -p "$OUTPUT_DIR"

# Check dependencies
if ! command -v parallel &> /dev/null; then
    echo "Installing GNU parallel..."
    apt-get update && apt-get install -y parallel || true
fi

if ! python -c "import wandb" 2>/dev/null; then
    echo "Installing wandb..."
    pip install wandb
fi

# Login to wandb
wandb login 2>/dev/null || true

echo "=============================================="
echo "Cosmos-WAM RoboTwin Parallel Evaluation"
echo "=============================================="
echo "Total tasks: ${#TASKS[@]}"
echo "Parallel jobs: $NUM_PARALLEL"
echo "GPUs per job: 2"
echo "Total GPUs: 8"
echo "Output dir: $OUTPUT_DIR"
echo "=============================================="

# Show GPU status
echo ""
echo "GPU Status:"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv | head -9

# Initialize WandB run
python3 << EOF
import wandb
import os

wandb.init(
    project="$WANDB_PROJECT",
    entity="$WANDB_ENTITY" or None,
    name="$WANDB_RUN_NAME",
    tags=["bf16", "step14000", "replan8", "parallel", "8gpu"],
    config={
        "checkpoint": "$CKPT",
        "tasks": $(printf '%s\n' "${TASKS[@]}" | jq -R . | jq -s .),
        "num_episodes": $NUM_EPISODES,
        "num_parallel": $NUM_PARALLEL,
        "num_inference_steps": $NUM_INFERENCE_STEPS,
        "replan_steps": $REPLAN_STEPS,
    },
)
print(f"WandB run initialized")
EOF

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
    
    # Create a Python script for this task with WandB logging
    python3 << PYEOF
import subprocess
import sys
import re
import wandb
import os

# Re-initialize wandb in this process
wandb.init(project="$WANDB_PROJECT", id=os.environ.get("WANDB_RUN_ID"), resume="must")

cmd = [
    sys.executable,
    "experiments/robotwin/eval_robotwin_single.py",
    f"ckpt=$CKPT",
    f"EVALUATION.task_name=$TASK_NAME",
    f"EVALUATION.eval_num_episodes=$NUM_EPISODES",
    f"EVALUATION.robotwin_root=$ROBOTWIN_ROOT",
    f"EVALUATION.dataset_stats_path=$DATASET_STATS",
    f"EVALUATION.num_inference_steps=$NUM_INFERENCE_STEPS",
    f"EVALUATION.replan_steps=$REPLAN_STEPS",
    f"EVALUATION.instruction_type=$INSTRUCTION_TYPE",
    f"mixed_precision=$MIXED_PRECISION",
    f"gpu_id=$MODEL_GPU",
    "EVALUATION.use_online_text_encoder=true",
    f"EVALUATION.online_text_encoder_path=$ONLINE_TEXT_ENCODER_PATH",
    "EVALUATION.text_encoder_device=cuda:0",
    "EVALUATION.device=cuda:1",
]

process = subprocess.Popen(
    cmd,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    bufsize=1,
)

with open("$LOG_FILE", "w") as f:
    for line in process.stdout:
        f.write(line)
        f.flush()
        print(f"[$TASK_NAME] {line}", end="")
        
        # Parse success rate
        match = re.search(r"Success rate:\s*(\d+)/(\d+)\s*=>\s*([\d.]+)%", line)
        if match:
            suc, test_num, rate = int(match.group(1)), int(match.group(2)), float(match.group(3))
            wandb.log({
                f"${TASK_NAME}/cumulative_success_rate": rate,
                f"${TASK_NAME}/success_count": suc,
                f"${TASK_NAME}/episode": test_num,
            }, step=test_num)

process.wait()
exit(process.returncode)
PYEOF
    
    local EXIT_CODE=$?
    echo "[$(date)] Task $TASK_NAME completed with exit code $EXIT_CODE"
    return $EXIT_CODE
}

export -f run_task
export CKPT ROBOTWIN_ROOT DATASET_STATS ONLINE_TEXT_ENCODER_PATH
export NUM_EPISODES NUM_INFERENCE_STEPS REPLAN_STEPS INSTRUCTION_TYPE MIXED_PRECISION
export OUTPUT_DIR WANDB_PROJECT

# Save WANDB_RUN_ID for child processes
export WANDB_RUN_ID=$(python3 -c "import wandb; print(wandb.run.id)")

# Run tasks in parallel using GNU parallel
echo ""
echo "Starting parallel evaluation..."
echo ""

if command -v parallel &> /dev/null; then
    # Use GNU parallel
    parallel --jobs $NUM_PARALLEL --line-buffer \
        --joblog "$OUTPUT_DIR/parallel.log" \
        run_task {1} {2} {#} \
        :::: "$TASK_FILE" \
        ::: "${GPU_PAIRS[@]}"
else
    # Fallback: manual background process management
    echo "GNU parallel not available, using manual process management..."
    
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
    
    # Wait for remaining jobs
    wait
fi

# Finalize WandB
python3 << EOF
import wandb
import os
import json

wandb.init(project="$WANDB_PROJECT", id=os.environ.get("WANDB_RUN_ID"), resume="must")

# Compute overall statistics
results = []
import glob
for log_file in glob.glob("$OUTPUT_DIR/*.log"):
    with open(log_file) as f:
        content = f.read()
        # Find final success rate
        matches = re.findall(r"Success rate:\s*(\d+)/(\d+)\s*=>\s*([\d.]+)%", content)
        if matches:
            suc, total, rate = matches[-1]
            results.append(float(rate))

if results:
    avg_rate = sum(results) / len(results)
    wandb.summary["overall/average_success_rate"] = avg_rate
    wandb.summary["overall/num_tasks"] = len(results)

wandb.finish()
print(f"Average success rate: {avg_rate:.1f}%" if results else "No results")
EOF

echo ""
echo "=============================================="
echo "All evaluations completed!"
echo "Results saved to: $OUTPUT_DIR"
echo "WandB run: $WANDB_RUN_NAME"
echo "=============================================="
