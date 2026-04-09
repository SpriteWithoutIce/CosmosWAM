#!/bin/bash
# Cosmos-WAM LIBERO Batch Evaluation Script
# Evaluates multiple tasks in a suite

# Configuration
CKPT="/home/jwhe/linyihan/CosmosWAM/outputs/cosmos_2b_libero_20260408_125315/checkpoints/step_0004000.pt"
DATASET_STATS="./dataset_stats.json"
TEXT_EMBED_CACHE="./data/text_embeds_cache/libero"

# LIBERO library path (can be overridden via environment)
export LIBERO_PATH="${LIBERO_PATH:-/home/jwhe/linyihan/LIBERO}"

# Task suite
TASK_SUITE="libero_spatial"  # Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
NUM_TRIALS=50
NUM_INFERENCE_STEPS=4
REPLAN_STEPS=5
ACTION_HORIZON=32

# Number of tasks in the suite (libero_spatial, libero_object, libero_goal have 10 tasks each)
NUM_TASKS=10

echo "=============================================="
echo "Cosmos-WAM LIBERO Batch Evaluation"
echo "=============================================="
echo "LIBERO Path: $LIBERO_PATH"
echo "Task Suite: $TASK_SUITE"
echo "Num Tasks: $NUM_TASKS"
echo "Num Trials per Task: $NUM_TRIALS"
echo "Checkpoint: $CKPT"
echo "=============================================="

for TASK_ID in $(seq 0 $((NUM_TASKS - 1))); do
    echo ""
    echo "[$((TASK_ID + 1))/$NUM_TASKS] Evaluating Task $TASK_ID..."
    echo ""
    
    python experiments/libero/eval_libero_single.py \
        ckpt="$CKPT" \
        EVALUATION.task_suite_name="$TASK_SUITE" \
        EVALUATION.task_id=$TASK_ID \
        EVALUATION.num_trials=$NUM_TRIALS \
        EVALUATION.dataset_stats_path="$DATASET_STATS" \
        EVALUATION.text_embedding_cache_dir="$TEXT_EMBED_CACHE" \
        EVALUATION.num_inference_steps=$NUM_INFERENCE_STEPS \
        EVALUATION.replan_steps=$REPLAN_STEPS \
        EVALUATION.action_hoirzon=$ACTION_HORIZON \
        EVALUATION.gpu_id=0 \
        mixed_precision=bf16
done

echo ""
echo "=============================================="
echo "Batch Evaluation Complete!"
echo "Results saved to: ./evaluate_results/libero/$TASK_SUITE/"
echo "=============================================="

# Compute average success rate
echo ""
echo "Computing statistics..."
python3 << EOF
import json
import glob
from pathlib import Path

results_dir = Path("./evaluate_results/libero") / "$TASK_SUITE"
result_files = list(results_dir.glob("*_results.json"))

if not result_files:
    print(f"No result files found in {results_dir}")
    exit(1)

total_successes = 0
total_trials = 0

for f in sorted(result_files):
    with open(f) as fp:
        data = json.load(fp)
    successes = data.get("successes", 0)
    total = data.get("total_episodes", 50)
    rate = successes / total * 100
    total_successes += successes
    total_trials += total
    print(f"Task {data.get('task_id', '?'):2d}: {successes:2d}/{total} = {rate:5.1f}%")

overall_rate = total_successes / total_trials * 100
print(f"\nOverall: {total_successes}/{total_trials} = {overall_rate:.1f}%")
EOF
