#!/bin/bash
# Train Cosmos-WAM on LIBERO dataset
# Usage: bash scripts/train_libero.sh <num_gpus> [additional_args]

set -e

NUM_GPUS="${1:-8}"
shift || true

echo "Training Cosmos-WAM on LIBERO with ${NUM_GPUS} GPUs"

# Run training with torchrun
torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node="${NUM_GPUS}" \
    scripts/train.py \
    --config-name train_cosmos_2b_libero \
    "$@"
