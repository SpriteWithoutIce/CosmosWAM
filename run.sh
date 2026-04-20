LOG_FILE="log/train_libero_$(date +%Y%m%d_%H%M%S).log"
export CUDA_VISIBLE_DEVICES=2,3
torchrun --nproc_per_node=2 scripts/train.py \
    --config-name train_cosmos_2b_libero \
    # 2>&1 | tee "$LOG_FILE"