LOG_FILE="log/train_robotwin_$(date +%Y%m%d_%H%M%S).log"

torchrun --nproc_per_node=4 scripts/train_robotwin.py 2>&1 | tee "$LOG_FILE"