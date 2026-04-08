# 多 GPU 并行评估指南

如果你有 8 个 GPU，可以使用多 GPU 并行评估来加速 RoboTwin 测试。

## 硬件配置建议

| GPU 配置 | 设置 |
|---------|------|
| 8 x RTX 4090 (24GB) | 运行 4 个并行任务 |
| 8 x A100 (40GB/80GB) | 运行 4-8 个并行任务 |

每个任务需要 2 个 GPU：
- GPU 0: Cosmos-Reason1-7B 文本编码器 (~14GB)
- GPU 1: Cosmos-WAM 主模型 (~10GB)

## 快速开始

### 方法 1: 使用并行脚本（推荐）

```bash
cd /home/handoff/Desktop/CosmosWAM
bash robotwin_wandb_parallel.sh
```

这会自动：
1. 将 20 个任务分配给 4 个并行工作进程
2. 每个工作进程使用 2 个 GPU
3. 实时上传结果到 WandB

### 方法 2: 使用 Python 多进程

```bash
python experiments/robotwin/eval_robotwin_wandb_multigpu.py \
    --ckpt /mnt/data/linyihan/ckpt/step_0014000_bf16.pt \
    --tasks adjust_bottle click_bell grab_roller handover_block \
           lift_pot open_laptop pick_diverse_bottles place_bread_basket \
           click_alarmclock beat_block_hammer move_can_pot move_pillbottle_pad \
           move_playingcard_away move_stapler_pad open_microwave \
           pick_dual_bottles place_a2b_left place_a2b_right \
           hanging_mug handover_mic \
    --num_episodes 50 \
    --num_workers 4 \
    --gpu_start_id 0 \
    --wandb_project cosmos-wam-robotwin \
    --online_text_encoder_path /mnt/data/linyihan/Cosmos-Reason1-7b
```

### 方法 3: 手动分配任务

```bash
# Terminal 1 - GPUs 0,1
export CUDA_VISIBLE_DEVICES=0,1
python experiments/robotwin/eval_robotwin_single.py ...

# Terminal 2 - GPUs 2,3
export CUDA_VISIBLE_DEVICES=2,3
python experiments/robotwin/eval_robotwin_single.py ...

# Terminal 3 - GPUs 4,5
export CUDA_VISIBLE_DEVICES=4,5
python experiments/robotwin/eval_robotwin_single.py ...

# Terminal 4 - GPUs 6,7
export CUDA_VISIBLE_DEVICES=6,7
python experiments/robotwin/eval_robotwin_single.py ...
```

## GPU 分配

默认配置（8 GPU 系统）：

| Worker | Reason1 GPU | Main Model GPU | Tasks |
|--------|------------|----------------|-------|
| 0 | 0 | 1 | Task 1, 5, 9, 13, 17 |
| 1 | 2 | 3 | Task 2, 6, 10, 14, 18 |
| 2 | 4 | 5 | Task 3, 7, 11, 15, 19 |
| 3 | 6 | 7 | Task 4, 8, 12, 16, 20 |

## 自定义 GPU 分配

如果你有 4 个 GPU：

```bash
python experiments/robotwin/eval_robotwin_wandb_multigpu.py \
    --num_workers 2 \
    --gpu_start_id 0 \
    ...
```

如果你有 16 个 GPU：

```bash
python experiments/robotwin/eval_robotwin_wandb_multigpu.py \
    --num_workers 8 \
    --gpu_start_id 0 \
    ...
```

## 监控进度

### 查看 GPU 使用率

```bash
watch -n 1 nvidia-smi
```

### 查看 WandB 实时结果

运行后会显示 WandB URL：
```
WandB run: https://wandb.ai/your-username/cosmos-wam-robotwin/runs/abc123
```

每个任务会有独立的图表，显示累积成功率。

## 性能对比

假设每个任务 50 episodes：

| 配置 | 并行度 | 预估时间 | 加速比 |
|------|--------|---------|--------|
| 单 GPU | 1 | ~50 分钟 | 1x |
| 4 GPU (2 workers) | 2 | ~25 分钟 | 2x |
| 8 GPU (4 workers) | 4 | ~13 分钟 | 4x |

*实际时间取决于任务复杂度和硬件性能*

## 故障排除

### 问题: CUDA out of memory

可能原因：
1. GPU 被其他进程占用
2. 显存碎片

解决方案：
```bash
# 清理 GPU 显存
sudo fuser -v /dev/nvidia*  # 查看占用进程
kill -9 <pid>               # 杀死占用进程

# 或者重启 Python 进程
```

### 问题: WandB 连接失败

```bash
# 检查网络连接
wandb login --relogin

# 或者使用离线模式
export WANDB_MODE=offline
```

### 问题: 某些任务失败

查看日志文件：
```bash
tail -f evaluate_results/parallel_*/<task_name>_gpu*.log
```

## 高级配置

### 修改并行任务数

编辑 `robotwin_wandb_parallel.sh`：

```bash
NUM_PARALLEL=2  # 改为 2，只使用 4 个 GPU
GPU_PAIRS=("0,1" "2,3")  # 只使用前 4 个 GPU
```

### 为特定任务指定 GPU

```bash
# 创建一个任务配置文件
cat > tasks_gpu0.txt << EOF
adjust_bottle
click_bell
EOF

# 只在 GPU 0,1 上运行这些任务
parallel --jobs 1 run_task {1} 0,1 ::: $(cat tasks_gpu0.txt)
```

## 推荐工作流

1. **预计算 text embeddings**（可选，节省显存）：
   ```bash
   python scripts/precompute_text_embeddings.py \
       --dataset_root /path/to/RoboTwin \
       --model_path /path/to/Cosmos-Reason1-7B \
       --output_dir /path/to/cache
   ```

2. **单 GPU 测试**：
   ```bash
   bash robotwin.sh  # 测试单个任务
   ```

3. **多 GPU 批量评估**：
   ```bash
   bash robotwin_wandb_parallel.sh  # 评估所有任务
   ```

4. **分析结果**：
   - 在 WandB 中查看所有任务的累积成功率曲线
   - 对比不同配置的效果
