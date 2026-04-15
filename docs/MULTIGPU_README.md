# 多 GPU 批量评估

## 新增文件

| 文件 | 说明 |
|------|------|
| `experiments/robotwin/eval_robotwin_wandb_multigpu.py` | Python 多进程多 GPU 评估 |
| `experiments/robotwin/run_single_task.sh` | 单任务 GPU 分配脚本 |
| `robotwin_wandb_multigpu.sh` | 多 GPU 评估启动脚本（Python 版） |
| `robotwin_wandb_parallel.sh` | 多 GPU 评估启动脚本（GNU parallel 版） |
| `docs/multigpu_evaluation.md` | 多 GPU 使用文档 |

## 快速使用

### 8 GPU 系统

```bash
cd /home/handoff/Desktop/CosmosWAM
bash robotwin_wandb_parallel.sh
```

这会：
- 使用 8 个 GPU
- 运行 4 个并行任务
- 每个任务评估 50 episodes
- 实时上传 WandB

## GPU 分配

```
Worker 0: GPUs 0,1 (Reason1 on 0, Main model on 1)
Worker 1: GPUs 2,3 (Reason1 on 2, Main model on 3)
Worker 2: GPUs 4,5 (Reason1 on 4, Main model on 5)
Worker 3: GPUs 6,7 (Reason1 on 6, Main model on 7)
```

## 自定义任务

编辑 `robotwin_wandb_parallel.sh` 中的 `TASKS` 数组：

```bash
TASKS=(
    "adjust_bottle"
    "click_bell"
    # 添加更多任务...
)
```

## 性能

| GPU 数量 | 并行任务 | 20任务耗时 | 加速比 |
|---------|---------|-----------|--------|
| 2 (1组) | 1 | ~17小时 | 1x |
| 4 (2组) | 2 | ~8.5小时 | 2x |
| 8 (4组) | 4 | ~4.3小时 | 4x |

## WandB 监控

运行后查看实时图表：
- 每个任务的累积成功率曲线
- 实时更新的成功次数
- 自动计算的总体平均成功率

访问: https://wandb.ai/your-username/cosmos-wam-robotwin
