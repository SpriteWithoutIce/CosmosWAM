# RoboTwin 批量评估 + WandB 监控

## 新增文件

1. **`experiments/robotwin/eval_robotwin_wandb.py`** - 带 WandB 日志的批量评估脚本
2. **`robotwin_wandb.sh`** - 方便的 shell 脚本，运行10个任务
3. **`docs/wandb_evaluation.md`** - 详细使用文档

## 使用方法

### 1. 快速开始

```bash
cd /home/handoff/Desktop/CosmosWAM
bash robotwin_wandb.sh
```

这会：
- 运行 10 个任务
- 每个任务 50 个 episodes
- 自动上传到 WandB
- 显示实时成功率曲线

### 2. 修改任务列表

编辑 `robotwin_wandb.sh` 中的 `TASKS` 数组：

```bash
TASKS=(
    "adjust_bottle"
    "click_bell"
    # ... 添加或删除任务
)
```

### 3. 自定义 Episode 数量

```bash
NUM_EPISODES=100  # 默认 50
```

## WandB 监控内容

对于每个任务，你会看到：

### 图表

1. **{task_name}/cumulative_success_rate**
   - X轴: Episode 数量
   - Y轴: 累积成功率 (%)

2. **{task_name}/success_count**
   - X轴: Episode 数量  
   - Y轴: 成功次数

### 汇总信息

- `overall/average_success_rate` - 所有任务的平均成功率
- `{task_name}/final_success_rate` - 每个任务的最终成功率
- `{task_name}/final_success_count` - 每个任务的成功次数

## 示例输出

运行后会显示：

```
==============================================
Cosmos-WAM RoboTwin Evaluation with WandB
==============================================
Tasks: 10 tasks
Episodes per task: 50
WandB Project: cosmos-wam-robotwin
WandB Run: cosmos-wam-10tasks-20240408-123456
==============================================

[1/10] Evaluating task: adjust_bottle
...
✓ Task adjust_bottle completed: 68.0% success rate

[2/10] Evaluating task: click_bell
...
✓ Task click_bell completed: 72.0% success rate

...

==============================================
Evaluation Complete!
Average success rate across 10 tasks: 65.4%
Check WandB dashboard for results.
==============================================
```

## 常见问题

### Q: 如何查看 WandB 结果？

运行后会显示 URL：
```
WandB run: https://wandb.ai/your-entity/cosmos-wam-robotwin/runs/abc123
```

点击链接即可查看实时图表。

### Q: 如何对比不同实验？

在 WandB 仪表板中：
1. 点击 "Add to comparison"
2. 选择要对比的 runs
3. 查看叠加的成功率曲线

### Q: 如何只运行单个任务？

```bash
python experiments/robotwin/eval_robotwin_wandb.py \
    --ckpt /mnt/data/linyihan/ckpt/step_0014000_bf16.pt \
    --tasks adjust_bottle \
    --num_episodes 50
```

## 下一步

运行脚本，然后在 WandB 中查看结果！
