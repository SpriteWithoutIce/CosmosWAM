# WandB 批量评估指南

这个文档介绍如何使用 WandB 批量运行 RoboTwin 评估并实时监控成功率。

## 功能特点

- **批量任务评估**: 一次运行10个任务，每个任务50个episode
- **实时监控**: 在 WandB 仪表板中查看每个任务的累积成功率曲线
- **自动日志**: 所有结果自动上传到 WandB，方便对比和分享

## 快速开始

### 1. 安装依赖

```bash
pip install wandb
```

### 2. 登录 WandB

```bash
wandb login
```

### 3. 运行评估

```bash
cd /home/handoff/Desktop/CosmosWAM
bash robotwin_wandb.sh
```

或者使用 Python 脚本直接运行：

```bash
python experiments/robotwin/eval_robotwin_wandb.py \
    --ckpt /mnt/data/linyihan/ckpt/step_0014000_bf16.pt \
    --tasks adjust_bottle click_bell grab_roller handover_block lift_pot \
           open_laptop pick_diverse_bottles place_bread_basket \
           click_alarmclock beat_block_hammer \
    --num_episodes 50 \
    --wandb_project cosmos-wam-robotwin \
    --wandb_run_name my-experiment \
    --use_online_text_encoder \
    --online_text_encoder_path /mnt/data/linyihan/Cosmos-Reason1-7b
```

## WandB 仪表板

运行后会显示 WandB run URL，例如：

```
WandB run: https://wandb.ai/your-entity/cosmos-wam-robotwin/runs/abc123
```

### 查看的指标

每个任务会有以下图表：

1. **Cumulative Success Rate** (`{task_name}/cumulative_success_rate`)
   - X轴: Episode 数量
   - Y轴: 累积成功率 (%)

2. **Success Count** (`{task_name}/success_count`)
   - X轴: Episode 数量
   - Y轴: 成功次数

### 示例图表

```
Success Rate (%)
    │
100 ┤                    ╭────
    │               ╭────╯
 80 ┤          ╭────╯
    │     ╭────╯
 60 ┤╭────╯
    │
 40 ┤
    │
 20 ┤
    │
  0 ┼────┬────┬────┬────┬────
    0   10   20   30   40   50
              Episode
```

## 自定义配置

### 修改任务列表

编辑 `robotwin_wandb.sh` 中的 `TASKS` 数组：

```bash
TASKS=(
    "adjust_bottle"
    "click_bell"
    # 添加更多任务...
)
```

### 修改 Episode 数量

```bash
NUM_EPISODES=100  # 默认是 50
```

### 修改 WandB 项目

```bash
WANDB_PROJECT="your-project-name"
WANDB_ENTITY="your-username"  # 可选
```

## 对比不同配置

你可以运行多次实验，WandB 会自动对比：

```bash
# 实验1: replan_steps=4
bash robotwin_wandb.sh  # 修改 REPLAN_STEPS=4

# 实验2: replan_steps=8
bash robotwin_wandb.sh  # 修改 REPLAN_STEPS=8
```

然后在 WandB 仪表板中使用 "Add to comparison" 功能对比不同配置的成功率曲线。

## 故障排除

### WandB 登录问题

如果提示未登录：

```bash
wandb login
# 按照提示输入 API key
```

### GPU 显存不足

如果显存不足，可以：

1. 减少 `NUM_EPISODES`
2. 使用更小的模型
3. 使用 `fp16` 而不是 `bf16`

### 任务不存在

如果提示任务不存在，检查任务名称拼写：

```bash
ls /root/linyihan/RoboTwin/envs/*.py | xargs -n1 basename -s .py
```

## 保存和导出结果

WandB 自动保存所有结果。你可以：

1. **导出 CSV**: 在 WandB 仪表板中点击 "Download CSV"
2. **生成报告**: 使用 WandB 的报告功能创建可视化报告
3. **对比实验**: 使用 WandB 的 "Workspace" 功能对比不同运行

## 高级用法

### 添加自定义标签

```bash
python experiments/robotwin/eval_robotwin_wandb.py \
    --wandb_tags "experiment_1" "baseline" "seed_42"
```

### 只运行特定任务

```bash
python experiments/robotwin/eval_robotwin_wandb.py \
    --tasks adjust_bottle click_bell
```

### 禁用在线 text encoder

```bash
python experiments/robotwin/eval_robotwin_wandb.py \
    --no-use_online_text_encoder
```
