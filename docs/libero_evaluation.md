# LIBERO 评估指南

## 检查结论

**timestep 问题已修复**：`cosmos_wam/models/cosmos_wam.py` 中的 `infer_action` 函数已使用正确的 timestep 方向：
```python
# 正确：t 从 0 升到 1
for i in range(num_inference_steps):
    t = torch.full((B,), i / num_inference_steps, ...)  # t=0: noise, t=1: action
```

## 快速开始

### 1. 单个任务测试

```bash
cd /home/handoff/Desktop/CosmosWAM
bash libero.sh
```

默认测试 `libero_spatial` 套件的第 0 个任务。

### 2. 批量测试所有任务

```bash
bash libero_batch.sh
```

这会依次测试 `libero_spatial` 套件的全部 10 个任务。

## 配置说明

### 任务套件 (Task Suite)

| 套件名 | 任务数 | 描述 |
|--------|--------|------|
| `libero_spatial` | 10 | 空间关系任务 |
| `libero_object` | 10 | 物体操作任务 |
| `libero_goal` | 10 | 目标达成任务 |
| `libero_10` | 10 | 10个精选任务 |
| `libero_90` | 90 | 90个完整任务 |

### 关键参数

编辑 `libero.sh` 或命令行覆盖：

```bash
python experiments/libero/eval_libero_single.py \
    ckpt=/path/to/checkpoint.pt \
    EVALUATION.task_suite_name=libero_spatial \
    EVALUATION.task_id=0 \
    EVALUATION.num_trials=50 \
    EVALUATION.dataset_stats_path=./dataset_stats.json \
    EVALUATION.text_embedding_cache_dir=./data/text_embeds_cache/libero \
    mixed_precision=bf16
```

### 性能参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `num_inference_steps` | 4 | 扩散采样步数 (LIBERO 通常 4-8) |
| `replan_steps` | 5 | 每次重规划执行的动作数 |
| `action_horizon` | 32 | 预测的动作序列长度 |
| `num_trials` | 50 | 每个任务的测试次数 |

## 输出结果

结果保存在 `./evaluate_results/libero/{task_suite}/`：

```
evaluate_results/libero/libero_spatial/
├── gpu0_task0_results.json      # 任务结果统计
├── gpu0_task1_results.json
├── ...
└── videos/                       # 视频记录
    ├── task0_trial0_success.mp4
    ├── task0_trial1_fail.mp4
    └── ...
```

### 结果文件格式

```json
{
    "task_suite": "libero_spatial",
    "task_id": 0,
    "task_description": "pick up the ...",
    "successes": 45,
    "total_episodes": 50,
    "success_episodes": [0, 1, 2, ...],
    "failure_episodes": [5, 12, ...],
    "duration": 1234.5
}
```

## 批量测试结果统计

运行 `libero_batch.sh` 后会自动计算：

```
Task  0: 45/50 =  90.0%
Task  1: 42/50 =  84.0%
Task  2: 38/50 =  76.0%
...
Overall: 415/500 = 83.0%
```

## 故障排除

### 问题：text embedding cache 不存在

```bash
# 需要预计算 LIBERO 的 text embeddings
python scripts/precompute_text_embeddings.py \
    --dataset_root /path/to/libero \
    --model_path /path/to/Cosmos-Reason1-7B \
    --output_dir ./data/text_embeds_cache/libero
```

### 问题：CUDA out of memory

减小 `action_horizon` 或 `num_inference_steps`：

```bash
EVALUATION.action_horizon=16 \
EVALUATION.num_inference_steps=4
```

### 问题：dataset_stats.json 不存在

需要准备数据集统计文件（包含动作归一化参数）。

## 与 RoboTwin 的区别

| 特性 | LIBERO | RoboTwin |
|------|--------|----------|
| 动作维度 | 7 (eef 6 + gripper 1) | 16 (双臂) |
| 图像尺寸 | 224x224 | 240x320 |
| 相机数量 | 1 (agentview) | 1-2 |
| 推理步数 | 4-8 | 20 |
| 重规划步数 | 5 | 4-8 |

## 预期成功率

基于训练配置和 FastWAM 的 LIBERO 结果，预期成功率：

| 套件 | 预期成功率 |
|------|-----------|
| libero_spatial | 85-95% |
| libero_object | 80-90% |
| libero_goal | 75-85% |

如果成功率显著低于预期（<50%），可能是：
1. timestep 方向问题（已修复）
2. checkpoint 未正确加载
3. 动作归一化参数不匹配
4. Text embedding 不匹配
