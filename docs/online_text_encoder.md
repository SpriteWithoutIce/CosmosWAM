# 在线 Text Encoder 使用指南

如果你在使用 24GB VRAM 的 GPU（如 RTX 4090）进行 RoboTwin 推理，可以启用**在线 Text Encoder** 功能，让 Cosmos-Reason1-7B 和 Cosmos-WAM 分别跑在不同的 GPU 上。

## 显存需求分析

| 组件 | 显存需求 |
|------|---------|
| Cosmos-Reason1-7B (bf16) | ~14 GB |
| Cosmos-WAM (bf16) | ~8-10 GB |
| 总计 | ~22-24 GB |

单卡 24GB 可能不够，建议双卡配置。

## 配置方法

### 方法 1: 修改配置文件

编辑 `configs/sim_robotwin.yaml`:

```yaml
EVALUATION:
  # 原有配置...
  
  # 在线 Text Encoder 设置
  use_online_text_encoder: true
  online_text_encoder_path: /path/to/Cosmos-Reason1-7B
  text_encoder_device: cuda:0  # Text encoder 使用 GPU 0
  device: cuda:1               # 主模型使用 GPU 1
```

### 方法 2: 命令行参数

```bash
python experiments/robotwin/eval_robotwin_single.py \
  ckpt=/path/to/checkpoint.pt \
  EVALUATION.task_name=adjust_bottle \
  EVALUATION.use_online_text_encoder=true \
  EVALUATION.online_text_encoder_path=/path/to/Cosmos-Reason1-7B \
  EVALUATION.text_encoder_device=cuda:0 \
  EVALUATION.device=cuda:1 \
  gpu_id=1
```

### 方法 3: RoboTwin 直接使用 (通过 overrides)

```bash
python script/eval_policy.py \
  --config policy/cosmos_wam_policy/deploy_policy.yml \
  --overrides \
  --use_online_text_encoder true \
  --online_text_encoder_path /path/to/Cosmos-Reason1-7B \
  --text_encoder_device cuda:0 \
  --device cuda:1
```

## 单卡 24GB 使用建议

如果你只有一张 24GB GPU，可以尝试以下配置：

```yaml
# 1. 使用 fp16 减少显存
mixed_precision: fp16

# 2. 减小 action_horizon
action_horizon: 4

# 3. 减少推理步数
num_inference_steps: 10
```

或者使用 cache 文件（推荐）：
```bash
# 预计算所有需要的 text embeddings
python scripts/precompute_text_embeddings.py \
  --dataset_root /path/to/RoboTwin \
  --model_path /path/to/Cosmos-Reason1-7B \
  --output_dir /path/to/text_embeds_cache
```

## 故障排除

### 问题: `ModuleNotFoundError: transformers`
```bash
pip install transformers
```

### 问题: Text encoder 加载失败
检查模型路径是否正确：
```bash
ls /path/to/Cosmos-Reason1-7B/model.safetensors
```

### 问题: CUDA out of memory
- 确保 `text_encoder_device` 和 `device` 是不同的 GPU
- 或者减小 `action_horizon` 和 `num_inference_steps`

## 性能对比

| 模式 | 首次推理延迟 | 后续推理延迟 | 显存需求 |
|------|------------|------------|---------|
| Cache 文件 | 快 | 快 | ~10 GB |
| 在线计算 | 慢 (~2-3s) | 中等 (~0.5s) | ~22-24 GB |

推荐使用 cache 文件进行批量评估，在线计算用于快速测试。
