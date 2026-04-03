# Cosmos-WAM: 基于 Cosmos-Predict2.5 的世界动作模型

使用 NVIDIA Cosmos-Predict2.5 作为基础模型，配合 14 层 Action Head 进行视频-动作联合训练。

## 概述

本项目实现了**世界动作模型 (World Action Model, WAM)**，同时学习两个任务：
1. **生成未来视频帧**（世界模型）- 使用 Cosmos 2B DiT
2. **预测机器人动作** - 使用 14 层 Transformer，关注视频 DiT 的中间层特征

### 主要特性

- **基础模型**: Cosmos-Predict2.5-2B (Reason-1 7B 版本)
- **动作头 (Action Head)**: 14 层 Transformer，带 AdaLN，cross-attention 到 DiT 第 14-27 层
- **数据格式**: LeRobot 格式 (parquet + mp4)
- **文本编码器**: Reason-1 7B (100352 维 embedding)
- **训练方式**: 联合视频 + 动作流匹配训练，使用 DeepSpeed ZeRO-2

---

## 安装

### 1. 克隆并安装

```bash
cd cosmos-wam
pip install -e .
```

### 2. Cosmos 依赖

```bash
export PYTHONPATH=/home/jwhe/linyihan/cosmos-predict2.5:$PYTHONPATH
```

### 3. 安装其他依赖

```bash
pip install torch torchvision accelerate deepspeed hydra-core omegaconf einops transformers
```

---

## 数据准备

### 1. 数据集格式: LeRobot

训练管道期望使用 **LeRobot 格式**。每个数据集的目录结构如下：

```
data/
└── libero_spatial_no_noops_lerobot/
    ├── data/
    │   └── chunk-000/
    │       ├── episode_000000.parquet
    │       ├── episode_000001.parquet
    │       └── ...
    ├── meta/
    │   ├── episodes.jsonl
    │   ├── info.json
    │   ├── stats.json
    │   └── tasks.jsonl          # 任务描述文件
    └── videos/
        └── chunk-000/
            ├── observation.images.cam_high/
            │   ├── episode_000000.mp4
            │   └── ...
            └── observation.images.wrist_image/
                └── ...
```

#### 必需的文件

- **parquet 文件**: 存储 `observation.state`, `action`, `timestamp`, `task_index`
- **videos/**: 每个相机、每个 episode 的 MP4 视频文件
- **meta/tasks.jsonl**: 任务描述（用于生成文本 embedding）
  ```jsonl
  {"task": "pick up the black bowl"}
  {"task": "place the cube on the plate"}
  ```

### 2. 转换数据到 LeRobot 格式

如果你有原始的机器人演示数据，使用 LeRobot 库进行转换：

```bash
# 安装 LeRobot
pip install lerobot

# 使用他们的转换脚本或编写自定义转换器
# 参考: https://github.com/huggingface/lerobot
```

**parquet 文件中的关键字段**：
- `observation.state`: 机器人本体感知（例如 8 维）
- `action`: 机器人动作（例如 7 维：6D 位姿 + 夹爪）
- `timestamp`: Episode 时间戳
- `task_index`: 对应 tasks.jsonl 中的索引

### 3. 预计算文本 Embedding（Reason-1 7B）

**⚠️ 重要**: Cosmos 2B Posttrain checkpoint 使用 **Reason-1 7B** 文本编码器，不是 T5！

文本 embedding 的维度是 **[512, 100352]**。

#### 设置

你需要访问 Reason-1 7B 模型。请查看你的 checkpoint 发布说明获取确切的模型路径。

#### 预计算脚本

创建一个脚本（需要根据你的 checkpoint 进行调整）：

```python
# scripts/precompute_text_embeds.py
import os
import hashlib
import torch
import json
from pathlib import Path

# 从 cosmos_predict2 导入（根据你的 checkpoint 调整）
from cosmos_predict2._src.predict2.text_encoders.text_encoder import TextEncoder

# 初始化 Reason-1 7B 文本编码器
text_encoder = TextEncoder(config, device="cuda")  # 根据 checkpoint 调整 config
text_encoder.eval()

CACHE_DIR = "./data/text_embeds_cache/libero_reason1"
MAX_LEN = 512
os.makedirs(CACHE_DIR, exist_ok=True)

def encode_and_cache(task: str):
    prompt = f"A video recorded from a robot's point of view executing the following instruction: {task}"
    hashed = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    cache_path = os.path.join(CACHE_DIR, f"{hashed}.pt")
    if os.path.exists(cache_path):
        return
    
    with torch.no_grad():
        # 编码（根据 checkpoint API 调整）
        context, mask = text_encoder.encode(prompt, max_length=MAX_LEN)
        context = context.cpu().to(torch.bfloat16)  # [512, 100352]
        mask = mask.cpu().bool()                     # [512]
    
    torch.save({"context": context, "mask": mask}, cache_path)

# 处理所有任务
for dataset_dir in [
    "./data/libero_spatial_no_noops_lerobot",
    "./data/libero_object_no_noops_lerobot",
    "./data/libero_goal_no_noops_lerobot",
    "./data/libero_10_no_noops_lerobot",
]:
    tasks_path = Path(dataset_dir) / "meta" / "tasks.jsonl"
    if not tasks_path.exists():
        continue
    with open(tasks_path) as f:
        for line in f:
            task = json.loads(line)["task"]
            encode_and_cache(task)

print("文本 embedding 预计算完成！")
```

**预期输出**：
```
data/text_embeds_cache/libero_reason1/
├── {hash1}.pt   # 包含 {"context": [512, 100352], "mask": [512]}
├── {hash2}.pt
└── ...
```

---

## Checkpoint 准备

### 1. 下载 Cosmos-Predict2.5-2B-Posttrain

从 NVIDIA 官方发布下载 checkpoint。你应该得到一个 consolidated `.pt` 文件：
```
nvidia/Cosmos-Predict2.5-2B-Posttrain/
└── model.pt   # ~15GB
```

### 2. 提取 DIT 和 VAE 权重

运行提取脚本：

```bash
python scripts/extract_ckpt.py \
    --input /path/to/nvidia/Cosmos-Predict2.5-2B-Posttrain/model.pt \
    --output_dir ./checkpoints
```

**输出**：
```
checkpoints/
├── cosmos_dit.pt   # DiT 权重 (~4GB)
└── cosmos_vae.pt   # VAE 权重 (~1GB)
```

### 3. 验证提取

```python
import torch
dit_state = torch.load("./checkpoints/cosmos_dit.pt", map_location="cpu")
print(f"DIT keys: {len(dit_state)}")
print(f"示例 keys: {list(dit_state.keys())[:5]}")
```

---

## 训练

### 1. 配置训练参数

编辑 `configs/train_cosmos_2b.yaml`：

```yaml
model:
  dit_checkpoint: ./checkpoints/cosmos_dit.pt
  vae_checkpoint: ./checkpoints/cosmos_vae.pt
  
data:
  train:
    dataset_dirs:
      - ./data/libero_spatial_no_noops_lerobot
      # 添加更多数据集...
    text_embedding_cache_dir: ./data/text_embeds_cache/libero_reason1
    
trainer:
  output_dir: ./outputs/cosmos_2b_libero
  batch_size: 2              # 每 GPU（80GB 可以容纳 2-4）
  learning_rate: 2.0e-5      # Video DiT 学习率
  action_learning_rate: 1.0e-4  # Action head 学习率
```

### 2. 启动训练

**单节点，4 张 GPU**：
```bash
torchrun --nproc_per_node=4 scripts/train.py
```

**使用 DeepSpeed ZeRO-2**（已在 yaml 中配置）：
```bash
accelerate launch --config_file accelerate_config.yaml scripts/train.py
```

**注意**：
- 首次运行会计算数据集统计信息用于归一化
- 训练日志保存在 `./outputs/cosmos_2b_libero/`
- 每 5000 步保存 checkpoint

### 3. 监控训练

```bash
# 查看日志
tail -f outputs/cosmos_2b_libero/logs/latest.log

# 检查 GPU 使用率
watch -n 1 nvidia-smi
```

**预期指标**：
- `loss_video`: ~0.5-1.0（速度预测 MSE）
- `loss_action`: ~0.1-0.5（动作 MSE，取决于归一化）
- `loss_total`: 联合损失

---

## 推理

### 仅动作推理

```python
from cosmos_wam.models import CosmosWAM
import torch

# 加载模型
model = CosmosWAM(dit=dit, vae=vae, action_head=action_head)
model.load_state_dict(torch.load("./outputs/cosmos_2b_libero/checkpoints/final.pt")["model"])
model.eval().cuda()

# 准备第一帧 (3, H, W)
first_frame = ...  # torch.Tensor [3, 224, 448]
context = ...      # 预计算的文本 embedding [512, 100352]

# 推理动作
action = model.infer_action(
    first_frame_pixels=first_frame,
    action_horizon=32,
    context=context,
    num_inference_steps=20
)
print(action.shape)  # [1, 32, 7]
```

### 联合视频 + 动作生成

```python
result = model.infer_joint(
    first_frame_pixels=first_frame,
    action_horizon=32,
    context=context,
    num_inference_steps=20
)
video = result["video_pixels"]  # [1, 3, T, H, W]
action = result["action"]       # [1, 32, 7]
```

---

## 项目结构

```
cosmos-wam/
├── cosmos_wam/
│   ├── models/
│   │   ├── cosmos_wam.py         # 主模型包装器（含 hooks）
│   │   ├── action_head.py        # 14 层 ActionDiT
│   │   ├── dit_wrapper.py        # MiniTrainDIT（28 层）
│   │   ├── vae_wrapper.py        # Wan2pt1 VAE
│   │   └── ckpt_loader.py        # Checkpoint 加载
│   ├── datasets/lerobot/         # LeRobot 数据集支持
│   ├── trainer.py                # 训练循环（Accelerate + DeepSpeed）
│   ├── runtime.py                # 训练入口
│   └── utils/
├── configs/
│   └── train_cosmos_2b.yaml      # 训练配置
└── scripts/
    ├── train.py                  # 训练脚本
    └── extract_ckpt.py           # Checkpoint 提取
```

---

## 关键设计决策

### 1. 为什么 Action Head 使用第 14-27 层？

- 第 0-13 层：底层视觉特征（边缘、纹理）
- 第 14-27 层：高层语义特征（物体、运动、 affordance）
- 动作决策需要语义理解，而非底层像素

### 2. 为什么视频条件不做 pooling？

- 保留完整空间结构（例如物体位置）
- Action head 使用空间位置编码来理解布局
- 计算量更大但空间精度更好

### 3. 视频和动作使用独立的 Timestep

- 视频和动作在训练期间可以有不同的噪声水平
- 比共享 timestep 更灵活
- 与 FastWAM 的设计保持一致

---

## 故障排除

### 显存不足

1. 减小 `batch_size`（尝试 1）
2. 启用梯度检查点（默认已开启）
3. 使用 ZeRO-3 替代 ZeRO-2
4. 减小配置中的 `video_size`（例如 [192, 384]）

### 文本 Embedding 维度不匹配

如果看到关于 `crossattn_proj` 维度的错误：
- 检查你的 checkpoint 使用 Reason-1 7B（100352 维）而不是 T5（1024 维）
- 验证 `use_crossattn_projection=True` 和 `crossattn_proj_in_channels=100352`

### 动作损失不收敛

1. 检查动作归一化（应该在 -1 到 1 之间）
2. 增大动作学习率（尝试 2e-4）
3. 验证文本 embedding 是否正确加载

---

## 引用

如果你使用本代码，请引用：
```bibtex
@software{cosmos_wam,
  title={Cosmos-WAM: World Action Model},
  year={2025},
  note={Based on NVIDIA Cosmos-Predict2.5}
}
```

---

## 许可证

Apache 2.0（遵循 Cosmos 许可证条款）
