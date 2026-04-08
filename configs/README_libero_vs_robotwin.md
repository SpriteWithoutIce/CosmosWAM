# LIBERO vs RoboTwin 配置对比

## 数据集差异

| 参数 | LIBERO | RoboTwin |
|------|--------|----------|
| **相机数量** | 2 (agentview + wrist) | 1 (head_camera) |
| **图像尺寸** | 224x224 (每相机), 拼接后 224x448 | 240x320 |
| **动作维度** | 7 (eef 6 + gripper 1) | 16 (双臂各 8) |
| **Proprio 维度** | 8 (pos 3 + axisangle 3 + gripper 2) | 16 |
| **任务数** | 4 suites (spatial/object/goal/10) | 10 tasks |
| **帧率** | 20 FPS | 50 FPS |
| **Action 频率** | 5 Hz (action_video_freq_ratio=4) | 12.5 Hz (action_video_freq_ratio=4) |

## 模型配置差异

| 参数 | LIBERO | RoboTwin |
|------|--------|----------|
| **DiT max_img_h** | 224 | 240 |
| **DiT max_img_w** | 448 | 320 |
| **Action Head action_dim** | 7 | 16 |
| **Context Length** | 128 | 512 |
| **Batch Size** | 16 | 16 |
| **Learning Rate** | 1e-4 | 2e-5 (video), 1e-4 (action) |
| **Weight Decay** | 1e-2 | 0.0 |

## 关键配置项

### LIBERO 特有
- `concat_multi_camera: "horizontal"` - 水平拼接 2 个相机
- `num_output_cameras: 2`
- `delta_action_dim_mask: [true, true, true, true, true, true, false]` - eef 是 delta，gripper 不是

### RoboTwin 特有
- `concat_multi_camera: null` 或 `"horizontal"` - 单相机无需拼接
- `num_output_cameras: 1`
- `delta_action_dim_mask: [true, true, true, true, true, true, true, false, ...]` - 双臂各 7 个 delta + 1 gripper

## 训练步骤

### 1. 预计算 Text Embeddings

```bash
# LIBERO
python scripts/precompute_libero_text_embeds.py \
    --model_path /path/to/Cosmos-Reason1-7B \
    --output_dir ./data/text_embeds_cache/libero \
    --context_len 128

# RoboTwin
python scripts/precompute_text_embeddings.py \
    --dataset_root /path/to/robotwin \
    --model_path /path/to/Cosmos-Reason1-7B \
    --output_dir ./data/text_embeds_cache/robotwin_reason1
```

### 2. 训练

```bash
# LIBERO (8 GPUs)
bash scripts/train_libero.sh 8

# RoboTwin (4 GPUs)
bash scripts/train_robotwin.sh 4
```

### 3. 评估

```bash
# LIBERO
python experiments/libero/eval_libero_single.py \
    ckpt=./outputs/cosmos_2b_libero/checkpoints/step_XXXXX.pt \
    EVALUATION.task_suite_name=libero_spatial \
    EVALUATION.task_id=0

# RoboTwin
python experiments/robotwin/eval_robotwin_single.py \
    ckpt=./outputs/cosmos_2b_robotwin/checkpoints/step_XXXXX.pt \
    EVALUATION.task_name=adjust_bottle
```

## 注意事项

1. **图像预处理**: LIBERO 训练数据预处理时会将图像旋转 180 度 (`[::-1, ::-1]`)，评估时也需要同样处理

2. **Gripper 动作**: 
   - 数据加载器将 gripper 映射为 0=close, 1=open
   - 环境需要 -1=open, +1=close
   - 评估时通过 `*2-1` 和 `invert_gripper_action` 转换

3. **Video Size**: 
   - LIBERO: `[224, 448]` (高度 x 宽度，2相机水平拼接)
   - RoboTwin: `[240, 320]` (高度 x 宽度，单相机)
