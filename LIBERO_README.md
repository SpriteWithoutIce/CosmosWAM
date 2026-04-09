# LIBERO 评估 - 快速参考

## 检查结论

✅ **timestep 方向已修复** - `cosmos_wam/models/cosmos_wam.py` 中的 `infer_action` 函数使用正确的 timestep 方向：
- `t = i / num_inference_steps` (从 0 到 1)
- 与训练时保持一致

## 启动测试

### 单任务测试
```bash
cd /home/handoff/Desktop/CosmosWAM
bash libero.sh
```

### 批量测试（10个任务）
```bash
bash libero_batch.sh
```

### 自定义参数
```bash
python experiments/libero/eval_libero_single.py \
    ckpt=/mnt/data/linyihan/ckpt/step_0014000_bf16.pt \
    EVALUATION.task_suite_name=libero_spatial \
    EVALUATION.task_id=0 \
    EVALUATION.num_trials=50 \
    EVALUATION.dataset_stats_path=./dataset_stats.json \
    EVALUATION.text_embedding_cache_dir=./data/text_embeds_cache/libero \
    mixed_precision=bf16
```

## 关键配置

| 参数 | LIBERO 值 | RoboTwin 值 |
|------|----------|------------|
| 动作维度 | 7 | 16 |
| 图像尺寸 | 224x224 | 240x320 |
| 相机数 | 1 | 1-2 |
| 推理步数 | 4 | 20 |
| 重规划步数 | 5 | 4-8 |

## 文件结构

```
experiments/libero/
├── eval_libero_single.py    # 主要评估脚本
├── libero_utils.py          # LIBERO 工具函数
├── libero.sh               # 单任务启动脚本
└── libero_batch.sh         # 批量任务启动脚本
```

## 输出位置

```
evaluate_results/libero/{task_suite}/
├── gpu0_task{N}_results.json    # 结果统计
└── videos/                       # 视频记录
```

## 下一步

1. 检查 text embedding cache 是否存在
2. 检查 dataset_stats.json 是否存在
3. 运行 `bash libero.sh` 测试单个任务
4. 运行 `bash libero_batch.sh` 测试全部任务
