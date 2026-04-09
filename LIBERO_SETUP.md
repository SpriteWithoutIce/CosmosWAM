# LIBERO 评估设置

## 问题
缺少 text embedding cache 文件。

## 解决方案

### 方案 1: 使用在线 Text Encoder（推荐，已配置）

修改后的代码支持在线计算 text embeddings，无需预计算 cache。

```bash
bash libero.sh
```

默认配置 (`USE_ONLINE_ENCODER=true`)：
- 自动加载 Cosmos-Reason1-7B 模型
- 实时计算 task description 的 embedding
- 需要约 14GB 额外显存

### 方案 2: 预计算 Text Embeddings

如果不想在评估时加载 Reason1-7B，可以预计算所有 task 的 embeddings：

```bash
python scripts/precompute_libero_text_embeddings.py \
    --model_path /mnt/data/linyihan/Cosmos-Reason1-7b \
    --output_dir ./data/text_embeds_cache/libero \
    --task_suite libero_spatial

# 然后修改 libero.sh
USE_ONLINE_ENCODER=false
TEXT_EMBED_CACHE="./data/text_embeds_cache/libero"
```

## 文件修改

1. `experiments/libero/eval_libero_single.py`
   - 添加了 `OnlineTextEncoder` 类
   - 支持 `use_online_text_encoder` 配置

2. `experiments/libero/libero_utils.py`
   - 添加了 LIBERO_PATH 自动检测

3. `libero.sh`
   - 支持在线/离线两种模式切换

## 显存需求

| 模式 | 显存需求 |
|------|---------|
| 在线 Encoder | ~24GB (14GB for Reason1 + 10GB for Cosmos-WAM) |
| 预计算 Cache | ~10GB (仅 Cosmos-WAM) |

## 故障排除

### 错误: `No module named 'libero'`
确保设置了正确的 LIBERO_PATH：
```bash
export LIBERO_PATH=/home/jwhe/linyihan/LIBERO
```

### 错误: `Missing text embedding cache`
使用在线 encoder 模式（默认已启用），或预计算 embeddings。

### 错误: `CUDA out of memory`
减小 batch size 或使用预计算 cache 模式。
