"""Precompute Reason1 text embeddings for Cosmos-WAM (per sub-dataset).

使用 Qwen2.5-VL-7B (Cosmos-Reason1-7B) 为每个子数据集单独预计算语言 embedding，
输出格式适配 Cosmos-WAM 数据集加载器。

Usage:
    python scripts/precompute_text_embeddings.py \
        --dataset_root /path/to/lerobot_robotwin_eef_clean_50 \
        --model_path /path/to/Cosmos-Reason1-7B \
        --output_dir ./data/text_embeds_cache/libero_reason1 \
        --batch_size 4

Output:
    生成的缓存文件结构:
    output_dir/
    ├── dataset_name_1/
    │   ├── {hash1}.t5_len512.pt   # {"context": [512, 100352], "mask": [512]}
    │   └── {hash2}.t5_len512.pt
    └── dataset_name_2/
        └── ...
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, Qwen2_5_VLForConditionalGeneration

NUM_EMBEDDING_PADDING_TOKENS = 512
FULL_CONCAT_DIM = 28 * 3584  # 100352，与 TextEncoder FULL_CONCAT 一致
CONTEXT_LEN = 512

# Cosmos-WAM 使用的 prompt 模板
DEFAULT_PROMPT = "A video recorded from a robot's point of view executing the following instruction: {task}"


def collect_tasks_per_subdir(dataset_root: str) -> List[Tuple[Path, List[str]]]:
    """对每个子目录收集本目录下 unique task 文本."""
    root = Path(dataset_root)
    result: List[Tuple[Path, List[str]]] = []
    for task_dir in sorted(root.iterdir()):
        if not task_dir.is_dir():
            continue
        tasks_file = task_dir / "meta" / "tasks.jsonl"
        if not tasks_file.exists():
            continue
        tasks = set()
        with open(tasks_file) as f:
            for line in f:
                obj = json.loads(line)
                tasks.add(obj["task"])
        if tasks:
            result.append((task_dir, sorted(tasks)))
    print(f"Found {len(result)} sub-datasets with tasks.jsonl")
    return result


@torch.no_grad()
def compute_embeddings_batch(
    model,
    tokenizer,
    tasks: List[str],
    pad_id: int,
) -> Dict[str, torch.Tensor]:
    """对一批 task 文本计算 (1, 512, 100352) 的 Reason1 FULL_CONCAT embedding."""
    input_ids_list = []
    for task_text in tasks:
        conversations = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "You are a helpful assistant who will provide prompts to an image generator.",
                    }
                ],
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": task_text}],
            },
        ]
        ids = tokenizer.apply_chat_template(
            conversations,
            tokenize=True,
            add_generation_prompt=False,
            return_tensors=None,
        )
        if len(ids) < NUM_EMBEDDING_PADDING_TOKENS:
            ids = ids + [pad_id] * (NUM_EMBEDDING_PADDING_TOKENS - len(ids))
        else:
            ids = ids[:NUM_EMBEDDING_PADDING_TOKENS]
        input_ids_list.append(torch.LongTensor(ids))

    input_ids_batch = torch.stack(input_ids_list).cuda()
    outputs = model(
        input_ids=input_ids_batch,
        output_hidden_states=True,
        return_dict=True,
    )
    hidden_states = outputs.hidden_states  # tuple of (num_layers+1) tensors

    # 28 层 mean-normalize 后在最后一维 concat -> (B, 512, 28*3584 = 100352)
    normalized = []
    for layer_idx in range(1, len(hidden_states)):
        h = hidden_states[layer_idx].float()
        h = (h - h.mean(dim=-1, keepdim=True)) / (h.std(dim=-1, keepdim=True) + 1e-8)
        normalized.append(h)
    text_emb = torch.cat(normalized, dim=-1)  # (B, 512, 100352)

    assert (
        text_emb.shape[-1] == FULL_CONCAT_DIM
    ), f"Unexpected Reason1 dim {text_emb.shape[-1]}, expected {FULL_CONCAT_DIM}"

    embeddings: Dict[str, torch.Tensor] = {}
    for j, task_text in enumerate(tasks):
        embeddings[task_text] = text_emb[j : j + 1].to(torch.bfloat16).cpu()
    return embeddings


def save_cosmos_wam_format(embeddings: Dict[str, torch.Tensor], cache_dir: str):
    """保存为 Cosmos-WAM 需要的格式: {hash}.t5_len512.pt"""
    os.makedirs(cache_dir, exist_ok=True)
    
    for task_text, emb in embeddings.items():
        # 使用 Cosmos-WAM 的 prompt 模板计算 hash
        prompt = DEFAULT_PROMPT.format(task=task_text)
        hashed = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        cache_path = os.path.join(cache_dir, f"{hashed}.t5_len{CONTEXT_LEN}.pt")
        
        # emb shape: (1, 512, 100352) -> squeeze to (512, 100352)
        emb = emb.squeeze(0)
        
        # 创建 mask: 所有位置都有效 (Cosmos-WAM 会在数据集端处理 padding)
        mask = torch.ones(CONTEXT_LEN, dtype=torch.bool)
        
        torch.save({
            "context": emb,      # [512, 100352], bfloat16
            "mask": mask,        # [512], bool
        }, cache_path)


def main():
    parser = argparse.ArgumentParser(
        description="Precompute text embeddings for Cosmos-WAM using Reason1-7B"
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        required=True,
        help="数据集根目录，包含各个子数据集文件夹 (每个子文件夹应有 meta/tasks.jsonl)",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Cosmos-Reason1-7B 模型路径 (如 /path/to/Cosmos-Reason1-7B)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="输出缓存目录 (如 ./data/text_embeds_cache/libero_reason1)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="批处理大小 (默认: 4)",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="若子数据集已有缓存，则跳过",
    )
    args = parser.parse_args()

    subdirs_tasks = collect_tasks_per_subdir(args.dataset_root)
    if not subdirs_tasks:
        print("No sub-datasets with tasks.jsonl found; nothing to do.")
        return

    print(f"Loading model from: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    for task_dir, tasks in tqdm(subdirs_tasks, desc="Sub-datasets"):
        # 每个子数据集单独保存到自己的缓存目录
        subdir_name = task_dir.name
        cache_dir = os.path.join(args.output_dir, subdir_name)
        
        # 检查是否已存在（简单检查：如果有任何 .pt 文件就认为已处理）
        if args.skip_existing and os.path.exists(cache_dir):
            existing = list(Path(cache_dir).glob("*.t5_len*.pt"))
            if len(existing) >= len(tasks):
                print(f"\nSkipping {subdir_name} (already cached)")
                continue

        print(f"\nProcessing {task_dir} ({len(tasks)} tasks) -> {cache_dir}")

        embeddings: Dict[str, torch.Tensor] = {}
        for start in range(0, len(tasks), args.batch_size):
            batch = tasks[start : start + args.batch_size]
            batch_emb = compute_embeddings_batch(model, tokenizer, batch, pad_id)
            embeddings.update(batch_emb)
            
            # 适当清理显存
            if (start // args.batch_size) % 10 == 0 and start > 0:
                torch.cuda.empty_cache()

        # 保存为 Cosmos-WAM 格式
        save_cosmos_wam_format(embeddings, cache_dir)

        sample_emb = next(iter(embeddings.values()))
        print(f"  Saved {len(embeddings)} embeddings, shape {tuple(sample_emb.squeeze(0).shape)}")

    print(f"\nDone! Text embeddings saved to: {args.output_dir}")
    print(f"Usage in config:")
    print(f"  data:")
    print(f"    train:")
    print(f"      text_embedding_cache_dir: {args.output_dir}/<dataset_name>")


if __name__ == "__main__":
    main()
