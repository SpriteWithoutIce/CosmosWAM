"""
Precompute text embeddings for LIBERO evaluation tasks.

Reads tasks from dataset meta/tasks.jsonl files and computes embeddings.
Hash is computed ONLY on the task description (no prompt template).

Usage:
    python scripts/precompute_libero_text_embeds.py \
        --dataset_root /home/jwhe/linyihan/datasets/libero_mujoco3.3.2 \
        --model_path /path/to/Cosmos-Reason1-7B \
        --output_dir ./data/text_embeds_cache/libero \
        --context_len 128
"""

import argparse
import hashlib
import json
import os
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, Qwen2_5_VLForConditionalGeneration

NUM_EMBEDDING_PADDING_TOKENS = 512
FULL_CONCAT_DIM = 28 * 3584  # 100352


def parse_args():
    parser = argparse.ArgumentParser(description="Precompute text embeddings for LIBERO")
    parser.add_argument(
        "--dataset_root",
        type=str,
        default="/home/jwhe/linyihan/datasets/libero_mujoco3.3.2",
        help="Root directory containing LIBERO suites (e.g., libero_10_no_noops_lerobot)",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="/home/jwhe/linyihan/CKPT/Cosmos-Reason1-7B",
        help="Path to Cosmos-Reason1-7B model",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data/text_embeds_cache/libero",
        help="Output cache directory",
    )
    parser.add_argument(
        "--context_len",
        type=int,
        default=128,
        help="Context length for embeddings",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for encoding",
    )
    return parser.parse_args()


def collect_tasks_from_dataset(dataset_root: str) -> list[str]:
    """Collect all unique task descriptions from LIBERO dataset meta files.
    
    Reads meta/tasks.jsonl from each suite and extracts the 'task' field.
    """
    dataset_root = Path(dataset_root)
    all_tasks = []
    
    # LIBERO has 4 suites
    suite_names = [
        "libero_spatial_no_noops_lerobot",
        "libero_object_no_noops_lerobot", 
        "libero_goal_no_noops_lerobot",
        "libero_10_no_noops_lerobot",
    ]
    
    for suite_name in suite_names:
        tasks_file = dataset_root / suite_name / "meta" / "tasks.jsonl"
        if not tasks_file.exists():
            print(f"Warning: {tasks_file} not found, skipping {suite_name}")
            continue
        
        with open(tasks_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    task_desc = obj.get("task", "")
                    if task_desc:
                        all_tasks.append(task_desc)
                except json.JSONDecodeError:
                    print(f"Warning: Failed to parse JSON line in {tasks_file}")
                    continue
        
        print(f"  Loaded {suite_name}: {len(all_tasks)} tasks so far")
    
    # Remove duplicates while preserving order
    seen = set()
    unique_tasks = []
    for task in all_tasks:
        if task not in seen:
            seen.add(task)
            unique_tasks.append(task)
    
    return unique_tasks


@torch.no_grad()
def compute_embeddings_batch(model, tokenizer, tasks: list[str], pad_id: int, context_len: int):
    """Compute embeddings for a batch of tasks."""
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
        if len(ids) < context_len:
            ids = ids + [pad_id] * (context_len - len(ids))
        else:
            ids = ids[:context_len]
        input_ids_list.append(torch.LongTensor(ids))

    input_ids_batch = torch.stack(input_ids_list).cuda()
    outputs = model(
        input_ids=input_ids_batch,
        output_hidden_states=True,
        return_dict=True,
    )
    hidden_states = outputs.hidden_states

    # 28 layers mean-normalize and concat -> (B, context_len, 100352)
    normalized = []
    for layer_idx in range(1, len(hidden_states)):
        h = hidden_states[layer_idx].float()
        h = (h - h.mean(dim=-1, keepdim=True)) / (h.std(dim=-1, keepdim=True) + 1e-8)
        normalized.append(h)
    text_emb = torch.cat(normalized, dim=-1)

    embeddings = {}
    for j, task_text in enumerate(tasks):
        embeddings[task_text] = text_emb[j:j+1].to(torch.bfloat16).cpu()
    return embeddings


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

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

    # Collect tasks from dataset files
    print(f"\nCollecting tasks from: {args.dataset_root}")
    all_tasks = collect_tasks_from_dataset(args.dataset_root)
    print(f"Total unique tasks: {len(all_tasks)}")
    
    if len(all_tasks) == 0:
        print("Error: No tasks found!")
        return

    # Compute embeddings
    for i in tqdm(range(0, len(all_tasks), args.batch_size), desc="Encoding"):
        batch = all_tasks[i:i+args.batch_size]
        embeddings = compute_embeddings_batch(model, tokenizer, batch, pad_id, args.context_len)

        # Save each embedding
        # NOTE: Hash is computed ONLY on the task description, NOT on any prompt template
        for task_text, emb in embeddings.items():
            hashed = hashlib.sha256(task_text.encode("utf-8")).hexdigest()
            cache_path = output_dir / f"{hashed}.t5_len{args.context_len}.pt"
            
            emb = emb.squeeze(0)
            mask = torch.ones(args.context_len, dtype=torch.bool)
            
            torch.save({
                "context": emb,
                "mask": mask,
                "text": task_text,
            }, cache_path)
            
            tqdm.write(f"Saved: {task_text[:50]}... -> {cache_path.name}")

    print(f"\nDone! Saved {len(all_tasks)} embeddings to {output_dir}")


if __name__ == "__main__":
    main()
