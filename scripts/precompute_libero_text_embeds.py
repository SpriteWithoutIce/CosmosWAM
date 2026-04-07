"""
Precompute text embeddings for LIBERO evaluation tasks.

Usage:
    python scripts/precompute_libero_text_embeds.py \
        --model_path /path/to/Cosmos-Reason1-7B \
        --output_dir ./data/text_embeds_cache/libero \
        --context_len 128
"""

import argparse
import hashlib
import os
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, Qwen2_5_VLForConditionalGeneration
from libero.libero import benchmark

NUM_EMBEDDING_PADDING_TOKENS = 512
FULL_CONCAT_DIM = 28 * 3584  # 100352
CONTEXT_LEN = 128

# Cosmos-WAM prompt template
DEFAULT_PROMPT = "A video recorded from a robot's point of view executing the following instruction: {task}"


def parse_args():
    parser = argparse.ArgumentParser(description="Precompute text embeddings for LIBERO")
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

    # Collect all tasks from LIBERO benchmarks
    benchmark_dict = benchmark.get_benchmark_dict()
    all_tasks = []
    
    for suite_name in ["libero_spatial", "libero_object", "libero_goal", "libero_10"]:
        task_suite = benchmark_dict[suite_name]()
        for task_id in range(len(task_suite)):
            task = task_suite.get_task(task_id)
            all_tasks.append(task.language)

    # Remove duplicates
    all_tasks = sorted(set(all_tasks))
    print(f"Total unique tasks: {len(all_tasks)}")

    # Compute embeddings
    for i in tqdm(range(0, len(all_tasks), args.batch_size), desc="Encoding"):
        batch = all_tasks[i:i+args.batch_size]
        embeddings = compute_embeddings_batch(model, tokenizer, batch, pad_id, args.context_len)

        # Save each embedding
        for task_text, emb in embeddings.items():
            prompt = DEFAULT_PROMPT.format(task=task_text)
            hashed = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
            cache_path = output_dir / f"{hashed}.t5_len{args.context_len}.pt"
            
            emb = emb.squeeze(0)
            mask = torch.ones(args.context_len, dtype=torch.bool)
            
            torch.save({
                "context": emb,
                "mask": mask,
                "text": task_text,
            }, cache_path)

    print(f"\nDone! Saved {len(all_tasks)} embeddings to {output_dir}")


if __name__ == "__main__":
    main()
