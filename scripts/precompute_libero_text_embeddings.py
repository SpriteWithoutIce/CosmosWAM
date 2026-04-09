"""Precompute text embeddings for LIBERO tasks.

Usage:
    python scripts/precompute_libero_text_embeddings.py \
        --model_path /path/to/Cosmos-Reason1-7B \
        --output_dir ./data/text_embeds_cache/libero \
        --task_suite libero_spatial
"""

import argparse
import hashlib
import os
from pathlib import Path

import sys
LIBERO_PATH = os.environ.get("LIBERO_PATH", "/home/jwhe/linyihan/LIBERO")
if LIBERO_PATH not in sys.path:
    sys.path.insert(0, LIBERO_PATH)

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, Qwen2_5_VLForConditionalGeneration

from libero.libero import benchmark

NUM_EMBEDDING_PADDING_TOKENS = 128  # LIBERO uses 128
FULL_CONCAT_DIM = 28 * 3584  # 100352
CONTEXT_LEN = 128


def get_task_descriptions(task_suite_name: str):
    """Get all task descriptions from a LIBERO task suite."""
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[task_suite_name]()
    
    descriptions = []
    for i in range(len(task_suite)):
        task = task_suite.get_task(i)
        descriptions.append(task.language)
    
    return descriptions


@torch.no_grad()
def compute_embedding(model, tokenizer, task_text: str, device: str = "cuda"):
    """Compute text embedding for a single task."""
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    
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
    
    input_ids = tokenizer.apply_chat_template(
        conversations,
        tokenize=True,
        add_generation_prompt=False,
        return_tensors=None,
    )
    
    # Pad or truncate
    if len(input_ids) < NUM_EMBEDDING_PADDING_TOKENS:
        input_ids = input_ids + [pad_id] * (NUM_EMBEDDING_PADDING_TOKENS - len(input_ids))
    else:
        input_ids = input_ids[:NUM_EMBEDDING_PADDING_TOKENS]
    
    input_ids_tensor = torch.LongTensor(input_ids).unsqueeze(0).to(device)
    
    outputs = model(
        input_ids=input_ids_tensor,
        output_hidden_states=True,
        return_dict=True,
    )
    hidden_states = outputs.hidden_states
    
    # 28 layers mean-normalize and concat
    normalized = []
    for layer_idx in range(1, len(hidden_states)):
        h = hidden_states[layer_idx].float()
        h = (h - h.mean(dim=-1, keepdim=True)) / (h.std(dim=-1, keepdim=True) + 1e-8)
        normalized.append(h)
    
    text_emb = torch.cat(normalized, dim=-1)  # (1, 128, 100352)
    text_emb = text_emb.squeeze(0).to(torch.bfloat16)  # (128, 100352)
    
    return text_emb.cpu()


def save_embedding(task_text: str, embedding: torch.Tensor, output_dir: Path):
    """Save embedding to cache."""
    hashed = hashlib.sha256(task_text.encode("utf-8")).hexdigest()
    cache_path = output_dir / f"{hashed}.t5_len{CONTEXT_LEN}.pt"
    
    mask = torch.ones(CONTEXT_LEN, dtype=torch.bool)
    
    torch.save({
        "context": embedding,
        "mask": mask,
    }, cache_path)
    
    return cache_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to Cosmos-Reason1-7B model")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for embeddings")
    parser.add_argument("--task_suite", type=str, default="libero_spatial",
                        choices=["libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90"],
                        help="LIBERO task suite")
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading model from: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map=args.device,
        trust_remote_code=True,
    )
    model.eval()
    
    print(f"Getting task descriptions from: {args.task_suite}")
    descriptions = get_task_descriptions(args.task_suite)
    print(f"Found {len(descriptions)} tasks")
    
    print(f"Computing embeddings...")
    for i, desc in enumerate(tqdm(descriptions, desc="Tasks")):
        # Check if already exists
        hashed = hashlib.sha256(desc.encode("utf-8")).hexdigest()
        cache_path = output_dir / f"{hashed}.t5_len{CONTEXT_LEN}.pt"
        
        if cache_path.exists():
            print(f"Skipping task {i}: already cached")
            continue
        
        embedding = compute_embedding(model, tokenizer, desc, args.device)
        save_embedding(desc, embedding, output_dir)
        print(f"Task {i}: {desc[:50]}... -> {cache_path.name}")
    
    print(f"\nDone! Embeddings saved to: {output_dir}")


if __name__ == "__main__":
    main()
