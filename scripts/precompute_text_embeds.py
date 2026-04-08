"""
Precompute text embeddings for Cosmos-WAM training/evaluation.

Supports both LIBERO and RoboTwin datasets by reading meta/tasks.jsonl from dataset_dirs.
Supports multiple datasets (like FastWAM's precompute_text_embeds.py).

Usage:
    # Single dataset
    python scripts/precompute_text_embeds.py \
        --dataset_dir /path/to/dataset \
        --output_dir ./data/text_embeds_cache/my_dataset \
        --model_path /path/to/Cosmos-Reason1-7B \
        --context_len 128

    # Multiple datasets (like FastWAM)
    python scripts/precompute_text_embeds.py \
        --dataset_dir /path/to/dataset1 /path/to/dataset2 /path/to/dataset3 \
        --output_dir ./data/text_embeds_cache/my_dataset \
        --model_path /path/to/Cosmos-Reason1-7B \
        --context_len 128

    # Multi-GPU
    torchrun --standalone --nproc_per_node=4 scripts/precompute_text_embeds.py \
        --dataset_dir /path/to/dataset1 /path/to/dataset2 \
        --output_dir ./data/text_embeds_cache/my_dataset \
        --model_path /path/to/Cosmos-Reason1-7B \
        --context_len 128

Config options:
    --use_prompt_template: If True, hash = SHA256(DEFAULT_PROMPT.format(task=task))
                          If False, hash = SHA256(task) directly
    --default_prompt: The prompt template to use (default: Cosmos-WAM template)
"""

import argparse
import hashlib
import json
import logging
import os
import sys
from pathlib import Path
from typing import List, Optional

import torch
import torch.distributed as dist
from tqdm import tqdm
from transformers import AutoTokenizer, Qwen2_5_VLForConditionalGeneration

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Default Cosmos-WAM prompt template
DEFAULT_PROMPT = "A video recorded from a robot's point of view executing the following instruction: {task}"

# Full concat dimension for Reason1-7B: 28 layers * 3584 dim = 100352
FULL_CONCAT_DIM = 28 * 3584


def parse_args():
    parser = argparse.ArgumentParser(
        description="Precompute text embeddings using Cosmos-Reason1-7B"
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        nargs="+",
        required=True,
        help="Dataset directory(s) containing meta/tasks.jsonl. Can specify multiple.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for cached embeddings",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="/home/jwhe/linyihan/CKPT/Cosmos-Reason1-7B",
        help="Path to Cosmos-Reason1-7B model",
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
        default=8,
        help="Batch size for encoding",
    )
    parser.add_argument(
        "--use_prompt_template",
        type=lambda x: x.lower() in ('true', '1', 'yes'),
        default=True,
        help="If True, use DEFAULT_PROMPT.format(task=task) for hash. If False, hash task directly.",
    )
    parser.add_argument(
        "--default_prompt",
        type=str,
        default=DEFAULT_PROMPT,
        help="Default prompt template (use {task} as placeholder)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing cache files",
    )
    return parser.parse_args()


def init_distributed():
    """Initialize distributed training if available."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return False, 0, 1

    if not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend, init_method="env://")

    return True, dist.get_rank(), dist.get_world_size()


def read_tasks_from_jsonl(dataset_dirs: List[str]) -> List[str]:
    """Read unique task descriptions from multiple dataset meta/tasks.jsonl files.
    
    Expected format (LIBERO & RoboTwin):
        {"task_index": 0, "task": "turn on the stove..."}
        {"task_index": 1, "task": "put the pot on the stove..."}
    """
    all_tasks = []
    seen = set()
    
    for dataset_dir in dataset_dirs:
        dataset_path = Path(dataset_dir)
        tasks_file = dataset_path / "meta" / "tasks.jsonl"
        
        if not tasks_file.exists():
            # Try alternative path structure (some datasets have different layouts)
            alt_tasks_file = dataset_path / "tasks.jsonl"
            if alt_tasks_file.exists():
                tasks_file = alt_tasks_file
            else:
                logger.warning(f"tasks.jsonl not found in {dataset_path}, skipping")
                continue

        with tasks_file.open("r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    task_desc = record.get("task", "")
                    if task_desc and task_desc not in seen:
                        seen.add(task_desc)
                        all_tasks.append(task_desc)
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse JSON at line {line_idx} in {tasks_file}: {e}")
                    continue

        logger.info(f"Loaded from {dataset_path}: {len([t for t in all_tasks if t in seen])} unique tasks so far")
    
    logger.info(f"Total unique tasks from {len(dataset_dirs)} dataset(s): {len(all_tasks)}")
    return all_tasks


def format_prompt(task: str, use_template: bool, template: str) -> str:
    """Format prompt for embedding computation.
    
    If use_template is True, returns template.format(task=task).
    If False, returns task directly.
    """
    if use_template:
        return template.format(task=task)
    return task


def get_hash_key(task: str, use_template: bool, template: str) -> str:
    """Get the key used for hash computation.
    
    This determines what gets hashed for the cache filename.
    """
    if use_template:
        return template.format(task=task)
    return task


@torch.no_grad()
def encode_batch(
    model,
    tokenizer,
    texts: List[str],
    context_len: int,
    pad_id: int,
) -> torch.Tensor:
    """Encode a batch of texts to embeddings.
    
    Returns: [B, context_len, FULL_CONCAT_DIM] tensor
    """
    # Tokenize
    input_ids_list = []
    for text in texts:
        ids = tokenizer.encode(text, add_special_tokens=True)
        # Pad or truncate to context_len
        if len(ids) < context_len:
            ids = ids + [pad_id] * (context_len - len(ids))
        else:
            ids = ids[:context_len]
        input_ids_list.append(torch.LongTensor(ids))

    input_ids_batch = torch.stack(input_ids_list).to(model.device)
    
    # Get hidden states
    outputs = model(
        input_ids=input_ids_batch,
        output_hidden_states=True,
        return_dict=True,
    )
    hidden_states = outputs.hidden_states  # tuple of (num_layers+1) tensors
    
    # Concatenate all 28 layers (layers 1-28)
    normalized = []
    for layer_idx in range(1, len(hidden_states)):
        h = hidden_states[layer_idx].float()
        # Mean normalize
        h = (h - h.mean(dim=-1, keepdim=True)) / (h.std(dim=-1, keepdim=True) + 1e-8)
        normalized.append(h)
    
    # Concatenate along last dimension: [B, context_len, 28*3584=100352]
    text_emb = torch.cat(normalized, dim=-1)
    return text_emb.to(torch.bfloat16)


def save_embedding(
    task: str,
    embedding: torch.Tensor,
    mask: torch.Tensor,
    cache_dir: Path,
    context_len: int,
    use_template: bool,
    template: str,
):
    """Save embedding to cache file.
    
    Filename: {hash}.t5_len{context_len}.pt
    Where hash = SHA256(get_hash_key(task, use_template, template))
    """
    hash_key = get_hash_key(task, use_template, template)
    hashed = hashlib.sha256(hash_key.encode("utf-8")).hexdigest()
    cache_path = cache_dir / f"{hashed}.t5_len{context_len}.pt"
    
    # Squeeze batch dimension
    emb = embedding.squeeze(0) if embedding.ndim == 3 else embedding
    
    torch.save({
        "context": emb,
        "mask": mask,
        "text": task,
        "prompt": format_prompt(task, use_template, template),
    }, cache_path)
    
    return cache_path


def main():
    args = parse_args()
    
    # Initialize distributed
    is_distributed, rank, world_size = init_distributed()
    
    if is_distributed and rank == 0:
        logger.info(f"Distributed mode: {world_size} GPUs")
    elif not is_distributed:
        logger.info("Single GPU mode")
    
    # Setup paths
    output_dir = Path(args.output_dir)
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
    if is_distributed:
        dist.barrier()
    
    # Load model
    if rank == 0:
        logger.info(f"Loading model from: {args.model_path}")
    
    device = f"cuda:{rank}" if torch.cuda.is_available() else "cpu"
    
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
    )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map={"": device},
        trust_remote_code=True,
    )
    model.eval()
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    
    # Read tasks
    if rank == 0:
        logger.info(f"Reading tasks from {len(args.dataset_dir)} dataset(s):")
        for d in args.dataset_dir:
            logger.info(f"  - {d}")
    
    all_tasks = read_tasks_from_jsonl(args.dataset_dir)
    
    if len(all_tasks) == 0:
        logger.error("No tasks found!")
        return
    
    # Distribute tasks across GPUs
    if is_distributed:
        all_tasks = all_tasks[rank::world_size]
    
    logger.info(f"Rank {rank}: Processing {len(all_tasks)} tasks")
    
    # Process batches
    stats = {"new": 0, "overwrite": 0, "skip": 0}
    
    with tqdm(
        total=len(all_tasks),
        desc=f"Encoding (rank {rank})" if is_distributed else "Encoding",
        disable=is_distributed and rank != 0,
    ) as pbar:
        for i in range(0, len(all_tasks), args.batch_size):
            batch_tasks = all_tasks[i:i + args.batch_size]
            
            # Format texts for encoding (always use template for actual encoding)
            batch_texts = [
                format_prompt(task, args.use_prompt_template, args.default_prompt)
                for task in batch_tasks
            ]
            
            # Encode
            embeddings = encode_batch(
                model, tokenizer, batch_texts, args.context_len, pad_id
            )
            
            # Create mask (all valid since we pad/truncate)
            mask = torch.ones(args.context_len, dtype=torch.bool)
            
            # Save each embedding
            for j, task in enumerate(batch_tasks):
                emb = embeddings[j]
                
                # Check if exists
                hash_key = get_hash_key(task, args.use_prompt_template, args.default_prompt)
                hashed = hashlib.sha256(hash_key.encode("utf-8")).hexdigest()
                cache_path = output_dir / f"{hashed}.t5_len{args.context_len}.pt"
                
                if cache_path.exists() and not args.overwrite:
                    stats["skip"] += 1
                    continue
                
                if cache_path.exists():
                    stats["overwrite"] += 1
                else:
                    stats["new"] += 1
                
                save_embedding(
                    task=task,
                    embedding=emb,
                    mask=mask,
                    cache_dir=output_dir,
                    context_len=args.context_len,
                    use_template=args.use_prompt_template,
                    template=args.default_prompt,
                )
            
            pbar.update(len(batch_tasks))
    
    # Aggregate stats
    if is_distributed:
        stats_tensor = torch.tensor(
            [stats["new"], stats["overwrite"], stats["skip"]],
            device=device,
            dtype=torch.long,
        )
        dist.all_reduce(stats_tensor, op=dist.ReduceOp.SUM)
        stats["new"] = int(stats_tensor[0].item())
        stats["overwrite"] = int(stats_tensor[1].item())
        stats["skip"] = int(stats_tensor[2].item())
    
    if rank == 0:
        logger.info("Finished!")
        logger.info(f"  New: {stats['new']}")
        logger.info(f"  Overwritten: {stats['overwrite']}")
        logger.info(f"  Skipped: {stats['skip']}")
        logger.info(f"  Total: {stats['new'] + stats['overwrite'] + stats['skip']}")
        logger.info(f"Output directory: {output_dir}")
    
    if is_distributed and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
