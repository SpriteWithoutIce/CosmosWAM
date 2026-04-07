"""
Precompute text embeddings for RoboTwin evaluation tasks.

Usage:
    python scripts/precompute_robotwin_text_embeds.py \
        --text-encoder /path/to/cosmos-reason1-7b \
        --output-dir /path/to/output \
        --context-len 512

This will generate text embeddings for all 10 RoboTwin tasks with both
"seen" and "unseen" instructions.
"""

import argparse
import hashlib
import os
from pathlib import Path

import torch
from tqdm import tqdm

# RoboTwin task definitions
ROBOTWIN_TASKS = {
    "adjust_bottle": {
        "seen": "The task is adjust_bottle.",
        "unseen": "The task is to adjust the bottle.",
    },
    "beat_block_hammer": {
        "seen": "The task is beat_block_hammer.",
        "unseen": "The task is to beat the block with a hammer.",
    },
    "blocks_ranking_rgb": {
        "seen": "The task is blocks_ranking_rgb.",
        "unseen": "The task is to rank blocks by their color.",
    },
    "blocks_ranking_size": {
        "seen": "The task is blocks_ranking_size.",
        "unseen": "The task is to rank blocks by their size.",
    },
    "click_alarmclock": {
        "seen": "The task is click_alarmclock.",
        "unseen": "The task is to click the alarm clock.",
    },
    "click_bell": {
        "seen": "The task is click_bell.",
        "unseen": "The task is to click the bell.",
    },
    "dump_bin_bigbin": {
        "seen": "The task is dump_bin_bigbin.",
        "unseen": "The task is to dump the bin into the big bin.",
    },
    "grab_roller": {
        "seen": "The task is grab_roller.",
        "unseen": "The task is to grab the roller.",
    },
    "handover_block": {
        "seen": "The task is handover_block.",
        "unseen": "The task is to handover the block.",
    },
    "handover_mic": {
        "seen": "The task is handover_mic.",
        "unseen": "The task is to handover the microphone.",
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description="Precompute text embeddings for RoboTwin")
    parser.add_argument(
        "--text-encoder",
        type=str,
        default="/home/jwhe/linyihan/CKPT/Cosmos-Reason1-7B",
        help="Path to text encoder (Cosmos-Reason1-7B or similar T5 model)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./text_embeds_robotwin",
        help="Output directory for embeddings",
    )
    parser.add_argument(
        "--context-len",
        type=int,
        default=512,
        help="Context length for embeddings",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use for encoding",
    )
    return parser.parse_args()


def load_text_encoder(model_path: str, device: str):
    """Load T5 text encoder."""
    from transformers import AutoModel, AutoTokenizer
    
    print(f"Loading text encoder from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path)
    model = model.to(device).eval()
    
    return model, tokenizer


def encode_text(model, tokenizer, text: str, context_len: int, device: str):
    """Encode text to embedding."""
    with torch.no_grad():
        tokens = tokenizer(
            text,
            return_tensors="pt",
            max_length=context_len,
            truncation=True,
            padding="max_length",
        )
        input_ids = tokens["input_ids"].to(device)
        attention_mask = tokens["attention_mask"].to(device)
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        # Use last hidden state mean pooling
        hidden_states = outputs.last_hidden_state
        
        # Apply attention mask
        mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
        sum_embeddings = torch.sum(hidden_states * mask_expanded, dim=1)
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        embeddings = sum_embeddings / sum_mask
        
    return embeddings.cpu(), attention_mask.cpu()


def save_embedding(text: str, embedding: torch.Tensor, mask: torch.Tensor, output_dir: Path, context_len: int):
    """Save embedding to cache file."""
    hashed = hashlib.sha256(text.encode("utf-8")).hexdigest()
    filename = f"{hashed}.t5_len{context_len}.pt"
    filepath = output_dir / filename
    
    torch.save({
        "context": embedding,
        "mask": mask,
        "text": text,
    }, filepath)
    
    return filepath


def main():
    args = parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    model, tokenizer = load_text_encoder(args.text_encoder, args.device)
    
    print(f"Generating embeddings for {len(ROBOTWIN_TASKS)} tasks...")
    print(f"Output directory: {output_dir}")
    print(f"Context length: {args.context_len}")
    
    # Generate embeddings for all tasks
    all_prompts = []
    for task_name, instructions in ROBOTWIN_TASKS.items():
        for instruction_type, text in instructions.items():
            formatted_text = f"The task is {text}."
            all_prompts.append({
                "task": task_name,
                "type": instruction_type,
                "text": formatted_text,
            })
    
    print(f"Total prompts to encode: {len(all_prompts)}")
    
    for item in tqdm(all_prompts, desc="Encoding"):
        # Check if already exists
        hashed = hashlib.sha256(item["text"].encode("utf-8")).hexdigest()
        filename = f"{hashed}.t5_len{args.context_len}.pt"
        filepath = output_dir / filename
        
        if filepath.exists():
            tqdm.write(f"Skipping {item['task']}/{item['type']} (already exists)")
            continue
        
        # Encode
        embedding, mask = encode_text(
            model, tokenizer, item["text"], args.context_len, args.device
        )
        
        # Save
        save_embedding(item["text"], embedding, mask, output_dir, args.context_len)
        tqdm.write(f"Saved {item['task']}/{item['type']}: {filepath.name}")
    
    print(f"\nDone! Embeddings saved to {output_dir}")
    print(f"\nTo use these embeddings, set:")
    print(f"  EVALUATION.text_embedding_cache_dir={output_dir}")


if __name__ == "__main__":
    main()
