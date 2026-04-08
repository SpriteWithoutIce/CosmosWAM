"""
Cosmos-WAM RoboTwin Multi-GPU Evaluation with WandB logging.

This script runs evaluation on multiple tasks in parallel using multiple GPU pairs.
Each worker uses 2 GPUs: one for Cosmos-Reason1-7B text encoder, one for main model.

Usage:
    python experiments/robotwin/eval_robotwin_wandb_multigpu.py \
        --ckpt /mnt/data/linyihan/ckpt/step_0014000_bf16.pt \
        --tasks adjust_bottle click_bell ... \
        --num_episodes 50 \
        --num_workers 4 \
        --wandb_project cosmos-wam-robotwin

With 8 GPUs and --num_workers 4, each worker gets 2 GPUs.
"""

import argparse
import json
import multiprocessing as mp
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import wandb


def parse_args():
    parser = argparse.ArgumentParser(description="Cosmos-WAM RoboTwin Multi-GPU Evaluation with WandB")
    
    # Config
    parser.add_argument("--config", type=str, default="configs/sim_robotwin.yaml",
                        help="Path to Hydra config file")
    parser.add_argument("--ckpt", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--tasks", type=str, nargs="+", required=True,
                        help="List of task names to evaluate")
    parser.add_argument("--num_episodes", type=int, default=50,
                        help="Number of episodes per task")
    
    # Multi-GPU settings
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of parallel workers (each uses 2 GPUs)")
    parser.add_argument("--gpu_start_id", type=int, default=0,
                        help="Starting GPU ID (0, 2, 4, 6 for 8-GPU system)")
    
    # WandB settings
    parser.add_argument("--wandb_project", type=str, default="cosmos-wam-robotwin",
                        help="WandB project name")
    parser.add_argument("--wandb_entity", type=str, default=None,
                        help="WandB entity/team name")
    parser.add_argument("--wandb_run_name", type=str, default=None,
                        help="WandB run name (auto-generated if not provided)")
    parser.add_argument("--wandb_tags", type=str, nargs="+", default=[],
                        help="WandB tags")
    
    # Evaluation settings
    parser.add_argument("--robotwin_root", type=str, default="/root/linyihan/RoboTwin",
                        help="RoboTwin root directory")
    parser.add_argument("--dataset_stats_path", type=str, default="./dataset_stats.json",
                        help="Path to dataset stats JSON")
    parser.add_argument("--num_inference_steps", type=int, default=20,
                        help="Number of diffusion inference steps")
    parser.add_argument("--replan_steps", type=int, default=8,
                        help="Number of replan steps")
    parser.add_argument("--instruction_type", type=str, default="seen",
                        choices=["seen", "unseen"],
                        help="Instruction type")
    parser.add_argument("--mixed_precision", type=str, default="bf16",
                        choices=["no", "fp16", "bf16"],
                        help="Mixed precision mode")
    
    # Online text encoder settings
    parser.add_argument("--online_text_encoder_path", type=str, required=True,
                        help="Path to Cosmos-Reason1-7B model")
    
    return parser.parse_args()


def extract_success_rate(line: str) -> Optional[Tuple[int, int, float]]:
    """Extract success count, total count, and rate from log line."""
    pattern = r"Success rate:\s*(\d+)/(\d+)\s*=>\s*([\d.]+)%"
    match = re.search(pattern, line)
    if match:
        suc = int(match.group(1))
        test_num = int(match.group(2))
        rate = float(match.group(3))
        return suc, test_num, rate
    return None


def worker_process(
    worker_id: int,
    text_encoder_gpu: int,
    model_gpu: int,
    task_queue: mp.Queue,
    result_queue: mp.Queue,
    args,
    wandb_queue: mp.Queue,
):
    """Worker process that runs evaluation on assigned GPU pair."""
    
    print(f"[Worker {worker_id}] Starting with GPUs: text_encoder={text_encoder_gpu}, model={model_gpu}")
    
    while True:
        try:
            task_idx, task_name = task_queue.get(timeout=1)
        except:
            break
        
        if task_name is None:  # Poison pill
            break
        
        print(f"[Worker {worker_id}] Starting task {task_idx}: {task_name}")
        
        # Build Hydra overrides
        overrides = [
            f"ckpt={args.ckpt}",
            f"EVALUATION.task_name={task_name}",
            f"EVALUATION.eval_num_episodes={args.num_episodes}",
            f"EVALUATION.robotwin_root={args.robotwin_root}",
            f"EVALUATION.dataset_stats_path={args.dataset_stats_path}",
            f"EVALUATION.num_inference_steps={args.num_inference_steps}",
            f"EVALUATION.replan_steps={args.replan_steps}",
            f"EVALUATION.instruction_type={args.instruction_type}",
            f"mixed_precision={args.mixed_precision}",
            f"gpu_id={model_gpu}",  # This controls CUDA_VISIBLE_DEVICES
            "EVALUATION.use_online_text_encoder=true",
            f"EVALUATION.online_text_encoder_path={args.online_text_encoder_path}",
            f"EVALUATION.text_encoder_device=cuda:0",  # Remapped by CUDA_VISIBLE_DEVICES
            f"EVALUATION.device=cuda:1",  # Remapped by CUDA_VISIBLE_DEVICES
        ]
        
        # Build command
        cmd = [
            sys.executable,
            "experiments/robotwin/eval_robotwin_single.py",
        ] + overrides
        
        # Set environment for this worker
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env["CUDA_VISIBLE_DEVICES"] = f"{text_encoder_gpu},{model_gpu}"
        env["WANDB_MODE"] = "offline"  # Workers don't log directly
        
        # Run subprocess
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
        
        # Parse output
        success_history = []
        final_success_rate = 0.0
        final_success_count = 0
        
        for line in process.stdout:
            print(f"[Worker {worker_id}/{task_name}] {line}", end="")
            
            result = extract_success_rate(line)
            if result:
                suc, test_num, rate = result
                success_history.append({
                    "episode": test_num,
                    "success_count": suc,
                    "cumulative_success_rate": rate,
                })
                
                # Send to main process for WandB logging
                wandb_queue.put({
                    "task_name": task_name,
                    "episode": test_num,
                    "success_count": suc,
                    "success_rate": rate,
                })
                
                final_success_rate = rate
                final_success_count = suc
        
        process.wait()
        
        # Send result to main process
        result_queue.put({
            "worker_id": worker_id,
            "task_name": task_name,
            "success_rate": final_success_rate,
            "success_count": final_success_count,
            "total_episodes": args.num_episodes,
            "success_history": success_history,
        })
        
        print(f"[Worker {worker_id}] Task {task_name} completed: {final_success_rate:.1f}%")
    
    print(f"[Worker {worker_id}] Shutting down")


def wandb_logger_process(
    wandb_queue: mp.Queue,
    result_queue: mp.Queue,
    num_tasks: int,
    args,
):
    """Process that receives logs from workers and sends to WandB."""
    
    # Initialize WandB
    run_name = args.wandb_run_name or f"cosmos-wam-multigpu-{time.strftime('%Y%m%d-%H%M%S')}"
    
    wandb_run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=run_name,
        tags=args.wandb_tags + ["robotwin", "multigpu", f"workers_{args.num_workers}"],
        config={
            "checkpoint": args.ckpt,
            "tasks": args.tasks,
            "num_episodes": args.num_episodes,
            "num_workers": args.num_workers,
            "num_inference_steps": args.num_inference_steps,
            "replan_steps": args.replan_steps,
            "instruction_type": args.instruction_type,
            "mixed_precision": args.mixed_precision,
        },
    )
    
    print(f"[WandB] Run started: {wandb_run.url}")
    
    completed_tasks = 0
    active_tasks = set()
    
    while completed_tasks < num_tasks:
        # Process WandB logs
        try:
            log_data = wandb_queue.get(timeout=0.1)
            task_name = log_data["task_name"]
            active_tasks.add(task_name)
            
            wandb_run.log({
                f"{task_name}/cumulative_success_rate": log_data["success_rate"],
                f"{task_name}/success_count": log_data["success_count"],
                f"{task_name}/episode": log_data["episode"],
            }, step=log_data["episode"])
        except:
            pass
        
        # Process results
        try:
            result = result_queue.get(timeout=0.1)
            task_name = result["task_name"]
            
            wandb_run.summary[f"{task_name}/final_success_rate"] = result["success_rate"]
            wandb_run.summary[f"{task_name}/final_success_count"] = result["success_count"]
            wandb_run.summary[f"{task_name}/total_episodes"] = result["total_episodes"]
            
            completed_tasks += 1
            print(f"[WandB] {completed_tasks}/{num_tasks} tasks completed: {task_name} = {result['success_rate']:.1f}%")
        except:
            pass
    
    # Compute overall statistics
    # (This would need to accumulate results - simplified here)
    
    print(f"[WandB] All tasks completed")
    wandb.finish()


def main():
    args = parse_args()
    
    # Validate GPU configuration
    total_gpus_needed = args.num_workers * 2
    print(f"Using {args.num_workers} workers, each using 2 GPUs")
    print(f"Total GPUs needed: {total_gpus_needed}")
    print(f"GPU assignment: Worker i gets GPUs {(args.gpu_start_id, args.gpu_start_id + 1)}, etc.")
    
    # Create task queue
    task_queue = mp.Queue()
    for i, task in enumerate(args.tasks):
        task_queue.put((i + 1, task))
    
    # Add poison pills for workers
    for _ in range(args.num_workers):
        task_queue.put((None, None))
    
    # Create result and wandb queues
    result_queue = mp.Queue()
    wandb_queue = mp.Queue()
    
    # Start WandB logger process
    wandb_proc = mp.Process(
        target=wandb_logger_process,
        args=(wandb_queue, result_queue, len(args.tasks), args),
    )
    wandb_proc.start()
    
    # Start worker processes
    workers = []
    for i in range(args.num_workers):
        text_encoder_gpu = args.gpu_start_id + i * 2
        model_gpu = args.gpu_start_id + i * 2 + 1
        
        worker = mp.Process(
            target=worker_process,
            args=(i, text_encoder_gpu, model_gpu, task_queue, result_queue, args, wandb_queue),
        )
        worker.start()
        workers.append(worker)
    
    # Wait for all workers to complete
    for worker in workers:
        worker.join()
    
    # Wait for WandB logger to finish
    wandb_proc.join()
    
    print("\n" + "="*80)
    print("All evaluations completed!")
    print("="*80)


if __name__ == "__main__":
    # Required for multiprocessing
    mp.set_start_method("spawn", force=True)
    main()
