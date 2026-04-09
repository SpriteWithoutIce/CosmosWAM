"""
Cosmos-WAM RoboTwin Evaluation with WandB logging.

This script runs evaluation on multiple tasks and logs success rates to WandB
in real-time. For each task, it tracks cumulative success rate vs episode number.

Usage:
    python experiments/robotwin/eval_robotwin_wandb.py \
        --config configs/sim_robotwin.yaml \
        --tasks adjust_bottle click_bell ... \
        --num_episodes 50 \
        --wandb_project cosmos-wam-robotwin \
        --wandb_entity your_entity

Or use the wrapper script:
    bash robotwin_wandb.sh
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import wandb


def parse_args():
    parser = argparse.ArgumentParser(description="Cosmos-WAM RoboTwin Evaluation with WandB")
    
    # Config
    parser.add_argument("--config", type=str, default="configs/sim_robotwin.yaml",
                        help="Path to Hydra config file")
    parser.add_argument("--ckpt", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--tasks", type=str, nargs="+", required=True,
                        help="List of task names to evaluate")
    parser.add_argument("--num_episodes", type=int, default=50,
                        help="Number of episodes per task")
    
    # WandB settings
    parser.add_argument("--wandb_project", type=str, default="cosmos-wam-robotwin",
                        help="WandB project name")
    parser.add_argument("--wandb_entity", type=str, default=None,
                        help="WandB entity/team name")
    parser.add_argument("--wandb_run_name", type=str, default=None,
                        help="WandB run name (auto-generated if not provided)")
    parser.add_argument("--wandb_tags", type=str, nargs="+", default=[],
                        help="WandB tags")
    
    # Evaluation settings (overrides)
    parser.add_argument("--gpu_id", type=int, default=0,
                        help="GPU ID for evaluation")
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
    parser.add_argument("--use_online_text_encoder", action="store_true",
                        help="Use online text encoder")
    parser.add_argument("--online_text_encoder_path", type=str, default=None,
                        help="Path to Cosmos-Reason1-7B model")
    parser.add_argument("--text_encoder_device", type=str, default="cuda:0",
                        help="Device for text encoder")
    parser.add_argument("--model_device", type=str, default="cuda:1",
                        help="Device for main model")
    
    return parser.parse_args()


def extract_success_rate(line: str) -> Optional[Tuple[int, int, float]]:
    """Extract success count, total count, and rate from log line.
    
    Returns:
        (suc, test_num, rate) or None if not found
    """
    # Pattern: "Success rate: suc/test_num => XX.X%"
    pattern = r"Success rate:\s*(\d+)/(\d+)\s*=>\s*([\d.]+)%"
    match = re.search(pattern, line)
    if match:
        suc = int(match.group(1))
        test_num = int(match.group(2))
        rate = float(match.group(3))
        return suc, test_num, rate
    return None


def run_single_task(
    task_name: str,
    args,
    wandb_run: wandb.sdk.wandb_run.Run,
) -> dict:
    """Run evaluation for a single task and log to WandB.
    
    Returns:
        Dict with task results (success_rate, num_success, etc.)
    """
    print(f"\n{'='*80}")
    print(f"Starting evaluation: {task_name}")
    print(f"{'='*80}\n")
    
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
        f"gpu_id={args.gpu_id}",
    ]
    
    # Add online text encoder settings if enabled
    if args.use_online_text_encoder:
        overrides.extend([
            "EVALUATION.use_online_text_encoder=true",
            f"EVALUATION.online_text_encoder_path={args.online_text_encoder_path}",
            f"EVALUATION.text_encoder_device={args.text_encoder_device}",
            f"EVALUATION.device={args.model_device}",
        ])
    
    # Build command (Hydra format: key=value arguments)
    cmd = [
        sys.executable,
        "experiments/robotwin/eval_robotwin_single.py",
    ] + overrides
    
    print(f"Command: {' '.join(cmd)}\n")
    
    # Run subprocess and parse output
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )
    
    # Track success rates
    success_history = []
    final_success_rate = 0.0
    final_success_count = 0
    
    # Parse output in real-time
    for line in process.stdout:
        print(line, end="")  # Print to console
        
        # Extract success rate
        result = extract_success_rate(line)
        if result:
            suc, test_num, rate = result
            success_history.append({
                "episode": test_num,
                "success_count": suc,
                "cumulative_success_rate": rate,
            })
            
            # Log to WandB
            if wandb_run is not None:
                try:
                    wandb_run.log({
                        f"{task_name}/cumulative_success_rate": rate,
                        f"{task_name}/success_count": suc,
                        f"{task_name}/episode": test_num,
                    }, step=test_num)
                    print(f"[WandB] Logged: {task_name} episode {test_num}, rate={rate:.1f}%")
                except Exception as e:
                    print(f"[WandB] Warning: Failed to log - {e}")
            
            final_success_rate = rate
            final_success_count = suc
    
    process.wait()
    
    # Log final summary
    if wandb_run is not None:
        wandb_run.summary[f"{task_name}/final_success_rate"] = final_success_rate
        wandb_run.summary[f"{task_name}/final_success_count"] = final_success_count
        wandb_run.summary[f"{task_name}/total_episodes"] = args.num_episodes
    
    return {
        "task_name": task_name,
        "success_rate": final_success_rate,
        "success_count": final_success_count,
        "total_episodes": args.num_episodes,
        "success_history": success_history,
    }


def main():
    args = parse_args()
    
    # Initialize WandB
    run_name = args.wandb_run_name or f"cosmos-wam-{time.strftime('%Y%m%d-%H%M%S')}"
    
    wandb_run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=run_name,
        tags=args.wandb_tags + ["robotwin", f"episodes_{args.num_episodes}"],
        settings=wandb.Settings(start_method="thread"),
        config={
            "checkpoint": args.ckpt,
            "tasks": args.tasks,
            "num_episodes": args.num_episodes,
            "num_inference_steps": args.num_inference_steps,
            "replan_steps": args.replan_steps,
            "instruction_type": args.instruction_type,
            "mixed_precision": args.mixed_precision,
            "use_online_text_encoder": args.use_online_text_encoder,
        },
    )
    
    if wandb_run is None:
        print("Running without WandB logging")
    else:
        print(f"WandB run: {wandb_run.url}")
    print(f"Evaluating {len(args.tasks)} tasks, {args.num_episodes} episodes each\n")
    
    # Run all tasks
    all_results = []
    for i, task_name in enumerate(args.tasks, 1):
        print(f"\n[{i}/{len(args.tasks)}] Evaluating task: {task_name}")
        
        try:
            result = run_single_task(task_name, args, wandb_run)
            all_results.append(result)
            
            print(f"\n✓ Task {task_name} completed: {result['success_rate']:.1f}% success rate")
            
        except Exception as e:
            print(f"\n✗ Task {task_name} failed: {e}")
            if wandb_run:
                wandb_run.log({f"{task_name}/error": str(e)})
    
    # Compute overall statistics
    avg_success_rate = sum(r["success_rate"] for r in all_results) / len(all_results) if all_results else 0
    
    if wandb_run is not None:
        wandb_run.summary["overall/average_success_rate"] = avg_success_rate
        wandb_run.summary["overall/num_tasks"] = len(all_results)
    
    print(f"\n{'='*80}")
    print(f"Evaluation Complete!")
    print(f"{'='*80}")
    print(f"Average success rate across {len(all_results)} tasks: {avg_success_rate:.1f}%")
    if wandb_run is not None:
        print(f"\nWandB run URL: {wandb_run.url}")
        wandb.finish()
    else:
        print("\n(WandB not available)")


if __name__ == "__main__":
    main()
