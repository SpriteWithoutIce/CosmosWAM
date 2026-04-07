# Cosmos-WAM RoboTwin Evaluation

This directory contains the evaluation code for running Cosmos-WAM on the RoboTwin benchmark.

## Structure

```
experiments/robotwin/
├── cosmos_wam_policy/              # Policy implementation for RoboTwin
│   ├── __init__.py
│   ├── deploy_policy.py            # Main policy class (CosmosWAMRobotWinPolicy)
│   └── deploy_policy.yml           # Policy configuration
├── eval_robotwin_single.py         # Single task evaluation entry point
├── run_robotwin_manager.py         # Multi-GPU task manager for all tasks
└── README.md                       # This file
```

## Prerequisites

1. **RoboTwin Installation**: Ensure RoboTwin is installed at `third_party/RoboTwin`
   ```bash
   mkdir -p third_party
   git clone https://github.com/RoboTwin-Platform/RoboTwin.git third_party/RoboTwin
   # Follow RoboTwin's installation instructions
   ```

2. **Setup Policy Symlink**:
   ```bash
   bash scripts/setup_robotwin_eval.sh
   ```

3. **Text Embedding Cache**: Precompute text embeddings for all tasks
   ```bash
   # TODO: Add script to precompute text embeddings for RoboTwin tasks
   ```

4. **Dataset Statistics**: Use the `dataset_stats.json` from training

## Evaluation

### Single Task

Evaluate on a single RoboTwin task:

```bash
python experiments/robotwin/eval_robotwin_single.py \
  ckpt=./outputs/cosmos_2b_robotwin_20260407_090447/checkpoints/step_0005000_bf16.pt \
  EVALUATION.task_name=click_alarmclock \
  EVALUATION.dataset_stats_path=./dataset_stats.json \
  EVALUATION.text_embedding_cache_dir=/home/jwhe/linyihan/datasets/text_embeds_cache \
  gpu_id=0
```

### All Tasks (Multi-GPU)

Evaluate on all 10 RoboTwin tasks in parallel:

```bash
python experiments/robotwin/run_robotwin_manager.py \
  ckpt=./outputs/cosmos_2b_robotwin_20260407_090447/checkpoints/step_0005000_bf16.pt \
  EVALUATION.dataset_stats_path=./dataset_stats.json \
  EVALUATION.text_embedding_cache_dir=/home/jwhe/linyihan/datasets/text_embeds_cache \
  MULTIRUN.num_gpus=4
```

This will:
1. Run all 10 tasks in parallel across 4 GPUs
2. Each task runs both `demo_clean` and `demo_randomized` phases
3. Results are saved to `evaluate_results/robotwin/{ckpt_tag}/{timestamp}/`

## Configuration

Key configuration options in `configs/sim_robotwin.yaml`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `EVALUATION.action_horizon` | 8 | Number of actions to predict |
| `EVALUATION.replan_steps` | 4 | Actions to execute before replanning |
| `EVALUATION.num_inference_steps` | 20 | Diffusion sampling steps |
| `EVALUATION.eval_num_episodes` | 50 | Number of evaluation episodes |
| `EVALUATION.instruction_type` | unseen | Use "unseen" or "seen" instructions |

Override via command line:
```bash
EVALUATION.action_horizon=16 EVALUATION.replan_steps=8
```

## Output

Results are saved to:
```
evaluate_results/robotwin/{ckpt_tag}/{timestamp}/
├── summary.csv                     # Per-task success rates
├── summary.json                    # Detailed results
├── failed_tasks.txt                # List of failed tasks (if any)
├── {task_name}/
│   ├── _result_clean.txt           # Clean phase success rate
│   └── _result_random.txt          # Random phase success rate
└── eval_{task_name}_{timestamp}.log # Evaluation logs
```

## Implementation Details

### Image Processing

- Uses only `head_camera` (single camera, 240×320)
- Resized from original resolution to match training
- Normalized to [0, 1] range for VAE encoding

### Action Prediction

1. Encode first frame with VAE
2. Run DiT to extract video conditioning features
3. Use Action Head with flow matching to predict action sequence
4. Denormalize actions using dataset statistics

### Replanning Strategy

- Predict `action_horizon` actions at each replan step
- Execute `replan_steps` actions before observing and replanning
- Default: predict 8 actions, execute 4, then replan
