python experiments/robotwin/eval_robotwin_single.py \
    ckpt=/mnt/data/linyihan/ckpt/step_0014000_bf16.pt \
    EVALUATION.task_name=adjust_bottle \
    EVALUATION.dataset_stats_path=./dataset_stats.json \
    EVALUATION.robotwin_root=/root/linyihan/RoboTwin \
    EVALUATION.fixed_text_embedding_path=/mnt/data/linyihan/ckpt/0aa37248d1da4ad461c558c5652997440e6bced3eb30752248000c9cc081774e.t5_len512.pt \
    EVALUATION.num_inference_steps=20 \
    EVALUATION.replan_steps=8 \
    EVALUATION.instruction_type=seen \
    gpu_id=7