python experiments/robotwin/eval_robotwin_single.py \
    ckpt=/mnt/data/linyihan/ckpt/step_0002000_bf16.pt \
    EVALUATION.task_name=adjust_bottle \
    EVALUATION.dataset_stats_path=./dataset_stats.json \
    EVALUATION.fixed_text_embedding_path=/mnt/data/linyihan/ckpt/0aa37248d1da4ad461c558c5652997440e6bced3eb30752248000c9cc081774e.t5_len512.pt \
    gpu_id=7