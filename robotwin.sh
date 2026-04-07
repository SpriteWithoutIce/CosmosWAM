python experiments/robotwin/eval_robotwin_single.py \
    ckpt=/mnt/data/linyihan/ckpt/step_0002000_bf16.pt \
    EVALUATION.task_name=beat_block_hammer \
    EVALUATION.dataset_stats_path=./dataset_stats.json \
    EVALUATION.robotwin_root=/root/linyihan/RoboTwin \
    EVALUATION.fixed_text_embedding_path=/mnt/data/linyihan/text_embeds_cache/robotwin_reason1 \
    EVALUATION.num_inference_steps=8 \
    EVALUATION.replan_steps=4 \
    gpu_id=7