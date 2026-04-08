python experiments/robotwin/eval_robotwin_single.py \
    ckpt=/mnt/data/linyihan/ckpt/step_0014000_bf16.pt \
    EVALUATION.task_name=beat_block_hammer \
    EVALUATION.dataset_stats_path=./dataset_stats.json \
    EVALUATION.robotwin_root=/root/linyihan/RoboTwin \
    EVALUATION.fixed_text_embedding_path=/mnt/data/linyihan/text_embeds_cache/robotwin_reason1/beat_block_hammer-demo_clean_collect_200-50/ae866c5598d41b239064669a6ad9326543b4225a826e1c6967cbe92a8ec89a47.t5_len512.pt \
    EVALUATION.num_inference_steps=20 \
    EVALUATION.replan_steps=4 \
    EVALUATION.instruction_type=seen \
    gpu_id=7