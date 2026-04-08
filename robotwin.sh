python experiments/robotwin/eval_robotwin_single.py \
    ckpt=/mnt/data/linyihan/ckpt/step_0014000_bf16.pt \
    EVALUATION.task_name=adjust_bottle \
    EVALUATION.use_online_text_encoder=true \
    EVALUATION.dataset_stats_path=./dataset_stats.json \
    EVALUATION.robotwin_root=/root/linyihan/RoboTwin \
    EVALUATION.online_text_encoder_path=/mnt/data/linyihan/Cosmos-Reason1-7b \
    EVALUATION.text_encoder_device=cuda:0 \
    EVALUATION.device=cuda:1 \
    EVALUATION.num_inference_steps=20 \
    EVALUATION.replan_steps=8 \
    EVALUATION.instruction_type=seen \
    gpu_id=1

