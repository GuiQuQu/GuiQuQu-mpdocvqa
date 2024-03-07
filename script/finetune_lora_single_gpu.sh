#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1
DIR='/home/klwang/code/python/GuiQuQu-mpdocvqa'
cd $DIR
MODEL="/home/klwang/pretrain-model/QWen-VL-Chat" 
# Qwen/Qwen-VL-Chat-Int4 Set the path if you do not want to load from huggingface directly
# ATTENTION: specify the path to your training data, which should be a json file consisting of a list of conversations.
# See the section for finetuning in README for more information.
DATA_PATH="/home/klwang/data/MPDocVQA/"
TRAIN_DATA="$DATA_PATH/train_filter.json"
IMAGE_DIR="$DATA_PATH/images"
export CUDA_VISIBLE_DEVICES=0

python src/model_dec_only_finetune.py \
    --model_name_or_path $MODEL \
    --data_path $TRAIN_DATA \
    --image_dir $IMAGE_DIR \
    --bf16 True \
    --fix_vit True \
    --output_dir output_qwen_vl_lora \
    --num_train_epochs 5 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 10 \
    --learning_rate 1e-5 \
    --weight_decay 0.1 \
    --adam_beta2 0.95 \
    --warmup_ratio 0.01 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --report_to "tensorboard" \
    --model_max_length 1024 \
    --lazy_preprocess True \
    --gradient_checkpointing \
    --use_lora