#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1
DIR='/home/klwang/code/python/GuiQuQu-mpdocvqa'
cd $DIR
MODEL="/home/klwang/pretrain-model/eva02_large_patch14_clip_224.merged2b_s4b_b131k" 
# Qwen/Qwen-VL-Chat-Int4 Set the path if you do not want to load from huggingface directly
# ATTENTION: specify the path to your training data, which should be a json file consisting of a list of conversations.
# See the section for finetuning in README for more information.
DATA_PATH="/home/klwang/data/MPDocVQA/"
TRAIN_DATA="$DATA_PATH/train.json"
IMAGE_DIR="$DATA_PATH/images"
export CUDA_VISIBLE_DEVICES=0

python src/eva02_clip_triplet_trainer.py \
    --model_name "EVA02-L-14" \
    --cpkt_path $MODEL \
    --train_json_path $TRAIN_DATA \
    --image_dir $IMAGE_DIR \
    --output_dir output_clip_triplet \
    --bf16 True \
    --num_train_epochs 5 \
    --per_device_train_batch_size 64 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "epoch" \
    --save_total_limit 10 \
    --learning_rate 1e-5 \
    --weight_decay 0.1 \
    --warmup_ratio 0.1 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --report_to "tensorboard" \
    --deepspeed ds_config/ds_config_zero2_for_clip.json