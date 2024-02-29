DIR='/home/klwang/code/python/GuiQuQu-mpdocvqa'
cd $DIR

DATA_PATH="/home/klwang/data/MPDocVQA/"
BASE_MODEL_PATH="/home/klwang/pretrain-model/Qwen-VL-Chat-Int4"
ADAPTER_PATH=$DIR/output_qwen_vl

python src/model_dec_only_predict_eval.py \
    --output_dir $DIR/mpdocvqa-result \
    --adapter_name_or_path $ADAPTER_PATH/checkpoint-3000 \
    --base_model_name_or_path $BASE_MODEL_PATH \
    --do_eval \
    --image_dir $DATA_PATH/images \
    --eval_data_path $DATA_PATH/val_filter.json \
    --verbose