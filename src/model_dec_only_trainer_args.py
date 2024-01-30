"""
    用于trainer4.py的训练参数
"""
from typing import Optional, List
from dataclasses import dataclass, field

import torch

from transformers import TrainingArguments
from transformers.utils import is_torch_bf16_gpu_available


@dataclass
class TrainingArgumentsWithMyDefault(TrainingArguments):
    output_dir: str = field(
        default="../outputs",
        metadata={
            "help": (
                "The output directory where the model predictions and checkpoints will be written."
            )
        },
    )
    log_level: str = field(
        default="info",
        metadata={"help": ("The log level.")},
    )
    seed: int = field(
        default=42,
        metadata={"help": ("Random seed.")},
    )
    do_train: bool = field(
        default=True,
        metadata={"help": ("Whether to run training.")},
    )
    do_eval: bool = field(
        default=True,
        metadata={"help": ("Whether to run eval on the dev set.")},
    )
    do_predict: bool = field(
        default=False,
        metadata={"help": ("Whether to run predictions on the test set.")},
    )
    bf16: bool = field(
        default=torch.cuda.is_available() and is_torch_bf16_gpu_available(),
        metadata={"help": ("Whether to use bf16.")},
    )
    fp16: bool = field(
        default=torch.cuda.is_available() and not is_torch_bf16_gpu_available(),
        metadata={"help": ("Whether to use fp16.")},
    )
    evaluation_strategy: str = field(
        default="epoch",
        metadata={"help": ("The evaluation strategy to use.")},
    )
    per_device_train_batch_size: int = field(
        default=1,
        metadata={"help": ("Batch size per GPU/TPU core/CPU for training.")},
    )
    per_device_eval_batch_size: int = field(
        default=1,
        metadata={"help": ("Batch size per GPU/TPU core/CPU for evaluation.")},
    )
    learning_rate: float = field(
        default=5e-5,
        metadata={"help": ("The initial learning rate for Adam.")},
    )
    gradient_accumulation_steps: int = field(
        default=1,
        metadata={
            "help": (
                "Number of updates steps to accumulate before performing a backward/update pass."
            )
        },
    )
    weight_delay: float = field(
        default=0.01,
        metadata={"help": ("The delay of evaluation.")},
    )
    num_train_epochs: float = field(
        default=1.0,
        metadata={"help": ("Total number of training epochs to perform.")},
    )
    logging_strategy: str = field(
        default="steps",
        metadata={"help": ("The logging strategy to use.")},
    )
    logging_steps: int = field(
        default=20,
        metadata={"help": ("Log every X updates steps.")},
    )
    save_strategy: str = field(
        default="epoch",
        metadata={"help": ("The saving strategy to use.")},
    )
    report_to: str = field(
        default="tensorboard",
        metadata={"help": ("The report strategy to use.")},
    )
    use_lora: bool = field(
        default=True,
        metadata={"help": ("Whether to use lora.")},
    )
    gradient_checkpointing: bool = field(
        default=False,
        metadata={"help": ("Whether to use gradient checkpointing.")},
    )


pretrained_model_name_or_path = "Qwen/Qwen-VL-Chat"


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        default=pretrained_model_name_or_path,
        metadata={"help": ("The model checkpoint for weights initialization.")},
    )
    # tokenizer
    config_name_or_path: str = field(
        default=pretrained_model_name_or_path,
        metadata={"help": ("The name of the config.")},
    )
    tokenizer_name_or_path: str = field(
        default=pretrained_model_name_or_path,
        metadata={"help": ("The name of the tokenizer.")},
    )
    cache_dir: Optional[str] = field(
        default="/root/autodl-tmp/pretrain-model",
        metadata={
            "help": "Where do you want to store the pretrained models downloaded from huggingface.co"
        },
    )
    classification_head_hidden_size: Optional[int] = field(
        default=4096,
        metadata={"help": ("The hidden size of classification head.")},
    )
    num_pages: Optional[int] = field(
        default=20,
        metadata={"help": ("The number of pages.")},
    )


@dataclass
class LoraArguments:
    lora_r: int = field(
        default=1,
        metadata={"help": ("The number of r.")},
    )
    lora_dropout: float = field(
        default=0.05,
        metadata={"help": ("The dropout of lora.")},
    )
    lora_alpha: int = field(
        default=16,
        metadata={"help": ("The alpha of lora.")},
    )
    lora_target_modules: List[str] = field(
        default_factory=lambda: [
            "c_attn",
            "attn.c_proj",
            "w1",
            "w2",
        ]  ##["in_proj","out_proj","c_fc"]
    )
    lora_weight_path: str = field(
        default="",
        metadata={"help": ("The path of lora weight.")},
    )
    lora_bias: str = field(
        default="none",
        metadata={"help": ("The bias of lora.")},
    )
    q_lora: bool = field(
        default=False,
        metadata={"help": ("Whether to use q_lora.")},
    )


DATA_DIR = "/root/autodl-tmp/data"


@dataclass
class DataTrainingArguments:
    # dataset
    train_json_path: Optional[str] = field(
        default=f"{DATA_DIR}/train_filter.json",
        metadata={"help": ("The path of train dataset.")},
    )
    eval_json_path: Optional[str] = field(
        default=f"{DATA_DIR}/val_filter.json",
        metadata={"help": ("The path of eval dataset.")},
    )
    test_json_path: Optional[str] = field(
        default=f"{DATA_DIR}/test.json",
        metadata={"help": ("The path of test dataset.")},
    )
    image_dir: str = field(
        default=f"{DATA_DIR}/images",
        metadata={"help": ("The path of images.")},
    )
    # tokenizer
    padding_side: str = field(
        default="right",
        metadata={"help": ("The padding side for the tokenizer.")},
    )
    max_seq_length: int = field(
        default=1024,
        metadata={
            "help": ("The maximum total input sequence length after tokenization.")
        },
    )
