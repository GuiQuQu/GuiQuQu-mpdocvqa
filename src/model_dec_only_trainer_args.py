"""
    用于trainer4.py的训练参数
"""
from typing import Optional
from dataclasses import dataclass, field

import torch

from transformers import TrainingArguments


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
    bf16: bool = field(
        default=True,
        metadata={"help": ("Whether to use bf16.")},
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
        metadata={"help": ("Number of updates steps to accumulate before performing a backward/update pass.")},
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
        default=100,
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

@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        default="../pretrain-model/QWen-VL-Chat",
        metadata={"help": ("The model checkpoint for weights initialization.")},
    )
    # tokenizer
    config_name_or_path: str = field(
        default="../pretrain-model/QWen-VL-Chat",
        metadata={"help": ("The name of the config.")},
    )
    tokenizer_name_or_path: str = field(
        default="../pretrain-model/QWen-VL-Chat",
        metadata={"help": ("The name of the tokenizer.")},
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where do you want to store the pretrained models downloaded from huggingface.co"
        },
    )


@dataclass
class ModelArgumentsWithLora(ModelArguments):
    lora_rank: Optional[int] = field(
        default=1,
        metadata={"help": "The rank of lora model."},
    )
    lora_alpha: Optional[float] = field(
        default=32,
        metadata={"help": "lora alpha."},
    )
    lora_dropout: Optional[float] = field(
        default=0.05,
        metadata={"help": "lora dropout."},
    )


@dataclass
class DataTrainingArguments:
    # dataset
    train_json_path: Optional[str] = field(
        default="../data/MPDocVQA/train.json",
        metadata={"help": ("The path of train dataset.")},
    )
    eval_json_path: Optional[str] = field(
        default="../data/MPDocVQA/val.json",
        metadata={"help": ("The path of eval dataset.")},
    )
    test_json_path: Optional[str] = field(
        default="../data/MPDocVQA/test.json",
        metadata={"help": ("The path of test dataset.")},
    )
    image_dir: str = field(
        default="../data/MPDocVQA/images",
        metadata={"help": ("The path of images.")},
    )
    # tokenizer
    padding_side: str = field(
        default="left",
        metadata={"help": ("The padding side for the tokenizer.")},
    )
    max_seq_length: int = field(
        default=1024,
        metadata={
            "help": ("The maximum total input sequence length after tokenization.")
        },
    )
