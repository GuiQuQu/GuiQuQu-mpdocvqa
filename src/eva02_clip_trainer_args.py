from typing import Optional
from dataclasses import dataclass, field
from datetime import datetime

import torch

from transformers import TrainingArguments

today_date = datetime.now().strftime("%Y-%m-%d")


@dataclass
class TrainingArgumentsWithDefault(TrainingArguments):
    output_dir: str = field(
        default=f"../clip-outputs/{today_date}",
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
    do_train: bool = field(
        default=True,
        metadata={"help": ("Whether to run training.")},
    )
    do_eval: bool = field(
        default=True,
        metadata={"help": ("Whether to run eval on the dev set.")},
    )
    fp16: bool = field(
        default=True,
        metadata={"help": ("Whether to use fp16.")},
    )
    bf16: bool = field(
        default=False,
        metadata={"help": ("Whether to use bf16.")},
    )
    evaluation_strategy: str = field(
        default="epoch",
        metadata={"help": ("The evaluation strategy to use.")},
    )
    per_device_train_batch_size: int = field(
        default=32,
        metadata={"help": ("Batch size per GPU/TPU core/CPU for training.")},
    )
    per_device_eval_batch_size: int = field(
        default=128,
        metadata={"help": ("Batch size per GPU/TPU core/CPU for evaluation.")},
    )
    seed: int = field(
        default=42,
        metadata={
            "help": ("Random seed that will be set at the beginning of training.")
        },
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
    weight_decay: float = field(
        default=0.05,
        metadata={"help": ("Weight decay.")},
    )
    num_train_epochs: float = field(
        default=15.0,
        metadata={"help": ("Total number of training epochs to perform.")},
    )
    logging_strategy: str = field(
        default="steps",
        metadata={"help": ("The logging strategy to use.")},
    )
    logging_steps: int = field(
        default=20,
        metadata={"help": ("The logging strategy to use.")},
    )
    save_strategy: str = field(
        default="epoch",
        metadata={"help": ("The save strategy to use.")},
    )
    report_to: Optional[str] = field(
        default="tensorboard",
        metadata={"help": ("The report strategy to use.")},
    )


@dataclass
class ModelArguments:
    model_name: str = field(
        default="EVA02-L-14",
        metadata={"help": ("The model name.")},
    )
    cpkt_path: str = field(
        default="../pretrain-model/eva02_large_patch14_clip_224.merged2b_s4b_b131k",
        metadata={"help": ("The checkpoint path.")},
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
