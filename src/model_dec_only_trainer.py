"""
    训练的模型文件为model3.py
    架构为qwen-vl+分类头
"""
import sys
import logging
import torch
from torch.utils.data import DataLoader

import transformers
from transformers import (
    Trainer,
    HfArgumentParser,
    TrainingArguments
)

from qwen_vl_chat import QWenTokenizer

from model_dec_only import (
    MPModel,
    MPModelConfig,
    load_lora_qwen_vl_model
)
from utils import seed_everything
from model_dec_only_trainer_args import (
    TrainingArgumentsWithMyDefault,
    ModelArgumentsWithLora,
    DataTrainingArguments,
)


logger = logging.getLogger(__name__)

class MPModelTrainer(Trainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


def load_dataset():
    pass

def main():
    pass

if __name__ == "__main__":
    pass