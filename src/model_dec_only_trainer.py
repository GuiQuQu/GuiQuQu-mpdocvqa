"""
    训练的模型文件为model_dec_only.py
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
    TrainingArguments,
    BitsAndBytesConfig,
)

from qwen_vl_chat import QWenTokenizer, QWenLMHeadModel, QWenConfig

from model_dec_only import MPModel, MPModelConfig, load_lora_qwen_vl_model
from utils import seed_everything
from model_dec_only_trainer_args import (
    TrainingArgumentsWithMyDefault,
    ModelArguments,
    LoraArguments,
    DataTrainingArguments,
)

from model_dec_only_dataset import (
    MPDocVQADataset,
    collate_fn_for_MPModel,
    collate_fn_for_qwen_vl_lora,
)

logger = logging.getLogger(__name__)


class MPModelTrainer(Trainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_train_dataloader(self) -> DataLoader:
        # super().get_train_dataloader()
        train_dataset = self.train_dataset
        data_collator = self.data_collator
        dataloader_params = {
            "batch_size": self._train_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
        }
        if not isinstance(train_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_train_sampler()
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = transformers.trainer_utils.seed_worker

        return self.accelerator.prepare(DataLoader(train_dataset, **dataloader_params))


def load_dataset(
    training_args: TrainingArguments,
    data_args: DataTrainingArguments,
    tokenizer: QWenTokenizer,
):
    train_dataset, eval_dataset, test_dataset = None, None, None

    if training_args.do_train:
        assert data_args.train_json_path is not None
        train_dataset = MPDocVQADataset(
            json_path=data_args.train_json_path,
            image_dir=data_args.image_dir,
            tokenizer=tokenizer,
            split="train",
        )
    if training_args.do_eval:
        assert data_args.eval_json_path is not None
        eval_dataset = MPDocVQADataset(
            json_path=data_args.eval_json_path,
            image_dir=data_args.image_dir,
            tokenizer=tokenizer,
            split="train",
        )
    if training_args.do_predict:
        assert data_args.test_json_path is not None
        test_dataset = MPDocVQADataset(
            json_path=data_args.test_json_path,
            image_dir=data_args.image_dir,
            tokenizer=tokenizer,
            split="test",
        )
    return train_dataset, eval_dataset, test_dataset


def load_tokenizer(
    model_args: ModelArguments, data_args: DataTrainingArguments
):
    tokenizer = QWenTokenizer.from_pretrained(
        model_args.tokenizer_name_or_path,
        cache_dir=model_args.cache_dir,
        model_max_length=data_args.max_seq_length,
        padding_side=data_args.padding_side,
    )
    tokenizer.pad_token = "<|endoftext|>"
    tokenizer.eos_token = "<|endoftext|>"
    return tokenizer


def load_config(
    model_args: ModelArguments,
    training_args: TrainingArgumentsWithMyDefault,
):
    config = MPModelConfig.from_pretrained(
        model_args.config_name_or_path,
        cache_dir=model_args.cache_dir,
        classification_head_hidden_size=model_args.classification_head_hidden_size,
        num_pages=model_args.num_pages,
    )
    config.fp16 = training_args.fp16
    config.bf16 = training_args.bf16
    config.use_cache = False
    return config


def load_qwen_vl_model(
    model_args: ModelArguments, config: MPModelConfig, device: str
):
    qwen_vl = QWenLMHeadModel.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        config=config,
        device_map="auto",
    )
    return qwen_vl


def main():
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArgumentsWithMyDefault, LoraArguments)
    )
    model_args, data_args, training_args, lora_args = parser.parse_args_into_dataclasses()

    # logging setting
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    seed_everything(training_args.seed)
    logger.info("training_args.device: %s" % training_args.device)
    logger.info(f"bf16={training_args.bf16}, fp16={training_args.fp16}")
    # tokenizer
    tokenizer = load_tokenizer(model_args, data_args)
    # dataset
    train_dataset, eval_dataset, test_dataset = load_dataset(
        training_args, data_args, tokenizer
    )
    assert train_dataset is not None

    # model
    ## config
    config = load_config(model_args, training_args)
    qwen_vl = load_qwen_vl_model(model_args, config, None)
    if training_args.use_lora:
        qwen_vl = load_lora_qwen_vl_model(
            qwen_vl=qwen_vl,
            r=lora_args.lora_r,
            lora_alpha=lora_args.lora_alpha,
            lora_dropout=lora_args.lora_dropout,
        )
    model: MPModel = MPModel(config=config, qwen_vl=qwen_vl)

    model.print_trainable_parameters()

    trainer = MPModelTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=collate_fn_for_MPModel,
    )
    trainer.train()

    # # trainer
    # trainer = Trainer(
    #     model=qwen_vl_lora,
    #     args=training_args,
    #     train_dataset=train_dataset,
    #     eval_dataset=eval_dataset,
    #     tokenizer=tokenizer,
    #     data_collator=collate_fn_for_qwen_vl_lora,
    # )
    # trainer.train()


if __name__ == "__main__":
    main()
