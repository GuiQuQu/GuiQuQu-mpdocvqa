"""
    训练的模型文件为model3.py
    架构为qwen-vl+分类头
"""
import sys
import logging
import torch
from torch.utils.data import DataLoader

import transformers
from transformers import Trainer, HfArgumentParser, TrainingArguments

from qwen_vl_chat import QWenTokenizer, QWenLMHeadModel, QWenConfig

from model_dec_only import MPModel, MPModelConfig, load_lora_qwen_vl_model
from utils import seed_everything
from model_dec_only_trainer_args import (
    TrainingArgumentsWithMyDefault,
    ModelArgumentsWithLora,
    DataTrainingArguments,
)

from model_dec_only_dataset import MPDocVQADataset, collate_fn_for_mp_doc_vqa

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
    model_args: ModelArgumentsWithLora, data_args: DataTrainingArguments
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
    model_args: ModelArgumentsWithLora,
):
    config = MPModelConfig.from_pretrained(
        model_args.config_name_or_path,
        cache_dir=model_args.cache_dir,
        classification_head_hidden_size=model_args.classification_head_hidden_size,
        num_pages=model_args.num_pages,
    )
    config.fp16 = True
    config.use_cache = False
    return config


def load_qwen_vl_model(
    model_args: ModelArgumentsWithLora, config: MPModelConfig, device: str
):
    qwen_vl = QWenLMHeadModel.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        config=config,
        device_map=device,
    )
    return qwen_vl


def main():
    parser = HfArgumentParser(
        (ModelArgumentsWithLora, DataTrainingArguments, TrainingArgumentsWithMyDefault)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

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
    logger.info("current process device: %s" % training_args.device)
    # tokenizer
    tokenizer = load_tokenizer(model_args, data_args)
    # dataset
    train_dataset, eval_dataset, test_dataset = load_dataset(
        training_args, data_args, tokenizer
    )
    assert train_dataset is not None
    # collate_fn
    # collator_fn = CollatorForEncoderDecoderModel(tokenizer=tokenizer)

    # model
    ## config
    config = load_config(model_args)
    qwen_vl = load_qwen_vl_model(model_args, config, training_args.device)
    qwen_vl_lora = load_lora_qwen_vl_model(
        qwen_vl=qwen_vl,
        r=model_args.lora_rank,
        lora_alpha=model_args.lora_alpha,
        lora_dropout=model_args.lora_dropout,
    )
    model = MPModel(config=config, qwen_vl=qwen_vl_lora)
    # trainer
    trainer = MPModelTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=collate_fn_for_mp_doc_vqa,
    )
    trainer.train()


if __name__ == "__main__":
    main()
