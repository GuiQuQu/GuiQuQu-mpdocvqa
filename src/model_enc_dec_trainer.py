"""
    训练的model文件为model2.py
    架构为
    基于Hi-VT5,采用encoder-decoder架构,而且encoder和decoder均才用qwen-vl的模型结构
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
)

from qwen_vl import QWenTokenizer
from dataset import (
    MPDocVQADataset,
    MPDocVQADatasetForEncoderDecoderModel,
    MPDocVQADatasetForDecoderOnly2,
    CollatorForEncoderDecoderModel,
)
from model_enc_dec import (
    load_config,
    load_qwen_vl_model_lora_from_scratch,
    load_qwen_vl_tokenizer,
    MPModelConfig,
    MPModel,
)
from utils import seed_everything
from model_enc_dec_trainer_args import (
    ModelArgumentsWithLora,
    DataTrainingArguments,
    ModelArguments,
    TrainingArgumentsWithMyDefault,
)

logger = logging.getLogger(__name__)


def load_tokenizer(model_args: ModelArguments, data_args: DataTrainingArguments):
    tokenizer = load_qwen_vl_tokenizer(
        model_args.tokenizer_name_or_path,
        data_args.max_seq_length,
        data_args.padding_side,
    )
    return tokenizer


def load_lora_model(
    model_name_or_path: str,
    config: MPModelConfig,
    device: str,
    lora_r: int,
):
    qwen_vl = load_qwen_vl_model_lora_from_scratch(
        model_name_or_path, config, device, lora_r
    )
    model = MPModel(config=config, qwen_vl=qwen_vl)
    model.to(device)
    return model


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


def load_dataset(training_args: TrainingArguments, data_args: DataTrainingArguments):
    train_dataset, eval_dataset, test_dataset = None, None, None

    if training_args.do_train:
        assert data_args.train_json_path is not None
        train_dataset = MPDocVQADatasetForEncoderDecoderModel(
            json_path=data_args.train_json_path,
            image_dir=data_args.image_dir,
            split="train",
        )
    if training_args.do_eval:
        assert data_args.eval_json_path is not None
        eval_dataset = MPDocVQADatasetForEncoderDecoderModel(
            json_path=data_args.eval_json_path,
            image_dir=data_args.image_dir,
            split="eval",
        )
    if training_args.do_predict:
        assert data_args.test_json_path is not None
        test_dataset = MPDocVQADatasetForEncoderDecoderModel(
            json_path=data_args.test_json_path,
            image_dir=data_args.image_dir,
            split="test",
        )
    return train_dataset, eval_dataset, test_dataset


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
    train_dataset, eval_dataset, test_dataset = load_dataset(training_args, data_args)
    assert train_dataset is not None
    # collate_fn
    collator_fn = CollatorForEncoderDecoderModel(tokenizer=tokenizer)

    # model
    ## config
    config = load_config(
        model_args.config_name_or_path,
        pad_token_id=tokenizer.pad_token_id,
        model_max_length=data_args.max_seq_length,
    )
    logger.info(
        "bf16=%s, device: %s, lora_r=%s"
        % (config.bf16, training_args.device, model_args.lora_rank)
    )
    model = load_lora_model(
        model_name_or_path=model_args.model_name_or_path,
        config=config,
        device=training_args.device,
        lora_r=model_args.lora_rank,
    )

    # trainer
    trainer = MPModelTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=collator_fn,
    )
    trainer.train()


if __name__ == "__main__":
    main()
