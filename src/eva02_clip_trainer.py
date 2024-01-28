"""
    采用huggingface的Trainer类来训练clip
"""
import sys
import logging
from typing import Any, Dict, List, Optional, Union
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


import transformers
from transformers import Trainer, HfArgumentParser, TrainingArguments
from transformers.trainer_utils import EvalLoopOutput

from dataset import MPDocVQADatasetForCLIP, CollatorForCLIP

from eva02_clip import load_model_tokenizer_transform, ClipLoss

from utils import seed_everything
from trainer_clip_args import (
    ModelArguments,
    DataTrainingArguments,
    TrainingArgumentsWithDefault,
)

logger = logging.getLogger(__name__)


class EVA02CLIPTrainer(Trainer):
    def __init__(sself, **kwargs):
        super().__init__(**kwargs)

    def get_train_dataloader(self) -> DataLoader:
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

    def get_eval_dataloader(self, eval_dataset: Dataset | None = None) -> DataLoader:
        return super().get_eval_dataloader(eval_dataset)

    def compute_loss(self, model, inputs, return_outputs=False):
        clip_loss = ClipLoss(
            local_loss=True,
            gather_with_grad=False,
            cache_labels=True,
            rank=self.accelerator.local_process_index,
            world_size=self.accelerator.num_processes,
        )
        # [bsz,hidden_size] [bsz, hidden_size]
        image_features, text_features, logit_scale = model(**inputs)
        loss, acc = clip_loss(image_features, text_features, logit_scale)
        return (loss, acc) if return_outputs else loss

    def evaluate(
        self,
        eval_dataset: Dataset | None = None,
        ignore_keys: List[str] | None = None,
        metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:
        return super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)

    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: bool | None = None,
        ignore_keys: List[str] | None = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        return super().evaluation_loop(
            dataloader,
            description,
            prediction_loss_only,
            ignore_keys,
            metric_key_prefix,
        )


def load_dataset(
    train_args: TrainingArguments,
    data_args: DataTrainingArguments,
    tokenzier,
    preprocess_train=None,
    preprocess_eval=None,
):
    train_dataset, eval_dataset, test_dataset = None, None, None
    if train_args.do_train:
        assert data_args.train_json_path is not None
        assert preprocess_train is not None
        train_dataset = MPDocVQADatasetForCLIP(
            data_args.train_json_path,
            data_args.image_dir,
            tokenizer=tokenzier,
            transform=preprocess_train,
            split="train",
        )
    if train_args.do_eval:
        assert data_args.eval_json_path is not None
        assert preprocess_eval is not None
        eval_dataset = MPDocVQADatasetForCLIP(
            data_args.eval_json_path,
            data_args.image_dir,
            tokenizer=tokenzier,
            transform=preprocess_eval,
            split="eval",
        )
    if train_args.do_predict:
        assert data_args.test_json_path is not None
        assert preprocess_eval is not None
        test_dataset = MPDocVQADatasetForCLIP(
            data_args.test_json_path,
            data_args.image_dir,
            tokenizer=tokenzier,
            transform=preprocess_eval,
            split="test",
        )
    return train_dataset, eval_dataset, test_dataset


def main():
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArgumentsWithDefault)
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

    (
        model,
        preprocess_train,
        preprocess_eval,
        tokenizer,
    ) = load_model_tokenizer_transform(
        model_args.model_name,
        model_args.cpkt_path,
        precision="fp32",
        device=training_args.device,
    )
    train_dataset, eval_dataset, test_dataset = load_dataset(
        training_args,
        data_args,
        tokenizer,
        preprocess_train,
        preprocess_eval,
    )
    trainer = EVA02CLIPTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=CollatorForCLIP(),
    )

    trainer.train()


if __name__ == "__main__":
    main()
