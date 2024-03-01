
import sys
from typing import Optional
from dataclasses import dataclass, field
from datetime import datetime
import logging

import torch
from torch.utils.data import DataLoader
import transformers
from transformers import (
    TrainingArguments,
    Trainer,
    HfArgumentParser
)

from utils import seed_everything
from eva02_clip import load_model_tokenizer_transform, TripletLoss
from eva02_clip_dataset import TripletLossDataset

today_date = datetime.now().strftime("%Y-%m-%d")

eva02_clip_path = "/home/klwang/pretrain-model/eva02_large_patch14_clip_224.merged2b_s4b_b131k"

logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    model_name: str = "EVA02-L-14"
    cpkt_path: Optional[str] = field(default=eva02_clip_path)


@dataclass
class DataArguments:
    train_json_path: str = field(
        default="/home/klwang/data/MPDocVQA/train.json")
    eval_data_path: str = field(default="/home/klwang/data/MPDocVQA/val.json")
    image_dir: str = field(default="/home/klwang/data/MPDocVQA/images")


@dataclass
class Eva02ClipTrainingArguments(TrainingArguments):
    output_dir: str = field(default=f"../output_clip_triplet/{today_date}")
    optim: str = field(default="adamw_torch")
    margin: float = field(default=0.3, metadata={
                          "help": ("Triplet Loss Margin")})


class EVA02TripletLossTrainer(Trainer):
    def __init__(self, **kwargs):
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

    def compute_loss(self, model, inputs, return_outputs=False):
        triplet_loss_fct = TripletLoss(self.args.margin)
        anchor_features = model.encode_text(inputs["text"], normalize=True)
        positive_features = model.encode_image(
            inputs["positivate_image"], normalize=True)
        negitiva_features = model.encode_image(
            inputs["negitive_image"], normalize=True)
        loss = triplet_loss_fct(anchor_features,positive_features,negitiva_features)
        return loss

def main():
    parser = HfArgumentParser(
        (ModelArguments, DataArguments, Eva02ClipTrainingArguments))
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
    # load model
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
    # load dataset
    train_dataset = TripletLossDataset(
        data_args.train_json_path,
        data_args.image_dir,
        tokenizer,
        preprocess_train
    )
    trainer = EVA02TripletLossTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset
    )

    trainer.train()


if __name__ == "__main__":
    main()
