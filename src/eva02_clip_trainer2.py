"""
    自己手写ddp来训练clip

"""
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
import json
import logging
import pathlib
from pathlib import Path
from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, RandomSampler
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import torch.multiprocessing as mp

from transformers.optimization import get_scheduler

from eva02_clip_trainer2_args import get_args

logger = logging.getLogger(__name__)


def end_log(
    loss: float, lr: float, step: int, epoch: float, args
):
    state = {
        "loss": loss,
        "lr": lr,
        "step": step,
        "epoch": epoch,
    }
    log = json.dumps(state, ensure_ascii=False)
    tqdm.write(log)


def end_step(inner_logger: logging.Logger, loss: float, lr: float, step: int, args):
    pass


def unwarp_model(model: nn.Module) -> nn.Module:
    if hasattr(model, "module"):
        return model.module
    else:
        return model


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_dataset: Dataset,
        eval_dataset: Dataset,
        collate_fn,
        args,
    ) -> None:
        """
        model:传入fp32的模型,内部会根据参数自行转换
        """
        self.model = model
        # model precision change
        self.model = self.prepare_model()
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.args = args
        self.collate_fn = collate_fn

        self.initialize_output_dir()
        if self.args.use_ddp:
            self.initialize_ddp()
            self.finish_wrap_model = False
            self.wrap_model()
        pass

    def prepare_model(self):
        if self.args.fp16:
            fp16 = 1
        elif self.args.bf16:
            bf16 = 1
        else:
            fp32 = 1
        assert fp16 + bf16 + fp32 == 1, "Only one of fp16, bf16, fp32 can be set"
        if fp16:
            self.model = self.model.to(dtype=torch.float16)
        elif bf16:
            self.model = self.model.to(dtype=torch.bfloat16)

    def initialize_output_dir(self):
        if self.is_main_process:
            output_dir = self.args.output_dir
            overwrite_output_dir = self.args.overwrite_output_dir
            if output_dir is None:
                raise ValueError("output_dir must be not None")
            output_dir = Path(output_dir)
            if output_dir.exists() and output_dir.is_dir() and not overwrite_output_dir:
                raise ValueError(
                    "output_dir esists and is not empty, set overwrite_output_dir=True to overwrite"
                )
            output_dir.mkdir(parents=True, exist_ok=True)

    @property
    def _train_batch_size(self):
        return self.args.per_device_train_batch_size * max(1, self.args.n_gpu)

    @property
    def is_main_process(self):
        return self.args.local_rank == 0

    def initialize_ddp(self):
        dist.init_process_group(
            backend="nccl",
            init_method=f"tcp://localhost:{self.args.ddp_tcp_port}",
            rank=self.args.local_rank,
            world_size=self.args.world_size,
        )
        torch.cuda.set_device(self.args.gpu_id)
        pass

    def wrap_model(self):
        if self.args.use_dpp and not self._created_wrap_model:
            self._created_wrap_model = True
            self.model = nn.parallel.DistributedDataParallel(
                module=self.model,
                device_ids=[self.args.gpu_id],
            )

    def train(self):
        self.model.train()
        # logging
        num_examples = len(self.train_dataset)
        one_epoch_steps = num_examples // self._train_batch_size
        max_steps = self.args.num_train_epochs * one_epoch_steps
        if self.is_main_process:
            number_training_parameters = sum(
                p.numel() for p in self.model.parameters() if p.requires_grad
            )
            logger.info("***** Running training *****")
            logger.info("  Num examples = %d", num_examples)
            logger.info("  Num Epochs = %f", self.args.num_train_epochs)
            logger.info(
                "  Instantaneous per device batch size = %d",
                self.args.per_device_train_batch_size,
            )
            logger.info(
                "  Total train batch size (w. parallel, distributed) = %d",
                self._train_batch_size,
            )
            logger.info("  Total optimization steps = %d", max_steps)
            logger.info(
                " Number of trainable parameters: %i", number_training_parameters
            )

        # data loader
        train_dataloader = self.create_train_dataloader()
        # optimizer and scheduler
        optimizer, scheduler = self.create_optimizer_and_scheduler()

        # training loop
        with tqdm(
            total=max_steps, desc="Training", disable=not self.is_main_process
        ) as pbar:
            for epoch in range(self.args.num_train_epochs):
                for step, batch in enumerate(train_dataloader):
                    loss, lr = self.training_step(
                        self.model, batch, optimizer=optimizer, scheduler=scheduler
                    )

                    if self.is_main_process:
                        pbar.update(1)
                        end_step(
                            logger,
                            loss=loss,
                            lr=lr,
                            step=epoch * one_epoch_steps + step,
                            args=self.args,
                        )

    def training_step(self, model, input, optimizer, scheduler):
        with torch.cuda.amp.autocast():
            loss = self.compute_loss(model, input)

        pass

    def compute_loss(self, model, inputs):
        pass

    def evaluate(self):
        pass

    def evaluate_step(self):
        pass

    def create_train_dataloader(self):
        train_dataset = self.train_dataset
        data_collator = self.collate_fn
        dataloader_params = {
            "batch_size": self.args.per_device_train_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "samplers": self._get_train_sampler(),
            "drop_last": self.args.dataloader_drop_last,
        }
        return DataLoader(train_dataset, **dataloader_params)

    def _get_train_sampler(self):
        if self.args.use_ddp and self.train_sampler is None:
            self.train_sampler = DistributedSampler(
                self.train_dataset,
                num_replicas=self.args.world_size,
                rank=self.args.local_rank,
                shuffle=True,
                seed=self.args.seed,
                drop_last=self.args.dataloader_drop_last,
            )
        else:
            self.train_sampler = RandomSampler(self.train_dataset)
        return self.train_sampler

    def create_eval_dataloader(self):
        pass

    def create_optimizer(self):
        weight_decay_params, no_weight_decay_params = [], []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if "bias" in name or "LayerNorm.weight" in name:
                no_weight_decay_params.append(param)
            else:
                weight_decay_params.append(param)
        params = [
            {"params": weight_decay_params, "weight_decay": self.args.weight_decay},
            {"params": no_weight_decay_params, "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(
            params=params,
            lr=self.args.learning_rate,
            betas=(self.args.adam_beta1, self.args.adam_beta2),
            eps=self.args.adam_epsilon,
        )
        self.optimizer = optimizer

    def create_scheduler(self, num_train_steps, optimizer):
        if self.lr_scheduler is None:
            if self.warmup_steps > 0:
                warmup_steps = self.args.warmup_steps
            else:
                warmup_steps = int(num_train_steps * self.args.warmup_ratio)
            self.lr_scheduler = get_scheduler(
                name=self.args.lr_scheduler_type,
                optimizer=optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=num_train_steps,
            )
            self._created_lr_scheduler = True

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        self.create_optimizer()
        self.create_scheduler(num_training_steps, self.optimizer)

    def save_model(self):
        pass


def main_worker(args):
    pass


"""
    新增参数
    n_gpu
    use_ddp
    world_size
    local_rank
    ddp_tcp_port
    gpu_id
"""
if __name__ == "__main__":
    # launcher
    cuda_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    n_gpu = len(cuda_devices.split(","))
    world_size = n_gpu
    args = get_args()
    args.n_gpu = n_gpu

    args.world_size = world_size
    args.ddp_tcp_port = 23456
    if n_gpu > 1:
        args.use_ddp = True
        for local_rank in range(world_size):
            args.local_rank = local_rank
            args.gpu_id = local_rank
            mp.spawn(main_worker, nprocs=n_gpu, args=(args,))
    pass
