import argparse


def get_args():
    parser = argparse.ArgumentParser(description="Trainer For CLIP(EVA-02 Model)")
    # trainer args
    parser.add_argument(
        "--output_dir",
        type=str,
        default="../clip-outputs-ddp/",
        help="The output directory.",
    )
    parser.add_argument(
        "--overwrite_output_dir",
        type=bool,
        action="store_true",
        default=True,
        help="Overwrite the output directory.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="info",
    )
    parser.add_argument(
        "--fp16",
        type=bool,
        default=True,
        help="Whether to use fp16.",
    )
    parser.add_argument(
        "--bf16",
        type=bool,
        default=False,
        help="Whether to use bf16.",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=32,
        help="Batch size per GPU/TPU core/CPU for training.",
    )
    parser.add_argument(
        "num_train_epochs",
        type=float,
        default=10.0,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed that will be set at the beginning of training.",
    )
    # optimizer args
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="The initial learning rate for Adam.",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.05,
        help="Weight decay if we apply some.",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.1,
        help="Linear warmup over warmup_ratio * total_steps.this argument will be disenabled if warmup_steps > 0.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=str,
        choices=[
            "linear",
            "cosine",
            "cosine_with_restarts",
            "polynomial",
            "constant",
            "constant_with_warmup",
        ],
        default="cosine",
        help="The scheduler type to use.",
    )
    parser.add_argument(
        "--adam_beta1",
        type=float,
        default=0.9,
        help="Beta1 for Adam optimizer.",
    )
    parser.add_argument(
        "--adam_beta2",
        type=float,
        default=0.999,
        help="Beta2 for Adam optimizer.",
    )
    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-8,
        help="Epsilon for Adam optimizer.",
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.0,
        help="Max gradient norm.",
    )
    # logging args
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=100,
        help="Log every X updates steps.",
    )
    # save args
    parser.add_argument(
        "--save_strategy",
        type=str,
        choices=["epoch", "steps"],
        default="epoch",
        help="The save strategy to use.",
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=100,
        help="Save checkpoint every X updates steps. only enabled when save_strategy is 'steps'.",
    )
    # model args
    parser.add_argument(
        "--model_name", type=str, default="EVA02-L-14", help="The model name."
    )
    parser.add_argument(
        "--cpkt_path",
        type=str,
        default="../pretrain-model/eva02_large_patch14_clip_224.merged2b_s4b_b131k",
        help="The checkpoint path.",
    )
    # dataset args
    parser.add_argument(
        "--train_json_path",
        type=str,
        default="../data/MPDocVQA/train.json",
        help="The path of train dataset.",
    )
    parser.add_argument(
        "--eval_json_path",
        type=str,
        default="../data/MPDocVQA/val.json",
        help="The path of eval dataset.",
    )
    parser.add_argument(
        "--test_json_path",
        type=str,
        default="../data/MPDocVQA/test.json",
        help="The path of test dataset.",
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        default="../data/MPDocVQA/images",
        help="The path of images.",
    )
    # dataloader args
    parser.add_argument(
        "--pin_memory",
        type=bool,
        default=False,
        action="store_true",
        help="Whether to pin memory.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="The number of workers.",
    )
    parser.add_argument(
        "--dataloader_drop_last",
        type=bool,
        default=False,
        action="store_true",
        help="Whether to drop last batch.",
    )
    # report to
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        choices=["tensorboard", "wandb"],
        help="The report strategy to use.",
    )
    args = parser.parse_args()
    return args
