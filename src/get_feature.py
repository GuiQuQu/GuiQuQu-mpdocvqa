"""
    读取json文件,利用qwen-vl模型获取图像+文字的特征
"""
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from typing import Tuple, List
import json
import argparse
from tqdm import tqdm
from pathlib import Path
import datetime
import random

import numpy as np
import torch
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
from transformers import PreTrainedModel, PreTrainedTokenizer

from dataset import MPDocVQADataset, _prepare_input_ids
from qwen_vl.qwen_generation_utils import decode_tokens
from utils import _prepare_prompt1, _prepare_prompt2, _prepare_prompt3

DEFAULT_OUTPUT_DIR = "../data/MPDocVQA/2023-12-17-features"
DEFAULT_LOG_FILE = "../log/get_feature.log"
DEFAULT_LOG_DIR = "../log"

DEFFAULT_TRAIN_JSON_PATH = "../data/MPDocVQA/train.json"
DEFFAULT_VAL_JSON_PATH = "../data/MPDocVQA/val.json"
DEFFAULT_TEST_JSON_PATH = "../data/MPDocVQA/test.json"
DEFAULT_IMAGE_DIR = "../data/MPDocVQA/images"

DEFAULT_QWEN_VL_MODEL_PATH = "../pretrain-model/QWen-VL"


def seed_everything(seed: int):
    """
    设置随机种子
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def _get_args():
    parser = argparse.ArgumentParser(description="Get feature")
    parser.add_argument(
        "--input-json",
        type=str,
        default=DEFFAULT_TRAIN_JSON_PATH,
        help="input json file",
    )
    parser.add_argument(
        "--image-dir",
        type=str,
        default=DEFAULT_IMAGE_DIR,
        help="image dir",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "val", "test"],
        default="val",
        help="split, train/val/test",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="output jsonl file",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default=DEFAULT_QWEN_VL_MODEL_PATH,
        help="checkpoint path",
    )
    # parser.add_argument(
    #     "--batch-size",
    #     type=int,
    #     default=1,
    #     help="batch size",
    # )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="cache dir",
    )
    parser.add_argument(
        "--cpu-only",
        action="store_true",
        default=False,
        help="cpu only",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="verbose",
    )

    parser.add_argument(
        "--prompt-type",
        type=int,
        choices=[1, 2, 3],
        default=3,
        help="prompt type",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2023,
        help="seed",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="debug mode",
    )
    args = parser.parse_args()
    return args


def _load_model_tokenizer(args) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """
    加载模型和tokenizer
    """
    tokenizer = AutoTokenizer.from_pretrained(
        args.checkpoint_path,
        trust_remote_code=True,
        cache_dir=args.cache_dir,
        resume_download=True,
    )
    tokenizer.pad_token = "<|endoftext|>"
    tokenizer.eos_token = "<|endoftext|>"
    tokenizer.model_max_length = 1024
    if args.cpu_only:
        args.device = "cpu"
    else:
        args.device = "cuda"

    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint_path,
        device_map=args.device,
        trust_remote_code=True,
        resume_download=True,
        bf16=True,
        cache_dir=args.cache_dir if args.cache_dir else None,
    ).eval()

    model.generation_config = GenerationConfig.from_pretrained(
        args.checkpoint_path,
        trust_remote_code=True,
        resume_download=True,
    )

    return model, tokenizer


def qwen_vl_generate_features(
    args,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompt: str,
    ground_truth_answers: str = None,
    verbose: bool = False,
    return_responses: bool = True,
    endoftext_token: str = "<|endoftext|>",
) -> torch.Tensor | Tuple[torch.Tensor, str]:
    """
    outputs.sequences: (batch_size*num_return_sequences, sequence_length))
    outputs.hidden_states:(tuple(tuple(torch.FloatTensor))
    tuple1 for token
    tuple2 for layer
    torch.FloatTensor of shape (num_return_sequences*batch_size, generated_length, hidden_size).
    输出时tuple1中第一个元素是prompt对应的特征,一般长度都不是1
    """
    input_ids = _prepare_input_ids(tokenizer, args.device, prompt)
    outputs = model.generate(
        input_ids,
        return_dict_in_generate=True,
        output_hidden_states=True,
        max_new_tokens=128,
        generation_config=model.generation_config,
    )

    response = decode_tokens(
        tokens=outputs.sequences.squeeze(0),
        tokenizer=tokenizer,
        raw_text_len=len(prompt),
        context_length=len(input_ids),
        chat_format="raw",
        verbose=False,
    )
    trim_decoded_tokens = tokenizer.decode(
        outputs.sequences.squeeze(0), errors="replace"
    )[len(prompt) :]
    if verbose:
        logger.info("")
        # logger.info("input_ids: {}".format(input_ids))
        logger.info(f"prompt: {prompt}".replace("\n", "\\n"))
        logger.info(f"output.sequences.size(): {outputs.sequences.size()}")

        logger.info("trim_decoded_tokens: {}".format(trim_decoded_tokens))
        trim_tokens_ids = outputs.sequences.squeeze(0)[len(tokenizer.encode(prompt)) :]
        logger.info(
            f"trim_tokens_ids: {trim_tokens_ids.detach().cpu().numpy().tolist()}"
        )
        logger.info("response: {}".format(response))
        # 对应的token化之后的长度
        token_list = tokenizer.tokenize(trim_decoded_tokens)
        logger.info("token list: {}, len={}".format(token_list, len(token_list)))
        if args.split != "test":
            logger.info(f"ground truth answers: {ground_truth_answers}")

    last_hidden_states = []
    seq_length = 0
    if verbose:
        logger.info(
            f"outputs.hidden_states tuple1(token) num: {len(outputs.hidden_states)}"
        )
    for idx, token_hidden_states in enumerate(outputs.hidden_states):
        # target token_hidden_states[-1] size: (1, seq_length, hidden_size)
        feature_size = token_hidden_states[-1].size()
        seq_length += feature_size[1]
        # 采用prompt的最后一个序列代表第一个预测词
        if idx == 0 and feature_size[1] != 1:  # prompt
            if verbose:
                logger.info(f"detect prompt, prompt feature_size: {feature_size}")
            last_hidden_states.append(token_hidden_states[-1][:, -1, :].view(1, 1, -1))
        else:
            if verbose:
                # 预测得到'<|endoftext|>'的feature也进行了保存
                if (
                    outputs.sequences.squeeze(0)[seq_length] == tokenizer.eod_id
                ):  # 令人费解,居然不越界
                    logger.info(
                        f"detect endoftext_token_id, feature_size: {token_hidden_states[-1].size()}"
                    )
                else:
                    logger.info(
                        f"last_hidden_states.size(): {last_hidden_states[-1].size()}"
                    )
            last_hidden_states.append(token_hidden_states[-1])

    if len(last_hidden_states) == 0:
        logger.error("response is empty, last_hidden_states is empty!")
        return None, response
    else:
        last_hidden_states = torch.squeeze(torch.stack(last_hidden_states, dim=0))
        if len(last_hidden_states.size()) == 1:
            last_hidden_states = last_hidden_states.unsqueeze(0)
        if verbose:
            logger.info(f"last_hidden_states.size(): {last_hidden_states.size()}")
        return last_hidden_states, response


def add_log_file(args):
    log_dir = Path(DEFAULT_LOG_DIR)
    if not log_dir.exists():
        log_dir.mkdir(parents=True, exist_ok=True)
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    log_file_path = log_dir / f"get_feature@{args.split}@{args.prompt_type}@{now}.log"
    logger.add(log_file_path, rotation="500 MB")


def log_args(args):
    for k, v in sorted(vars(args).items()):
        logger.info(f"{k:<15} = {v}")


def save_padding_feature(args, model: PreTrainedModel, tokenizer: PreTrainedTokenizer):
    "选择最后一个feature作为padding feature"
    pass


def main(args):
    add_log_file(args)
    log_args(args)
    output_dir = Path(args.output_dir) / args.split / f"@{args.prompt_type}"
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = MPDocVQADataset(
        args.input_json,
        args.image_dir,
        split=args.split,
        return_tensor=False,
    )

    model, tokenizer = _load_model_tokenizer(args)

    for idx, item in enumerate(dataset):
        if args.debug:
            if idx >= 10:
                break
        question = item["question"]
        for _, page_id in enumerate(item["page_ids"]):
            image_path = args.image_dir + "/" + page_id + ".jpg"
            prompt = _prepare_prompt(image_path, question, _type=args.prompt_type)
            ground_truth_answers = (
                f"{item['answers']}" if "answers" in item.keys() else None
            )
            last_hidden_states, response = qwen_vl_generate_features(
                args,
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                ground_truth_answers=ground_truth_answers,
                return_responses=True,
                verbose=args.verbose,
            )
            # save to pth
            pth_name = f"{item['questionId']}@{page_id}@{args.prompt_type}.pth"
            if last_hidden_states is None:
                tensor_info = f"last_hidden_states is None, name: {pth_name}"
                logger.error(f"{tensor_info:<60}response: {response}")
            else:
                tensor_info = f"{pth_name:<25}=> {last_hidden_states.size()}"
                logger.info(
                    f"{tensor_info:<60}response: {response:<12} ground_truth_answers:{ground_truth_answers}"
                )
                save_path = output_dir / pth_name
                torch.save(last_hidden_states, save_path)


def _prepare_prompt(image_path: str, question: str, _type: int = 1) -> str:
    if _type == 1:
        return _prepare_prompt1(image_path, question)
    elif _type == 2:
        return _prepare_prompt2(image_path, question)
    elif _type == 3:
        return _prepare_prompt3(image_path, question)
    else:
        raise NotImplementedError


def print_named_parameters(model):
    for name, p in model.named_parameters():
        print(name, p.size())


if __name__ == "__main__":
    args = _get_args()
    seed_everything(args.seed)
    main(args)
