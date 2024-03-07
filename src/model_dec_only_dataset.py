"""
    数据集
"""
import os
from typing import List, Dict
import torch
from torch.utils.data import Dataset
from loguru import logger
import json
import random
from pathlib import Path
from transformers import PreTrainedTokenizer
from transformers.trainer_pt_utils import LabelSmoother

from qwen_vl import QWenTokenizer

IGNORE_TOKEN_ID = LabelSmoother.ignore_index

"""
    用于model_dec_only.py的数据据
    利用clip模型对图像数据进行了过滤
    
    生成的数据类型
    1. input_ids
    2. targets (labels)
    3. attention_mask

    
    raw_data
     {
            "questionId": 49153,
            "question": "What is the ‘actual’ value per 1000, during the year 1975?",
            "doc_id": "pybv0228",
            "page_ids": [
                "pybv0228_p80"
            ],
            "answers": [
                "0.28"
            ],
            "answer_page_idx": 0,
            "data_split": "val",
            "filter_idx": [
                0
            ]
    },

    preprocess

    input_id:
    
    <im_start>system\n
    You are a helpful assistant.\n
    Give you some documnet pictures, you can answer the question according to these document pictures.<im_end>\n
    <im_start>user\n
    Document Picture 1: <img>/root/autodl-tmp/data/images/pybv0228_p80.jpg</img>\n
    Document Picture 2: <img>/root/autodl-tmp/data/images/pybv0228_p80.jpg</img>\n
    What is the ‘actual’ value per 1000, during the year 1975?<im_end>\n
    <|im_start|>assistant\n
    0.28<im_end>\n
    
    target:
    
    <im_start> IGNORE_IDX(system\n + system_message) <im_end>\n
    <im_start> IGNORE_IDX(user\n + prompt) <im_end>\n
    <im_start> IGNORE_IDX(assistnat\n) ANSWER <im_end>\n
"""

system_message = "You are a helpful assistant. Give you some documnet pictures, you can answer the question according to these document pictures."

def preprocess(
        data: List[dict],
        tokenizer: PreTrainedTokenizer,
        max_len: int,
        image_dir: Path,
        system_message: str = system_message
) -> Dict:
    roles = {"user": "<|im_start|>user", "assistant": "<|im_start|>assistant"}

    im_start = tokenizer.im_start_id
    im_end = tokenizer.im_end_id

    nl_token = tokenizer("\n").input_ids
    _system = tokenizer("system").input_ids + nl_token
    _user = tokenizer("user").input_ids + nl_token
    _assistant = tokenizer("assistant").input_ids + nl_token

    def prepare_question(question: str, page_ids: List[str]):
        prompt = ""
        for idx, image_name in enumerate(page_ids):
            image_path = os.path.join(image_dir, f"{image_name}.jpg")
            prompt += f"Document Picture {idx+1}: <img>{image_path}</img>\n"
        prompt = f"Question: {question}"
        return prompt

    input_ids, targets, page_idx_labels = [], [], []
    for i, item in enumerate(data):
        input_id, target = [], []

        # system message
        system = [im_start] + _system + \
            tokenizer(system_message).input_ids + [im_end] + nl_token
        input_id += system
        target += [im_start] + [IGNORE_TOKEN_ID] * \
            (len(system) - 3) + [im_end] + nl_token
        assert len(input_id) == len(target)

        # user message
        question = item["question"]
        # page_ids = item["page_ids"]
        page_ids = item["filter_page_ids"]
        answer = random.choice(item["answers"])
        answer_page_idx = item["answer_page_idx"]
        _input_id = tokenizer(roles["user"]).input_ids + nl_token \
            + tokenizer(prepare_question(question, page_ids)).input_ids + \
            [im_end] + nl_token
        _target = [im_start] + [IGNORE_TOKEN_ID] * \
            (len(_input_id) - 3) + [im_end] + nl_token
        input_id += _input_id
        target += _target
        assert (len(input_id) == len(target))
        # assistant message
        _input_id = tokenizer(roles["assistant"]).input_ids + nl_token \
            + tokenizer(answer).input_ids + [im_end] + nl_token
        # [IGNORE_TOKEN_ID] * (len(tokenizer(roles["assistant"]).input_ids)-1+1)
        _target = [im_start] + [IGNORE_TOKEN_ID] * len(tokenizer(roles["assistant"]).input_ids) + _input_id[len(
            tokenizer(roles["assistant"]).input_ids) + 1:-2] + [im_end] + nl_token
        input_id += _input_id
        target += _target
        assert len(input_id) == len(target)
        input_id = input_id + [tokenizer.pad_token_id] * \
            (max_len - len(input_id))
        target = target + [IGNORE_TOKEN_ID] * (max_len - len(target))

        input_ids.append(input_id[:max_len])
        targets.append(target[:max_len])
        page_idx_labels.append(answer_page_idx)

    input_ids = torch.tensor(input_ids, dtype=torch.int)
    labels = torch.tensor(targets, dtype=torch.int)
    page_idx_labels = torch.tensor(
        page_idx_labels, dtype=torch.int).view(-1, 1)
    return dict(
        input_ids=input_ids,
        labels=labels,
        attention_mask=input_ids.ne(tokenizer.pad_token_id),
        page_idx_labels=page_idx_labels
    )


class LazyMPDocVQADataset(Dataset):
    def __init__(self,
                 raw_data: List[dict],
                 tokenizer: PreTrainedTokenizer,
                 max_len: int,
                 image_dir: Path,
                 system_message: str = "You are a helpful assistant.\nGive you some documnet pictures, you can answer the question according to these pictures."):

        self.tokenizer = tokenizer
        self.max_len = max_len
        self.image_dir = image_dir
        self.system_message = system_message
        logger.info("Formatting inputs...Skip in lazy mode")
        self.raw_data = raw_data
        self.tokenizer = tokenizer
        self.cached_data_dict = {}

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        if i in self.cached_data_dict:
            return self.cached_data_dict[i]
        ret = preprocess([self.raw_data[i]], self.tokenizer,
                         self.max_len, self.image_dir, self.system_message)
        ret = dict(
            input_ids=ret["input_ids"][0],
            labels=ret["labels"][0],
            attention_mask=ret["attention_mask"][0],
            page_idx_labels=ret["page_idx_labels"][0]
        )
        self.cached_data_dict[i] = ret

        return ret


class MPDocVQADataset(Dataset):
    def __init__(
        self,
        json_path: str,
        image_dir: str,
        tokenizer: QWenTokenizer,
        split: str = "train",
    ) -> None:
        super().__init__()
        self.json_path = json_path
        self.image_dir = Path(image_dir)
        self.tokenizer = tokenizer
        self.split = split
        assert self.split in ["train", "val", "test"]
        self.data = self._load_data()

    def _load_data(self):
        logger.info(f"Loading {self.split} data from {self.json_path}")
        with open(self.json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        data = data["data"]
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        question = item["question"]
        question_id = item["questionId"]
        page_ids = item["page_ids"]
        filter_idx = item["filter_idx"]
        select_page_ids = []
        for idx, page_id in enumerate(page_ids):
            if idx in filter_idx:
                select_page_ids.append(page_id)
        prompt = self._prepare_prompt(question, select_page_ids)
        if self.split == "train":
            answers = item["answers"]
            answer = random.choice(answers) + self.tokenizer.eos_token
            answer_page_index = item["answer_page_idx"]
            prompt += answer

        # print(prompt)
        tokenizer_outputs = self.tokenizer(
            prompt, return_tensors="pt", padding="max_length", truncation=True
        )
        input_ids = tokenizer_outputs["input_ids"]
        attention_mask = tokenizer_outputs["attention_mask"]
        token_type_ids = tokenizer_outputs["token_type_ids"]
        # mask labels
        model_inputs = {
            "input_ids": input_ids,  # [bsz, seq_len]
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        }
        if self.split == "train":
            labels = input_ids.masked_fill(attention_mask == 0, -100)
            prompt_outputs = self.tokenizer(
                prompt[: -len(answer)],
                return_tensors="pt",
                padding="do_not_pad",
                truncation=True,
            )
            labels[:, : prompt_outputs["input_ids"].size(1)] = -100
            model_inputs["labels"] = labels
            model_inputs["page_idx_labels"] = torch.tensor(
                [answer_page_index], dtype=torch.long
            )
        return model_inputs

    def _prepare_prompt(self, question, input_image_names):
        prompt = ""
        for idx, image_name in enumerate(input_image_names):
            image_path = self.image_dir / f"{image_name}.jpg"
            prompt += f"Document Picture {idx+1}: <img>{image_path}</img>\n"
        prompt += f"Question: {question}\n"
        prompt += "Answer:"
        return prompt


def collate_fn_for_MPModel(batch):
    """
    batch: List[Dict]
    """
    bsz = len(batch)
    input_ids = torch.stack([item["input_ids"] for item in batch])
    attention_mask = torch.stack([item["attention_mask"] for item in batch])
    labels = torch.stack([item["labels"] for item in batch])
    page_idx_labels = torch.stack([item["page_idx_labels"] for item in batch])
    return {
        "input_ids": input_ids.view(bsz, -1),
        "attention_mask": attention_mask.view(bsz, -1),
        "labels": labels.view(bsz, -1),
        "page_idx_labels": page_idx_labels.view(bsz),
    }


def collate_fn_for_qwen_vl_lora(batch):
    """
    batch: List[Dict]
    """
    bsz = len(batch)
    input_ids = torch.stack([item["input_ids"] for item in batch])
    attention_mask = torch.stack([item["attention_mask"] for item in batch])
    labels = torch.stack([item["labels"] for item in batch])
    ret = {
        "input_ids": input_ids.view(bsz, -1),
        "attention_mask": attention_mask.view(bsz, -1),
        "labels": labels.view(bsz, -1),
    }
    for k, v in ret.items():
        logger.info(f"{k}: {v.size()}")
    return ret


def test_MPDocVQADataset():
    from qwen_vl_chat import QWenLMHeadModel
    from model_dec_only import MPModel, MPModelConfig, load_lora_qwen_vl_model

    pretrained_model_name_or_path = "Qwen/Qwen-VL-Chat"
    tokenizer = QWenTokenizer.from_pretrained(
        pretrained_model_name_or_path,
        cache_dir="/root/autodl-tmp/pretrain-model",
        model_max_length=1024,
        padding_side="right",
    )
    tokenizer.pad_token = "<|endoftext|>"
    tokenizer.eos_token = "<|endoftext|>"

    config = MPModelConfig.from_pretrained(
        pretrained_model_name_or_path,
        cache_dir="/root/autodl-tmp/pretrain-model",
        classification_head_hidden_size=4096,
        num_pages=20,
    )
    # config.bf16 = True
    config.fp16 = True
    config.use_cache = False
    qwen_vl = QWenLMHeadModel.from_pretrained(
        pretrained_model_name_or_path,
        cache_dir="/root/autodl-tmp/pretrain-model",
        config=config,
        device_map="cuda:0",
    )
    qwen_vl_lora = load_lora_qwen_vl_model(
        qwen_vl=qwen_vl, r=1, lora_alpha=32, lora_dropout=0.1
    )
    mp_model = MPModel(config=config, qwen_vl=qwen_vl_lora)
    mp_model.train()
    json_path = "/root/autodl-tmp/data/val_filter.json"
    image_dir = "/root/autodl-tmp/data/images"
    dataset = MPDocVQADataset(
        json_path=json_path,
        image_dir=image_dir,
        tokenizer=tokenizer,
        split="train",
    )
    for idx, item in enumerate(dataset):
        if idx >= 3:
            break
        item = {k: v.to("cuda:0") for k, v in item.items()}
        for k, v in item.items():
            print(k, v)
        ouputs = mp_model(**item, return_dict=True)
        print(ouputs.loss)


if __name__ == "__main__":
    test_MPDocVQADataset()
