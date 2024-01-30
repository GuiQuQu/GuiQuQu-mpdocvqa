"""
    数据集
"""

import torch
from torch.utils.data import Dataset
from loguru import logger
import json
import random
from pathlib import Path

from qwen_vl import QWenTokenizer


"""
    用于model_dec_only.py的数据据
    利用clip模型对图像数据进行了过滤
"""


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
