from typing import List
import json
from pathlib import Path
from loguru import logger
import random

from PIL import Image
import torch
from torch.utils.data import Dataset


def _load_raw_data(json_path: str) -> dict:
    logger.info(f"Loading data from {json_path}...")
    with open(json_path, "r") as f:
        data = json.load(f)
    data = data["data"]
    return data


class MPDocVQADatasetForCLIP(Dataset):
    def __init__(
        self, json_path: str, image_dir: str, tokenizer, transform, split: str = "train"
    ):
        """
        tokenizer: "open_clip.tokenizer.HFTokenizer"
        transform: is used to preprocess image
        split: 决定了如何返回数据
        train or eval: 返回text和对应正确的image
        test: 返回text和所有的image
        """
        super().__init__()
        self.json_path = json_path
        self.split = split
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.tokenizer = tokenizer
        assert self.split in ["train", "eval", "test"]
        self.data = _load_raw_data(self.json_path)  # load json data

    def __len__(self):
        return len(self.data)

    def _get_train_data(self, item) -> dict:
        question: str = item["question"]
        # answers: List[str] = item["answers"]
        page_ids: List[str] = item["page_ids"]
        answer_page_idx: int = item["answer_page_idx"]
        answer_page = page_ids[answer_page_idx]
        image_path = self.image_dir / f"{answer_page}.jpg"
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        question = self.tokenizer([question])
        return {
            "text": question,
            "image": image,
        }

    def _get_test_data(self, item) -> dict:
        question_id = item["questionId"]
        question: str = item["question"]
        images = []
        page_ids: List[str] = item["page_ids"]
        answer_page_idx: int = item["answer_page_idx"]
        for page_id in page_ids:
            image_path = self.image_dir / f"{page_id}.jpg"
            image = Image.open(image_path).convert("RGB")
            image = self.transform(image)
            images.append(image)

        question = self.tokenizer([question])
        image = torch.stack(images, dim=0)
        return {
            "questionId": question_id,
            "text": question,
            "image": image,
            "answer_page_idx": answer_page_idx,
        }

    def __getitem__(self, index) -> dict:
        """
        return text and image
        目前仅使用于训练集验证集
        """
        item: dict = self.data[index]
        if self.split == "train" or self.split == "eval":
            return self._get_train_data(item)
        elif self.split == "test":
            return self._get_test_data(item)
        else:
            raise ValueError(f"split {self.split} is not supported")


class CollatorForCLIP(object):
    def __init__(self, split="train") -> None:
        self.split = split

    def handle_train_batch(self, batch):
        bsz = len(batch)
        keys = batch[0].keys()
        result = {}
        for key in keys:
            result[key] = [item[key] for item in batch]
        for k, v in result.items():
            if k == "text" and isinstance(v[0], torch.Tensor):
                result[k] = torch.stack(v, dim=0)
                result[k] = result[k].reshape(
                    bsz, result[k].size()[-1])  # [k, len]
            elif k == "image" and isinstance(v[0], torch.Tensor):
                result[k] = torch.stack(v, dim=0)  # [k, 3, width,height]
                result[k] = result[k].reshape(
                    bsz, *result[k].size()[-3:]
                )  # [k, 3, width,height]
        return result

    def handle_predict_batch(self, batch):
        raise ValueError("please use dataset directly, one by one predict.")

    def __call__(self, batch):
        if self.split == "train":
            return self.handle_train_batch(batch)
        elif self.split == "predict":
            return self.handle_predict_batch(batch)


class TripletLossDataset(Dataset):
    def __init__(self, json_path: str, image_dir: str, tokenizer, transform):
        self.json_path = json_path
        self.transform = transform
        self.image_dir = Path(image_dir)
        self.tokenizer = tokenizer
        self.raw_data = _load_raw_data(json_path)

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, idx):
        item = self.raw_data[idx]
        question: str = item["question"]
        page_ids: List[str] = item["page_ids"]
        answer_page_idx: int = item["answer_page_idx"]
        ground_truth_page_id: str = page_ids[answer_page_idx]
        # 该条数据存在负例,则可以直接采样
        if len(page_ids) > 1:
            page_ids.pop(answer_page_idx)
            negitive_page_id: str = random.choice(page_ids)
        else:
            # 否则,从全部数据中随机采样一条
            idx_list = list(range(len(self.raw_data)))
            idx_list.pop(idx)
            negitive_idx = random.choice(idx_list)
            negitive_page_id:str = self.raw_data[negitive_idx]["page_ids"][self.raw_data[negitive_idx]["answer_page_idx"]]
            # logger.debug(f"len(page_ids) = 1, random choice from all data, negitive_page_id: {negitive_page_id}")
        positivate_image_path = self.image_dir / f"{ground_truth_page_id}.jpg"
        negitive_image_path = self.image_dir / f"{negitive_page_id}.jpg"
        positivate_image = Image.open(positivate_image_path).convert("RGB")
        negitive_image = Image.open(negitive_image_path).convert("RGB")
        positivate_image = self.transform(positivate_image)
        negitive_image = self.transform(negitive_image)
        question = self.tokenizer([question]).view(-1)
        return dict(
            text=question,
            positivate_image=positivate_image,
            negitive_image=negitive_image
        )

def collate_fn_for_merge_dict(batch):
    keys = batch[0].keys()
    result = {}
    for key in keys:
        result[key] = [item[key] for item in batch]
    for k, v in result.items():
        result[k] = torch.stack(v,dim=0)
    return result

def test_TripletLossDataset():
    from eva02_clip import load_model_tokenizer_transform

    model, transform_train, _, tokenizer = load_model_tokenizer_transform(
        model_name="EVA02-L-14",
        cpkt_path="/home/klwang/pretrain-model/eva02_large_patch14_clip_224.merged2b_s4b_b131k",
        precision="fp16",
        device="cuda:0"
    )

    dataset = TripletLossDataset(
        json_path="/home/klwang/data/MPDocVQA/val.json",
        image_dir="/home/klwang/data/MPDocVQA/images",
        tokenizer=tokenizer,
        transform=transform_train
    )
    def print_data(data):
        print(data["text"].shape)
        print(data["positivate_image"].shape)
        print(data["negitive_image"].shape)
    
    print_data(dataset[0])
    print_data(dataset[20])
    from torch.utils.data import DataLoader
    dataloader = DataLoader(dataset, batch_size=4, collate_fn=collate_fn_for_merge_dict)
    for idx, batch in enumerate(dataloader):
        if idx >= 1:
            break
        print_data(batch)

if __name__ == "__main__":
    test_TripletLossDataset()
