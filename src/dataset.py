import json
from pathlib import Path
from typing import List, Tuple, Optional
import random
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizer
from loguru import logger
from qwen_vl.qwen_generation_utils import make_context
import open_clip


from utils import _prepare_only_question_prompt

IGNORE_INDEX = -100


class MPDocVQADataset(Dataset):
    def __init__(
        self,
        json_path: str,
        image_dir: Optional[str] = None,
        return_tensor: bool = False,
        prompt_type: Optional[int] = None,
        tensor_dir: Optional[str] = None,
        split: str = "train",
    ):
        """
        json_path: 数据集json文件路径
        image_dir: 图片文件夹路径,在数据集里目前没用到
        split: 数据集划分
        prompt_type: prompt的类型,目前有三种,仅在获取对应的tensor时使用
        """
        self.json_path = json_path
        self.image_dir = Path(image_dir)
        self.split = split
        self.return_tensor = return_tensor
        if self.return_tensor:
            assert (
                tensor_dir is not None and prompt_type is not None
            ), "tensor_dir and prompt_type should not be None in 'return_tensor' mode"
            self.prompt_type = prompt_type
            self.tensor_dir = Path(tensor_dir)
        self.data = self._load_data()  # load json data

        if image_dir is not None:
            self.image_dir = Path(image_dir)

    def _load_data(self):
        logger.info(f"Loading {self.split} data from {self.json_path}")
        with open(self.json_path, "r") as f:
            data = json.load(f)
        data = data["data"]
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index) -> dict:
        """
        返回数据
        """
        item = self.data[index]
        if self.return_tensor:
            question_id = item["questionId"]
            for page_id in item["page_ids"]:
                tensor_name = f"{question_id}@{page_id}@{self.prompt_type}"
                tensor_path = self.tensor_dir / f"{tensor_name}.pth"
                item[tensor_name] = torch.load(tensor_path)
        return item


class MPDocVQADatasetForDecoderOnly2(Dataset):
    """
    这个数据集的__getitem__函数的返回可以直接用于模型输入
    """

    def __init__(
        self,
        json_path: str,
        tokenzier: PreTrainedTokenizer,
        return_tensor: bool = False,
        prompt_type: Optional[int] = None,
        tensor_dir: Optional[str] = None,
        tensor_padding: str = "max_length",
        tensor_padding_length: int = 20,
        pooling: str = "mean",
        split: str = "train",
    ):
        """
        json_path: 数据集json文件路径
        image_dir: 图片文件夹路径,在数据集里目前没用到
        split: 数据集划分
        prompt_type: prompt的类型,目前有三种,仅在获取对应的tensor时使用
        get_tensor: 是否获取tensor
        """
        super().__init__()
        self.json_path = json_path
        self.split = split
        self.return_tensor = return_tensor
        self.tokenzier = tokenzier
        self.pooling = pooling
        if self.return_tensor:
            assert (
                tensor_dir is not None and prompt_type is not None
            ), "tensor_dir and prompt_type should not be None in 'return_tensor' mode"
            self.prompt_type = prompt_type
            self.tensor_dir = Path(tensor_dir)
            self.tensor_padding = tensor_padding
            self.tensor_padding_length = tensor_padding_length

        self.data = self._load_data()  # load json data

    def _load_data(self):
        logger.info(f"Loading {self.split} data from {self.json_path}")
        with open(self.json_path, "r") as f:
            data = json.load(f)
        data = data["data"]
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index) -> dict:
        """
        返回数据
        1. 训练数据编码prompt+answer
        2. 验证数据编码prompt+answer,计算验证集的loss
        3. 推理数据编码prompt
        """
        item = self.data[index]
        prompt = _prepare_only_question_prompt(item["question"])

        # get encoder_attention_states and encoder_attention_mask
        encoder_attention_states = []
        if self.return_tensor:
            endoftext_tensor = []
            question_id = item["questionId"]
            for page_id in item["page_ids"]:
                tensor_name = f"{question_id}@{page_id}@{self.prompt_type}"
                load_tensor = torch.load(self.tensor_dir / f"{tensor_name}.pth")
                assert len(load_tensor.size()) == 2

                endoftext_tensor.append(load_tensor[-1, :].view(1, -1))

                # pooling method
                if self.pooling == "mean":
                    # 计算了全部的均值(包括了end of text)
                    load_tensor = load_tensor.mean(
                        dim=0, keepdim=True
                    )  # [1, hidden_size]
                else:
                    raise ValueError(f"pooling method {self.pooling} is not supported")

                encoder_attention_states.append(load_tensor)

            endoftext_tensor = torch.cat(endoftext_tensor, dim=0).mean(
                dim=0, keepdim=True
            )
            # 这里可以自己封装一下,封装成针对batch也适用的情景
            encoder_attention_states = torch.cat(
                encoder_attention_states, dim=0
            ).unsqueeze(
                0
            )  # [1,num_page, hidden_size]
            encoder_attention_mask = torch.zeros(
                1, self.tensor_padding_length, dtype=torch.long
            )
            seq_len = encoder_attention_states.size(1)
            encoder_attention_mask[0, :seq_len] = 1
            # add padding embedding
            padding_embedding = torch.cat(
                [endoftext_tensor] * (self.tensor_padding_length - seq_len), dim=0
            ).unsqueeze(
                0
            )  # [1,padding_length, hidden_size]
            encoder_attention_states = torch.cat(
                [encoder_attention_states, padding_embedding], dim=1
            )
        else:
            encoder_attention_states = None
            encoder_attention_mask = None

        # get input_ids, attention_mask, token_type_ids, labels
        if self.split == "test" or self.split == "val":
            tokenizer_outputs = self.tokenzier(
                prompt, return_tensors="pt", padding="max_length", truncation=True
            )
            input_ids = tokenizer_outputs["input_ids"]
            attention_mask = tokenizer_outputs["attention_mask"]
            token_type_ids = tokenizer_outputs["token_type_ids"]
            if self.split == "val":
                labels = {
                    "answers": item["answers"],
                    "answer_page_idx": item["answer_page_idx"],
                }
            else:
                labels = None
        else:
            # 训练集和验证集,需要准备训练时的模型输入
            # answers里面答案很多,目前固定选择第一个答案作为正确答案
            answer = item["answers"][0] + self.tokenizer.eos_token
            tokenizer_outputs = self.tokenzier(
                prompt + answer,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
            )

            input_ids = tokenizer_outputs["input_ids"]
            attention_mask = tokenizer_outputs["attention_mask"]
            token_type_ids = tokenizer_outputs["token_type_ids"]
            # -100 是 pytorch cross_entropy loss 的 ignore_index
            labels = input_ids.masked_fill(attention_mask == 0, IGNORE_INDEX)
            prompt_outputs = self.tokenzier(
                prompt, return_tensors="pt", padding="do_not_pad", truncation=True
            )
            # mask prompt part
            labels[:, : prompt_outputs["input_ids"].size(1)] = IGNORE_INDEX

        model_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "encoder_attention_states": encoder_attention_states,
            "encoder_attention_mask": encoder_attention_mask,
            "labels": labels,
        }
        return model_inputs


class MPDocVQADatasetForDecoderOnly(Dataset):
    def __init__(self, json_path: str, feature_dir: str, split: str = "train"):
        super().__init__()
        self.json_path = Path(json_path)
        self.feature_dir = Path(feature_dir)
        self.split = split
        self.data = self._load_data()  # load json data

    def _load_data(self):
        logger.info(f"Loading {self.split} data from {self.json_path}")
        with open(self.json_path, "r") as f:
            data = json.load(f)
        data = data["data"]
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index) -> dict:
        item = self.data[index]
        # get features
        features_list = []
        question_id = item["questionId"]
        for page_id in item["page_ids"]:
            tensor_name = f"{question_id}@{page_id}"
            tensor = torch.load(self.tensor_dir / f"{tensor_name}.pth")
            assert len(tensor.size()) == 2
            features_list.append(tensor)
        encoder_embeds = torch.cat(features_list, dim=0)  # [num_page, hidden_size]
        item["encoder_embeds"] = encoder_embeds
        return item


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
        self.data = self._load_data()  # load json data

    def _load_data(self):
        logger.info(f"Loading {self.split} data from {self.json_path}")
        with open(self.json_path, "r") as f:
            data = json.load(f)
        data = data["data"]
        return data

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


class CollatorForDecoderOnly(object):
    def __init__(
        self, tokenizer: PreTrainedTokenizer, embedding_layer: torch.nn.Embedding
    ):
        self.tokenizer = tokenizer
        self.max_seq_length = self.tokenizer.model_max_length
        self.embedding_layer = embedding_layer

    def __call__(self, batch):
        """
        get input_embeds, input_ids(for labels)
        """
        pass


class MPDocVQADatasetForEncoderDecoderModel(Dataset):
    def __init__(self, json_path: str, image_dir: str, split: str = "train"):
        self.json_path = json_path
        if image_dir is not None:
            self.image_dir = Path(image_dir)
        self.split = split
        self.data = self._load_data()  # load json data

    def _load_data(self):
        logger.info(f"Loading {self.split} data from {self.json_path}")
        with open(self.json_path, "r") as f:
            data = json.load(f)
        data = data["data"]
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        question = item["question"]
        question_id = item["questionId"]
        is_train = "answers" in item
        if is_train:
            answers = [a.lower() for a in item["answers"]]
            answer_page_idx = item["answer_page_idx"]
        images = []
        for page_id in item["page_ids"]:
            image_path = str(self.image_dir / f"{page_id}.jpg")
            images.append(image_path)
        result = {
            "question": question,
            "question_id": question_id,
            "images": images,
        }
        if is_train:
            result["answers"] = answers
            result["answer_page_idx"] = answer_page_idx
        return result


class CollatorForEncoderDecoderModel(object):
    def __init__(self, tokenizer: PreTrainedTokenizer):
        self.tokenizer = tokenizer
        self.max_seq_length = self.tokenizer.model_max_length

    def __call__(self, batch):
        # [bsz, num_page,seq_len]
        num_pages = [len(item["images"]) for item in batch]
        # question_batch = [item["question"] for item in batch]
        is_train = "answers" in batch[0]
        if is_train:
            answers_batch = [item["answers"] for item in batch]
            answer_page_idx_batch = [item["answer_page_idx"] for item in batch]
        bsz = len(num_pages)

        all_input_ids = torch.zeros(
            bsz, max(num_pages), self.max_seq_length, dtype=torch.long
        )
        all_attention_masks = torch.zeros(
            bsz, max(num_pages), self.max_seq_length, dtype=torch.long
        )
        all_image_masks = torch.zeros(bsz, max(num_pages), dtype=torch.long)

        def cat_qa_image(question: str, image_path: str):
            return f"Document Picture: <img>{image_path}</img>\nQuestion:{question}\nAnswer:"

        for idx, item in enumerate(batch):
            question = item["question"]
            answers = item["answers"] if "answers" in item else None
            text_messages = []
            for image_path in item["images"]:
                text_messages.append(cat_qa_image(question, image_path))
                cur_num_pages = len(text_messages)
                text_messages_tokens = (
                    self.tokenizer.batch_encode_plus(  # [cur_num_page,max_seq_length]
                        text_messages,
                        return_tensors="pt",
                        padding="max_length",
                        truncation=True,
                    )
                )
                all_input_ids[idx, :cur_num_pages, :] = text_messages_tokens[
                    "input_ids"
                ]
                all_attention_masks[idx, :cur_num_pages, :] = text_messages_tokens[
                    "attention_mask"
                ]
                all_image_masks[idx, :cur_num_pages] = 1
        result = {
            "input_ids": all_input_ids,  # [bsz, num_page, seq_len],
            "attention_mask": all_attention_masks,  # [bsz, num_page, seq_len],
            "page_mask": all_image_masks,  # [bsz, num_page]
        }
        if is_train:
            # Answer size is [bsz,cur_answer_len]
            # 因为不明确前面的context的长度,因此在具体使用时应该使用该数据重新构造数据,加入合适的padding
            answer_batch = [random.choice(answers) for answers in answers_batch]
            # print("answer text", answer_batch)
            answer_batch_tokens = self.tokenizer.batch_encode_plus(
                answer_batch,
                padding=False,
                truncation=True,
            )

            # is a list , len(it)=bsz, every elem is corresponding answer input_ids
            result["labels"] = answer_batch_tokens["input_ids"]
            # print("labels", result["labels"])
            result["answer_page_index"] = torch.tensor(
                answer_page_idx_batch, dtype=torch.long
            )
        return result


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
                result[k] = result[k].reshape(bsz, result[k].size()[-1])  # [k, len]
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


"""
    利用clip模型过滤图像
"""




def _prepare_input_ids(
    tokenizer: PreTrainedTokenizer, device: str, prompt: str
) -> torch.Tensor:
    """将文字转为input_ids

    item: Dict,数据
    device: 将数据放入的设备(cpu or cuda)
    tokenizer: PreTrainedTokenizer
    """
    raw_context, context_tokens = make_context(
        tokenizer,
        prompt,
        history=[],
        system="",
        max_window_size=6144,
        chat_format="raw",
    )

    # input_ids = tokenizer.batch_encode_plus([prompt], padding="max_length", return_tensors="pt")["input_ids"]
    # input_ids = input_ids.to(device)
    input_ids = torch.tensor([context_tokens], dtype=torch.long).to(device)
    return input_ids


def test_MPDocVQADatasetForEncoderDecoderModel():
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        "../pretrain-model/QWen-VL", trust_remote_code=True, model_max_length=512
    )
    tokenizer.pad_token = "<|endoftext|>"
    tokenizer.eos_token = "<|endoftext|>"
    train_dataset = MPDocVQADatasetForEncoderDecoderModel(
        json_path="../data/MPDocVQA/train.json",
        image_dir="../data/MPDocVQA/images",
        split="train",
    )
    collator = CollatorForEncoderDecoderModel(tokenizer)
    train_dataloader = DataLoader(
        train_dataset, batch_size=4, collate_fn=collator, shuffle=False
    )
    for idx, batch in enumerate(train_dataloader):
        if idx >= 1:
            break
        for k, v in batch.items():
            print(k, end=":")
            if isinstance(v, torch.Tensor):
                print(v.size())
            else:
                print(v)
            if k == "page_mask":
                print("page_mask", v)
            if k == "input_ids":
                for t in v:
                    print(t[:, :64])
        print()
    for idx, item in enumerate(train_dataset):
        if idx >= 1:
            break
        print(json.dumps(item, indent=4, ensure_ascii=False))


def test_MPDocVQADatasetForCLIP():
    from eva02_clip import load_model_tokenizer_transform

    model, transform_train, _, tokenizer = load_model_tokenizer_transform(
        model_name="EVA02-L-14",
        cpkt_path="../pretrain-model/eva02_large_patch14_clip_224.merged2b_s4b_b131k",
        precision="fp16",
        device="cuda:0",
    )
    dataset = MPDocVQADatasetForCLIP(
        json_path="../data/MPDocVQA/val.json",
        image_dir="../data/MPDocVQA/images",
        tokenizer=tokenizer,
        transform=transform_train,
        split="train",
    )
    for idx, item in enumerate(dataset):
        if idx >= 1:
            break
        print("text=>", item["text"].size())  # torch.Size([1, 77])
        print("image=>", item["image"].size())  # torch.Size([3, 224, 224])

    data_loader = DataLoader(
        dataset, batch_size=4, shuffle=False, collate_fn=CollatorForCLIP()
    )

    for idx, batch in enumerate(data_loader):
        if idx >= 1:
            break
        print("text=>", batch["text"].size())  # torch.Size([4, 1, 77])
        print("image=>", batch["image"].size())  # torch.Size([4, 3, 224, 224])


if __name__ == "__main__":
    test_MPDocVQADatasetForCLIP()
