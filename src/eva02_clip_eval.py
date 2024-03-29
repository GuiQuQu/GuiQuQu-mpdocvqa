from tqdm import tqdm
from pathlib import Path
from loguru import logger
import datetime
import json

import torch

from eva02_clip import load_model_tokenizer_transform
from dataset import MPDocVQADatasetForCLIP, CollatorForCLIP

CPKT_NAME = "checkpoint-15862"
CPKT_PATH = f"../clip-outputs/2024-01-16/{CPKT_NAME}"

MODEL_NAME = "EVA02-L-14"
TOKENIZER_PATH = "/home/wang-ke-long/pretrain-model/eva02_large_patch14_clip_224.merged2b_s4b_b131k"
DEFAULT_LOG_DIR = "../log"

DATASET_PATH = f"/home/wang-ke-long/data/MPDocVQA"
# 过滤train or val or test数据集
MODE = "val"


def get_cast_dtype(model) -> torch.dtype:
    if isinstance(model, torch.nn.Module):
        return next(model.parameters()).dtype
    else:
        return None


def get_cast_device(model) -> torch.device:
    if isinstance(model, torch.nn.Module):
        return next(model.parameters()).device
    else:
        return None

def eval():
    """
        评测模型检索结果
    """
    model, _, preprocess_eval, tokenizer = load_model_tokenizer_transform(
        model_name="EVA02-L-14",
        tokenizer_path=TOKENIZER_PATH,
        cpkt_path=CPKT_PATH,
        file_name="pytorch_model.bin",
        precision="fp16",
        device="cuda:0",
    )
    model.eval()
    eval_dataset = MPDocVQADatasetForCLIP(
        json_path=f"{DATASET_PATH}/val.json",
        image_dir=f"{DATASET_PATH}/images",
        tokenizer=tokenizer,
        transform=preprocess_eval,
        split="test",
    )
    hit_count = 0
    topk_hit_count = 0
    gt_3_count = 0
    total_count = len(eval_dataset)
    for idx, item in enumerate(tqdm(eval_dataset)):
        image = item["image"]  # [k,3,224,224]
        text = item["text"]  # [1, 77]
        image = image.to(get_cast_device(model)).to(get_cast_dtype(model))
        text = text.to(get_cast_device(model)).long()
        answer_page_idx = item["answer_page_idx"]
        image_features = model.encode_image(image)  # [k, hidden_size]
        text_features = model.encode_text(text)  # [1, hidden_size]

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        image_probs = (
            (100.0 * text_features @ image_features.T).softmax(dim=-1).view(-1)
        )
        if image_probs.size(0) >= 3:
            _, topk_idx = torch.topk(image_probs, k=3)
            gt_3_count += 1
            if answer_page_idx in topk_idx:
                hit_count += 1
                topk_hit_count += 1
            # tqdm.write(str(answer_page_idx) + "<=>" + str(topk_idx))
        else:
            hit_count += 1
            # tqdm.write("not enough 3,"+str(answer_page_idx) + "<=>" + str(image_probs.data))
    print(
        f"hit count: {hit_count}, top-k-hit count: {topk_hit_count}, total count: {total_count}"
    )
    print(f"hit rate {hit_count / total_count * 100:.4f}%")
    print(f"top-k-hit rate (topk_hit_count/gt_3_count): {topk_hit_count /gt_3_count * 100:.4f}%")


# 进行图像文件过滤
def add_log_file():
    log_dir = Path(DEFAULT_LOG_DIR)
    if not log_dir.exists():
        log_dir.mkdir(parents=True, exist_ok=True)
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    log_file_path = log_dir / f"{MODE}-{CPKT_NAME}-{now}.log"
    logger.add(log_file_path, rotation="500 MB")

def filter_image_by_clip(verbose=False):
    model, _, preprocess_eval, tokenizer = load_model_tokenizer_transform(
        model_name=MODEL_NAME,
        tokenizer_path=TOKENIZER_PATH,
        cpkt_path=CPKT_PATH,
        file_name="pytorch_model.bin",
        precision="fp16",
        device="cuda:0",
    )
    model.eval()
    dataset_path = f"{DATASET_PATH}/{MODE}.json"
    image_dir = f"{DATASET_PATH}/images"
    filter_dataset = MPDocVQADatasetForCLIP(
        json_path=dataset_path,
        image_dir=image_dir,
        tokenizer=tokenizer,
        transform=preprocess_eval,
        split="test",
    )
    filter_data_json = {}
    with open(dataset_path, mode="r", encoding="utf-8") as f:
        filter_data_json = json.load(f)
    # for idx, item in enumerate(tqdm(filter_dataset)):
    with torch.no_grad():
        for idx, item in enumerate(filter_dataset):
            image = item["image"]
            text = item["text"]
            image = image.to(get_cast_device(model)).to(get_cast_dtype(model))
            text = text.to(get_cast_device(model)).long()
            image_features = model.encode_image(image)  # [k, hidden_size]
            text_features = model.encode_text(text)  # [1, hidden_size]
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            image_probs = (
                (100.0 * text_features @ image_features.T).softmax(dim=-1).view(-1)
            )
            sorted_values, sorted_indices = torch.sort(image_probs, descending=True)
            filter_idx = sorted_indices[:3].cpu().detach().numpy().tolist()
            # filter_data_json["data"][idx]["filter_idx"] = filter_idx
            filter_data_json["data"][idx]["filter_page_ids"] = filter_data_json["data"][idx]["page_ids"][filter_idx]
            if verbose:
                logger.info(f"[{idx+1}|{len(filter_dataset)}]")
                logger.info(f"question: {filter_data_json['data'][idx]['question']}")
                logger.info(f"values: {sorted_values.cpu().detach().numpy().tolist()}")
                answer_page_idx = item["answer_page_idx"]
                logger.info(
                    f"indcies: {sorted_indices.cpu().detach().numpy().tolist()}, answer_page_idx: {answer_page_idx}"
                )
                logger.info(f"{filter_data_json['data'][idx]['page_ids']}")
            del (
                image,
                text,
                image_features,
                text_features,
                image_probs,
                sorted_values,
                sorted_indices,
            )
    new_dataset_path = dataset_path.replace(".json", "_filter.json")
    with open(new_dataset_path, mode="w", encoding="utf-8") as f:
        json.dump(filter_data_json, f, ensure_ascii=False, indent=4)

def main():
    eval()


if __name__ == "__main__":
    main()
