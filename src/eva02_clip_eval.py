import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from eva02_clip import load_model_tokenizer_transform
from dataset import MPDocVQADatasetForCLIP, CollatorForCLIP


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


def main():
    model, _, preprocess_eval, tokenizer = load_model_tokenizer_transform(
        model_name="EVA02-L-14",
        tokenizer_path="../pretrain-model/eva02_large_patch14_clip_224.merged2b_s4b_b131k",
        cpkt_path="../clip-outputs/2024-01-16/checkpoint-4532",
        file_name="pytorch_model.bin",
        precision="fp16",
        device="cuda:0",
    )
    model.eval()
    eval_dataset = MPDocVQADatasetForCLIP(
        json_path="../data/MPDocVQA/val.json",
        image_dir="../data/MPDocVQA/images",
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


if __name__ == "__main__":
    main()
