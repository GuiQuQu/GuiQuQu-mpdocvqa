from typing import List
import argparse
import json
import os
from pathlib import Path
import datetime

from loguru import logger
from transformers.generation import GenerationConfig
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer

from metrics import anls

system_message = "You are a helpful assistant."

def get_args():
    parser = argparse.ArgumentParser(
        description="Trainer For CLIP(EVA-02 Model)")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="../mpdocvqa-result/",
    )
    parser.add_argument(
        "--do_eval",
        action="store_true",
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        default="/home/klwang/data/MPDocVQA/images"
    )
    parser.add_argument(
        "--eval_data_path",
        type=str,
        default=None
    )
    parser.add_argument(
        "--do_predict",
        action="store_true",
    )
    parser.add_argument(
        "--predict_data_path",
        type=str,
        default=None
    )
    parser.add_argument(
        "--adapter_name_or_path",
        type=str,
        default=None
    )
    parser.add_argument(
        "--base_model_name_or_path",
        type=str,
        default="/home/klwang/pretrain-model/Qwen-VL-Chat-Int4"
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    return args


def add_log_file(args):
    log_dir = Path(args.output_dir) / "log"
    if not log_dir.exists():
        log_dir.mkdir(parents=True, exist_ok=True)
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    mode = "eval" if args.do_eval else "predict"
    cpkt_name = args.adapter_name_or_path.split("/")[-1]
    log_file_path = log_dir / f"{mode}-{cpkt_name}-{now}.log"
    logger.add(log_file_path, rotation="500 MB")

def get_query_from_list_format(list_format: List[dict]):
        text = ''
        num_images = 0
        for ele in list_format:
            if 'image' in ele:
                num_images += 1
                text += f'Document Picture {num_images}: '
                text += '<img>' + ele['image'] + '</img>'
                text += '\n'
            elif 'text' in ele:
                text += ele['text']
            else:
                raise ValueError("Unsupport element: " + str(ele))
        return text

def do_eval(model, tokenizer, eval_data_path, image_dir, args):
    with open(eval_data_path, "r", encoding="utf-8") as f:
        eval_data = json.load(f)["data"]
    retrieval_hit_count = 0
    eval_result = []
    predict_answer = []
    ground_truth_answer = []
    for i, item in enumerate(eval_data):
        # !!!!使用的page_ids是filter_page_ids
        page_ids = item["filter_page_ids"]
        question = item["question"]
        answers = item["answers"]
        answer_page_idx = item["answer_page_idx"]
        # list query
        list_format = []
        for page_id in page_ids:
            list_format.append({"image": os.path.join(image_dir, f"{page_id}.jpg")})
        # text = f"Answer the question according to the above image.\nQuestion: {question}"
        text = question
        list_format.append({"text": text})
        query = get_query_from_list_format(list_format) 
        if args.verbose:
            logger.info(f"idx {i} query:\n{query}")
            
        # retrieval result
        ground_truth_page_id = item["page_ids"][answer_page_idx]
        item["ground_truth_page_id"] = ground_truth_page_id
        if ground_truth_page_id in item["filter_page_ids"]:
            retrieval_hit_count += 1
        # predict answer
        response, _ = model.chat(
            tokenizer, query, history=[], system=system_message)
        item["predict_answer"] = response
        eval_result.append(item)
        predict_answer.append(response)
        ground_truth_answer.append(answers)
        if args.verbose:
            logger.info(f"idx {i} response:\n{response}")
            logger.info(f"idx {i} answers:\n{answers}")
            logger.info(f"idx {i} anls score: {anls([response], [answers])}")

    anls_result = anls(predict_answer, ground_truth_answer)
    logger.info(f"anls_result: {anls_result:.6f}")
    logger.info(
        f"retrieval_hit_count: {retrieval_hit_count}, retrieval_hit_rate: {(retrieval_hit_count / len(eval_data))*100:.4f}%")
    eval_result = {
        "anls": anls_result,
        "retrieval_hit_rate": retrieval_hit_count / len(eval_data),
        "eval_result": eval_result
    }
    result_dir = Path(args.output_dir) / \
        args.adapter_name_or_path.split("/")[-1]
    if not result_dir.exists():
        result_dir.mkdir(parents=True, exist_ok=True)
    eval_result_path = result_dir / "eval_result.json"
    with open(eval_result_path, "w", encoding="utf-8") as f:
        json.dump(eval_result, f, ensure_ascii=False, indent=4)


def do_predict(model, tokenizer, predict_data_path, image_dir, args):
    pass


def main(args):
    # check args
    if args.output_dir is None:
        raise ValueError("You must specify a output_dir")
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
    if args.adapter_name_or_path is None:
        raise ValueError("You must specify a adpater_name_or_path")
    if args.do_eval:
        if args.eval_data_path is None or args.image_dir is None:
            raise ValueError(
                "You must specify a 'eval_dataset' and 'image_dir' when do_eval is True")
    if args.do_predict:
        if args.predict_data_path is None or args.image_dir is None:
            raise ValueError(
                "You must specify a 'predict_dataset' and 'image_dir' when do_predict is True")

    model = AutoPeftModelForCausalLM.from_pretrained(
        args.adapter_name_or_path,
        device_map="auto",
        trust_remote_code=True).eval()
    generation_config = GenerationConfig.from_pretrained(
        args.base_model_name_or_path,
        trust_remote_code=True
    )
    model.generator_config = generation_config

    logger.info(f"Load adapter model from {args.adapter_name_or_path}...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model_name_or_path,
        trust_remote_code=True
    )
    if args.do_eval:
        do_eval(model, tokenizer, args.eval_data_path, args.image_dir, args)
    if args.do_predict:
        do_predict(model, tokenizer, args.predict_data_path,
                   args.image_dir, args)


if __name__ == "__main__":
    args = get_args()
    add_log_file(args)
    main(args)
