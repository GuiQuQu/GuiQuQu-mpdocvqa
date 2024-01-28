from qwen_vl_chat import QWenConfig, QWenLMHeadModel, QWenTokenizer

import subprocess


def main():
    # result = subprocess.run("source /etc/network_turbo", shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    model_name_or_path = "Qwen/Qwen-VL-Chat"
    config = QWenConfig.from_pretrained(
        model_name_or_path, cache_dir="/root/autodl-tmp/pretrain-model"
    )
    # config.bf16 = True
    config.fp16 = True
    tokenizer = QWenTokenizer.from_pretrained(
        model_name_or_path, cache_dir="/root/autodl-tmp/pretrain-model"
    )
    model = QWenLMHeadModel.from_pretrained(
        model_name_or_path,
        config=config,
        device_map="cuda:0",
        cache_dir="/root/autodl-tmp/pretrain-model",
    )
    model.eval()

    query = tokenizer.from_list_format(
        [
            {
                "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
            },
            {"text": "这是什么"},
        ]
    )
    response, history = model.chat(tokenizer, query=query, history=None)
    print(response)

    # result = subprocess.run("unset http_proxy && unset https_proxy", shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


if __name__ == "__main__":
    main()
