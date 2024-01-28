import json


if __name__ == "__main__":
    json_paths = [
        "/root/autodl-tmp/data/train.json",
        "/root/autodl-tmp/data/test.json",
        "/root/autodl-tmp/data/val.json",
    ]

    for json_path in json_paths:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)