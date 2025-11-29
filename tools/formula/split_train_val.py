import json
import random
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_json",
        type=str,
        required=True,
        help="현재 train만 있는 formula JSON 경로"
    )
    parser.add_argument(
        "--output_json",
        type=str,
        required=True,
        help="train/val로 나눠서 저장할 새 JSON 경로"
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.1,
        help="validation 비율 (예: 0.1 = 10%)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="셔플용 랜덤 시드"
    )
    args = parser.parse_args()

    print(f"[split] input_json  : {args.input_json}")
    print(f"[split] output_json : {args.output_json}")
    print(f"[split] val_ratio   : {args.val_ratio}")
    print(f"[split] seed        : {args.seed}")

    with open(args.input_json, "r") as f:
        data = json.load(f)

    if "train" not in data:
        raise KeyError(f"[split] JSON 안에 'train' 키가 없습니다. 존재하는 키: {list(data.keys())}")

    train_samples = data["train"]
    n = len(train_samples)
    print(f"[split] 전체 train 샘플 수: {n}")

    indices = list(range(n))
    random.seed(args.seed)
    random.shuffle(indices)

    n_val = int(n * args.val_ratio)
    val_indices = set(indices[:n_val])
    new_train = []
    new_val = []

    for i, sample in enumerate(train_samples):
        if i in val_indices:
            new_val.append(sample)
        else:
            new_train.append(sample)

    print(f"[split] train: {len(new_train)}, val: {len(new_val)}")

    new_data = {}
    # 기존 메타데이터(rel_categories, num_classes, symbol_to_id 등)는 그대로 복사
    for k, v in data.items():
        if k == "train":
            continue
        new_data[k] = v

    new_data["train"] = new_train
    new_data["val"] = new_val

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w") as f:
        json.dump(new_data, f)

    print(f"[split] 저장 완료: {out_path}")

if __name__ == "__main__":
    main()
