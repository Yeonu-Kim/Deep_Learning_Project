# tools/add_bboxes_to_formula_graphs.py

import argparse
import json
import os


def infer_num_nodes(relations):
    """
    relations 형식:
      - 현재 우리 JSON: [[from_idx, to_idx, rel_label], ...]  (list 기반)
      - 혹시 나중에 dict 형식으로 바뀌어도 견딜 수 있게 둘 다 처리

    return: 노드 개수 (max_idx + 1)
    """
    max_idx = -1

    for rel in relations:
        # case 1: dict 형태인 경우 (예: {"from": 0, "to": 1, "label": 3})
        if isinstance(rel, dict):
            frm = rel.get("from", None)
            to = rel.get("to", None)
        # case 2: list/tuple 형태인 경우 (예: [from, to, label])
        elif isinstance(rel, (list, tuple)) and len(rel) >= 2:
            frm, to = rel[0], rel[1]
        else:
            # 이상한 형식이면 스킵
            continue

        # None 아닌 것만 처리
        if frm is None or to is None:
            continue

        try:
            frm = int(frm)
            to = int(to)
        except (TypeError, ValueError):
            continue

        max_idx = max(max_idx, frm, to)

    return max_idx + 1 if max_idx >= 0 else 0


def add_dummy_bboxes_to_sample(sample):
    """
    sample 구조 (예시):
      {
        "relations": [[4, 23, 1], [23, 103, 1], ...],
        "filename": "crohme2019/test/UN19_1032_em_455.inkml",
        ...
      }

    여기에:
      "boxes": [[x, y, w, h], ...]  (노드 수만큼)
    를 dummy 값으로 추가.
    """
    relations = sample.get("relations", [])
    num_nodes = infer_num_nodes(relations)

    # node 개수만큼 dummy bbox 생성
    # 일단 전부 [0.0, 0.0, 1.0, 1.0] 같은 상자 넣어둠.
    boxes = [[0.0, 0.0, 1.0, 1.0] for _ in range(num_nodes)]
    sample["boxes"] = boxes
    return sample


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_json",
        type=str,
        required=True,
        help="입력 formula graphs JSON (예: data/formula/formula_graphs_dummy.json)",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        required=True,
        help="bbox를 추가해 저장할 JSON 경로",
    )
    args = parser.parse_args()

    print(f"[add_bboxes] Loading from: {args.input_json}")
    with open(args.input_json, "r") as f:
        data = json.load(f)

    # 구조 가정:
    # {
    #   "train": { id1: sample1, id2: sample2, ... },
    #   "val":   { ... },
    #   "test":  { ... }
    # }
    #
    # 만약 이런 구조가 아니라 그냥 {id: sample} 하나짜리면
    # 그 경우도 같이 처리하도록 만들어 둠.
    new_data = {}

    # train/val/test 같은 split이 있는 경우
    if all(k in data for k in ["train", "val", "test"]):
        for split in ["train", "val", "test"]:
            split_dict = data.get(split, {})
            new_split = {}
            for key, sample in split_dict.items():
                new_split[key] = add_dummy_bboxes_to_sample(sample)
            new_data[split] = new_split
    else:
        # 그냥 id → sample 구조인 경우
        for key, sample in data.items():
            new_data[key] = add_dummy_bboxes_to_sample(sample)

    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(new_data, f)

    # 간단한 통계 출력
    if "train" in new_data:
        num_train = len(new_data["train"])
        num_val = len(new_data.get("val", {}))
        num_test = len(new_data.get("test", {}))
        print(f"[add_bboxes] Saved to: {args.output_json}")
        print(f"  - train: {num_train} examples")
        print(f"  - val  : {num_val} examples")
        print(f"  - test : {num_test} examples")
    else:
        print(f"[add_bboxes] Saved to: {args.output_json}")
        print(f"  - total: {len(new_data)} examples")


if __name__ == "__main__":
    main()
