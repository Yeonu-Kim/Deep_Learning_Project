import json
from pathlib import Path
import argparse


def convert_hme_graphs_to_formula(input_json: str, output_json: str):
    input_path = Path(input_json)
    output_path = Path(output_json)

    with open(input_path, "r", encoding="utf-8") as f:
        src = json.load(f)

    out = {}

    # HME_to_graph 결과에서 쓸 수 있는 split 키 후보
    split_keys = ["train", "val", "test"]

    any_split_found = False
    for split in split_keys:
        if split not in src:
            continue

        split_dict = src[split]  # { file_id: {labels, relations, filename, ...}, ... }
        any_split_found = True

        samples_out = []
        for file_id, sample in split_dict.items():
            labels = sample.get("labels", [])
            relations = sample.get("relations", [])
            filename = sample.get("filename", file_id)

            # 아직 bbox가 없으므로 labels 길이만큼 dummy bbox 생성
            # 나중에 inkml/이미지 기반 실제 bbox로 바꾸면 됨.
            bboxes = [[0.0, 0.0, 1.0, 1.0] for _ in labels]

            samples_out.append(
                {
                    "filename": filename,
                    "labels": labels,
                    "relations": relations,
                    "bboxes": bboxes,
                }
            )

        out[split] = samples_out
        print(f"[convert] split='{split}' → {len(samples_out)} samples")

    if not any_split_found:
        raise ValueError(
            f"입력 JSON에서 train/val/test 중 어떤 split도 찾지 못했습니다: {input_json}"
        )

    # 메타 정보(rel_categories, num_classes, symbol_to_id)가 있으면 그대로 보존
    for meta_key in ["rel_categories", "num_classes", "symbol_to_id"]:
        if meta_key in src:
            out[meta_key] = src[meta_key]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print(f"[convert] Saved formula-style JSON to: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_json",
        type=str,
        required=True,
        help="HME_to_graph에서 생성된 *_graphs.json 경로",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        required=True,
        help="EGTR formula 포맷으로 저장할 JSON 경로",
    )

    args = parser.parse_args()
    convert_hme_graphs_to_formula(args.input_json, args.output_json)


if __name__ == "__main__":
    main()
