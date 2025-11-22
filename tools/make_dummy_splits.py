import json
import os


def main():
    # 원본 JSON (지금 네가 쓰고 있는 test_graphs.json)
    src_path = os.path.join("data", "formula", "test_graphs.json")

    # 출력 JSON (새로 만들 dummy 버전)
    dst_path = os.path.join("data", "formula", "formula_graphs_dummy.json")

    if not os.path.exists(src_path):
        raise FileNotFoundError(f"Source JSON not found: {src_path}")

    print(f"[make_dummy_splits] Loading from: {src_path}")
    with open(src_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 기존 구조 가정:
    # {
    #   "test": { ... },
    #   "rel_categories": { ... },
    #   "num_classes": 320,
    #   "symbol_to_id": { ... }
    # }
    if "test" not in data:
        raise KeyError(
            "'test' split not found in source JSON. "
            "현재 test_graphs.json 구조가 예상과 다를 수 있습니다."
        )

    test_graphs = data["test"]
    rel_categories = data.get("rel_categories", {})
    num_classes = data.get("num_classes", None)
    symbol_to_id = data.get("symbol_to_id", {})

    # ⚠️ 아직 진짜 train/val 분리가 없기 때문에
    # 일단 디버깅/구조 확인용으로
    # train / val / test 모두 동일한 내용을 복사해 둠.
    new_data = {
        "train": test_graphs,
        "val": test_graphs,
        "test": test_graphs,
        "rel_categories": rel_categories,
        "num_classes": num_classes,
        "symbol_to_id": symbol_to_id,
    }

    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    with open(dst_path, "w", encoding="utf-8") as f:
        json.dump(new_data, f, ensure_ascii=False, indent=2)

    print(f"[make_dummy_splits] Saved dummy splits to: {dst_path}")
    print("  - train:", len(new_data["train"]), "examples")
    print("  - val  :", len(new_data["val"]), "examples")
    print("  - test :", len(new_data["test"]), "examples")


if __name__ == "__main__":
    main()
