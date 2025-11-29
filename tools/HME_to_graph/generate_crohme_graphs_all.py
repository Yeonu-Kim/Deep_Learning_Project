from pathlib import Path
import json

from crohme.generator.graph_generator import GraphGenerator
from util.erase_enter_in_json import compact_json_file

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"          # ← tools/HME_to_graph/data
OUTPUT_DIR = BASE_DIR / "output"

# ✅ 여기만 네 데이터 구조에 맞게 수정하면 됨
DEFAULT_SPLITS = {
    "train": DATA_DIR / "crohme/annotation/crohme2019_train.txt",
    "val":   DATA_DIR / "crohme/annotation/crohme2019_valid.txt",
    "test":  DATA_DIR / "crohme/annotation/crohme2019_test.txt",
}
# ↑ 파일 이름이 실제로 다르면 (예: crohme2019_train.txt) 거기에 맞춰서 바꿔줘


def main():
    print("=== [CROHME Graph Generation] 시작 ===")
    print(f"BASE_DIR   : {BASE_DIR}")
    print(f"DATA_DIR   : {DATA_DIR}")
    print(f"OUTPUT_DIR : {OUTPUT_DIR}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # latex_class.json 로드
    latex_class_path = BASE_DIR / "latex_class.json"
    with open(latex_class_path, "r", encoding="utf-8") as f:
        latex_data = json.load(f)

    generator = GraphGenerator(latex_data)

    for split_name, ann_path in DEFAULT_SPLITS.items():
        print(f"\n--- Split: {split_name} ---")
        if not ann_path.exists():
            print(f"  [경고] annotation 파일이 없음: {ann_path}")
            continue

        out_json = OUTPUT_DIR / f"{split_name}_graphs.json"

        print(f"  어노테이션 파일: {ann_path}")
        generator.save_graph_json(
            annotation_file=str(ann_path),
            output_file=str(out_json),
            split_name=split_name,
            output_format="detailed",
        )

        # JSON compact
        print(f"읽는 중: {out_json}")
        compact_json_file(str(out_json), overwrite=True)

        print(f"✅ [{split_name}] 그래프 생성 완료 → {out_json}")

    print("\n=== [CROHME Graph Generation] 종료 ===")


if __name__ == "__main__":
    main()
