#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
CROHME 스타일 어노테이션(txt) → 그래프 JSON 생성 스크립트.

- GraphGenerator 위치: tools/HME_to_graph/crohme/generator/graph_generator.py
- latex_class.json:    tools/HME_to_graph/latex_class.json
- 더미 어노테이션:     tools/HME_to_graph/debug_data/dummy_ann.txt
- 출력:                tools/HME_to_graph/output/test_graphs.json (기본)

실행 예시 (Deep_Learning_Project 폴더에서):

    (formula-ocr) python tools/HME_to_graph/generate_crohme_graphs_all.py
"""

import json
from pathlib import Path

# ------------------------------------------------------------
# 1. 경로 설정
# ------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent          # .../Deep_Learning_Project/tools/HME_to_graph
LATEX_CLASS_PATH = BASE_DIR / "latex_class.json"
DEBUG_DIR = BASE_DIR / "debug_data"
OUTPUT_DIR = BASE_DIR / "output"

# 기본으로는 우리가 만든 dummy_ann.txt만 사용 (sanity check 용)
DEFAULT_SPLITS = {
    "test": DEBUG_DIR / "dummy_ann.txt",
}

# ------------------------------------------------------------
# 2. GraphGenerator import
#    실제 위치: tools/HME_to_graph/crohme/generator/graph_generator.py
#    → 모듈 경로: crohme.generator.graph_generator
# ------------------------------------------------------------
try:
    from crohme.generator.graph_generator import GraphGenerator
except ImportError as e:
    msg = (
        "[generate_crohme_graphs_all] GraphGenerator import 실패\n"
        "  - 기대 위치: tools/HME_to_graph/crohme/generator/graph_generator.py\n"
        "  - 모듈 경로:  crohme.generator.graph_generator\n"
        f"  - 원래 에러: {e}\n\n"
        "지금 파일 구조가 다음과 같은지 다시 확인해줘:\n"
        "  tools/HME_to_graph/\n"
        "    ├─ crohme/\n"
        "    │    ├─ __init__.py\n"
        "    │    ├─ converter.py\n"
        "    │    └─ generator/\n"
        "    │         ├─ __init__.py\n"
        "    │         └─ graph_generator.py (여기에 class GraphGenerator)\n"
    )
    raise ImportError(msg)

# ------------------------------------------------------------
# 3. (옵션) JSON compact 도구 로드
# ------------------------------------------------------------
try:
    # tools/HME_to_graph/util/erase_enter_in_json.py 에 있다고 가정
    from util.erase_enter_in_json import compact_json_file
except ImportError:
    compact_json_file = None


def main():
    print("=== [CROHME Graph Generation] 시작 ===")
    print(f"BASE_DIR        : {BASE_DIR}")
    print(f"LATEX_CLASS_PATH: {LATEX_CLASS_PATH}")
    print(f"OUTPUT_DIR      : {OUTPUT_DIR}")

    if not LATEX_CLASS_PATH.exists():
        raise FileNotFoundError(
            f"latex_class.json 을 찾을 수 없습니다: {LATEX_CLASS_PATH}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ----- latex_class.json 로드 -----
    with open(LATEX_CLASS_PATH, "r", encoding="utf-8") as f:
        latex_data = json.load(f)

    generator = GraphGenerator(latex_data=latex_data)

    # 지금은 dummy_ann.txt → test split 하나만 생성 (sanity check)
    splits = DEFAULT_SPLITS

    for split_name, ann_path in splits.items():
        print(f"\n--- Split: {split_name} ---")
        print(f"  어노테이션 파일: {ann_path}")

        if not ann_path.exists():
            raise FileNotFoundError(
                f"[{split_name}] 어노테이션 파일이 존재하지 않습니다: {ann_path}"
            )

        output_path = OUTPUT_DIR / f"{split_name}_graphs.json"

        # Graph JSON 생성
        generator.save_graph_json(
            annotation_file=str(ann_path),
            output_file=str(output_path),
            split_name=split_name,
            output_format="detailed",
        )

        # JSON 한 줄로 압축 (있으면)
        if compact_json_file is not None:
            try:
                compact_json_file(str(output_path), overwrite=True)
                print(f"  → compact_json_file 적용 완료: {output_path}")
            except Exception as e:
                print(f"  ⚠ compact_json_file 적용 중 오류 (무시): {e}")

        print(f"✅ [{split_name}] 그래프 생성 완료 → {output_path}")

    print("\n=== [CROHME Graph Generation] 종료 ===")


if __name__ == "__main__":
    main()
