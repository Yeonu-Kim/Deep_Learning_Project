# tools/formula/graph2latex/demo.py

"""
간단한 데모/테스트 스크립트.

예시 실행:

    # HME100K    
    (formula-ocr) python -m tools.formula.graph2latex.demo \\
        --mode hme100k \\
        --json_path tools/formula/graph2latex/examples/test_graphs_hme.json \\
        --key test_10
    
    # CROHME 
    (formula-ocr) python -m tools.formula.graph2latex.demo \\
        --mode crohme \\
        --json_path tools/formula/graph2latex/examples/crohme_graphs_sample.json \\
        --key some_id \\
        --latex_class_json data/formula/latex_class.json

"""

import argparse
import json
from typing import Any, Dict

from .adapters import (
    build_nodes_from_hme100k_json,
    build_nodes_from_crohme_json,
)
from .decoder import decode_formula_graph


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--json_path",
        type=str,
        required=True,
        help="입력 JSON 파일 경로",
    )
    parser.add_argument(
        "--key",
        type=str,
        required=False,
        help="JSON 상에서 샘플을 선택하기 위한 key (예: 'test_10')",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="hme100k",
        choices=["hme100k", "crohme"],
        help="HME100K 또는 CROHME 포맷 선택",
    )
    parser.add_argument(
        "--latex_class_json",
        type=str,
        default="data/formula/latex_class.json",
        help="CROHME 모드에서 class_id -> LaTeX 토큰 매핑 JSON 경로",
    )

    args = parser.parse_args()

    data = load_json(args.json_path)

    # 1) HME100K 모드
    # { "test": { "test_10": {...}, "test_1000": {...}, ... } }
    if args.mode == "hme100k":
        if "test" in data:
            bucket = data["test"]
        else:
            bucket = data

        if args.key is None:
            # key가 없으면 첫 번째 샘플 사용
            sample_key = next(iter(bucket.keys()))
        else:
            sample_key = args.key

        sample = bucket[sample_key]
        nodes = build_nodes_from_hme100k_json(sample)
        formula = decode_formula_graph(nodes)

        print(f"[mode    ] hme100k")
        print(f"[filename] {sample.get('filename')}")
        print(f"[gt latex] {sample.get('latex')}")
        print(f"[decoded ] {formula}")
        return
    
    # 2) CROHME 모드
    if args.mode == "crohme":
        # CROHME JSON 구조는 아직 확정되지 않았으므로
        # 일단 HME100K와 비슷하게 'bucket'에서 key로 하나 뽑는 틀만 맞춰둠
        if isinstance(data, dict) and args.key is not None:
            # 최상단 dict에서 바로 key로 접근하는 경우
            if args.key in data:
                sample = data[args.key]
            else:
                raise KeyError(f"Key '{args.key}' not found in JSON.")
        elif isinstance(data, dict) and args.key is None:
            # key를 안 줬다면 첫 번째 엔트리를 사용
            sample_key = next(iter(data.keys()))
            sample = data[sample_key]
        else:
            # 예상에 다른 구조라면 그대로 에러
            raise ValueError("Unexpected CROHME JSON structure")

        # class id -> LaTeX toekn mapping ( ex. latex_class.json)
        id2latex = load_json(args.latex_class_json)

        # 아직 build_nodes_from_crohme_json은 NotImplemented 상태라
        # 실제 실행 시에는 NotImplementedError가 날 수 있음
        nodes = build_nodes_from_crohme_json(sample, id2latex)
        formula = decode_formula_graph(nodes)

        print(f"[mode    ] crohme")
        print(f"[filename] {sample.get('filename', '(no filenema)')}")
        print(f"[decoded ] {formula}")
        return


if __name__ == "__main__":
    main()
