# tools/formula/graph2latex/demo.py

"""
간단한 데모/테스트 스크립트.

예시 실행:

    # 1) HME100K (예제 JSON)
    (formula-ocr) python -m tools.formula.graph2latex.demo \
        --mode hme100k \
        --json_path tools/formula/graph2latex/examples/test_graphs_hme.json \
        --split_key test \
        --sample_key test_10

    # 2) CROHME 그래프 전용 (train_graphs.json 등)
    (formula-ocr) python -m tools.formula.graph2latex.demo \
        --mode crohme_graphs \
        --json_path dataset/train/train_graphs.json \
        --split_key test \
        --sample_key MfrDB2372

설명:

- hme100k 모드:
    - JSON 구조: { "test": { "test_10": {...}, ... } } 또는 { "some_id": {...}, ... }
    - sample에는 'relations', 'filename', 'latex' 등이 들어 있다고 가정.
    - build_nodes_from_hme100k_json()으로 Node dict를 만든 뒤,
      decode_formula_graph()로 LaTeX 문자열을 복원.

- crohme_graphs 모드:
    - JSON 구조: { "test": { "MfrDB2372": {...}, ... } } 또는 { "MfrDB2372": {...}, ... }
    - sample에는 'relations', 'filename'만 있고, symbol label/bbox는 없다고 가정.
    - build_nodes_from_crohme_graphs_json() 안에서
      dummy label/bbox로 Node를 만들고, 관계 그래프 구조만 테스트용으로 디코딩.
"""

import argparse
import json
from typing import Any, Dict

from .adapters import (
    build_nodes_from_hme100k_json,
    build_nodes_from_crohme_graphs_json,
)
from .decoder import decode_formula_graph


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def pick_sample(
    data: Dict[str, Any],
    split_key: str | None,
    sample_key: str | None,
) -> tuple[str, Dict[str, Any]]:
    """
    공통 유틸:
    - data: json 전체
    - split_key: 최상단에 split 이름이 있을 경우 (예: 'test')
    - sample_key: split 안에서 사용할 key (예: 'MfrDB2372', 'test_10')

    return: (실제로 사용한 sample_key, sample_dict)
    """
    # 1) split 선택
    if split_key is not None and isinstance(data, dict) and split_key in data:
        bucket = data[split_key]
    else:
        bucket = data

    if not isinstance(bucket, dict):
        raise ValueError(f"Bucket is not a dict. Got type={type(bucket)}")

    # 2) sample 선택
    if sample_key is None:
        # key를 안 줬으면 첫 번째 key 사용
        sample_key = next(iter(bucket.keys()))

    if sample_key not in bucket:
        raise KeyError(f"Sample key '{sample_key}' not found in bucket keys.")

    return sample_key, bucket[sample_key]


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--json_path",
        type=str,
        required=True,
        help="입력 그래프 JSON 파일 경로",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="hme100k",
        choices=["hme100k", "crohme_graphs"],
        help="HME100K 또는 CROHME 그래프 전용 포맷 선택",
    )
    parser.add_argument(
        "--split_key",
        type=str,
        default=None,
        help="최상단에 split이 있을 경우 해당 key (예: 'test')",
    )
    parser.add_argument(
        "--sample_key",
        type=str,
        default=None,
        help="split 안에서 사용할 샘플 ID (예: 'test_10', 'MfrDB2372'). "
             "지정하지 않으면 첫 번째 샘플 사용",
    )

    args = parser.parse_args()

    data = load_json(args.json_path)
    sample_key, sample = pick_sample(
        data=data,
        split_key=args.split_key,
        sample_key=args.sample_key,
    )

    # =========================
    # 1) HME100K 모드
    # =========================
    if args.mode == "hme100k":
        nodes = build_nodes_from_hme100k_json(sample)
        formula = decode_formula_graph(nodes)

        print(f"[mode      ] hme100k")
        print(f"[sample key] {sample_key}")
        print(f"[filename  ] {sample.get('filename')}")
        print(f"[gt latex  ] {sample.get('latex')}")
        print(f"[decoded   ] {formula}")
        return

    # =========================
    # 2) CROHME 그래프 모드
    # =========================
    if args.mode == "crohme_graphs":
        nodes = build_nodes_from_crohme_graphs_json(sample)
        formula = decode_formula_graph(nodes)

        print(f"[mode      ] crohme_graphs")
        print(f"[sample key] {sample_key}")
        print(f"[filename  ] {sample.get('filename', '(no filename)')}")
        print(f"[decoded   ] {formula}")
        return


if __name__ == "__main__":
    main()
