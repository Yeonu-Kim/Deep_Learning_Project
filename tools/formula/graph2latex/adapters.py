# tools/formula/graph2latex/adapters.py

"""
여기에는 각기 다른 JSON 포맷을
공통 Node 딕셔너리(id -> Node)로 바꿔주는 어댑터를 모아둔다.

- test_graphs_hme.json: 승우님이 주신 HME100K sample(relations만 있는) JSON을 위한 임시 변환
- crohme_graphs_sample.json: (TODO) 연우가 만든 formula graph JSON을 위한 변환
"""

from typing import Dict, Any, Set

from .nodes import Node


def build_nodes_from_hme100k_json(sample: Dict[str, Any]) -> Dict[int, Node]:
    """
    HME100k JSON 예시 형식:

    {
        "relations": [
            [i, j, rel_id],
            ...
        ],
        "filename": "test_10.jpg",
        "latex": "2 3 . 5 + \\frac { 1 } { 2 } x = 1 3 + x"
    }

    이 JSON에는 bbox나 label 정보가 없으므로,
    디코더 "틀" 테스트용으로만 dummy Node를 생성한다.
    """
    node_ids: Set[int] = set()
    for s, o, _ in sample.get("relations", []):
        node_ids.add(int(s))
        node_ids.add(int(o))

    # label은 일단 "x{id}" 같은 dummy 토큰으로 채운다.
    nodes: Dict[int, Node] = {
        node_id: Node(node_id=node_id, label=f"x{node_id}")
        for node_id in sorted(node_ids)
    }

    # rel_id에 상관없이 모두 "right"로 연결하는 임시 버전
    for s, o, rel_id in sample.get("relations", []):
        s_id = int(s)
        o_id = int(o)
        if s_id in nodes and o_id in nodes:
            nodes[s_id].add_child("right", nodes[o_id])

    return nodes


def build_nodes_from_crohme_json(
    sample: Dict[str, Any],
    id2latex: Dict[int, str],
) -> Dict[int, Node]:
    """
    TODO: 연우가 만든 formula graph JSON 포맷에 맞춰 구현할 예정.

    예상 포맷 예시(가정):

    sample = {
        "symbols": [
            {"id": 0, "class_id": 12, "bbox": [x1,y1,x2,y2]},
            ...
        ],
        "relations": [
            {"from": 0, "to": 1, "type": "Right"},
            ...
        ]
    }

    - id2latex: class_id -> LaTeX 토큰 매핑 (latex_class.json)
    """
    # 아직 포맷이 확정되지 않았으므로 일단 NotImplementedError로 남겨둔다.
    raise NotImplementedError(
        "build_nodes_from_crohme_json()은 연우 JSON 포맷 확정 후 구현 예정입니다."
    )
