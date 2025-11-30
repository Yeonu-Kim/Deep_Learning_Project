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


def build_nodes_from_crohme_graphs_json(sample_dict):
    """
    CROHME 스타일의 *graph-only* JSON 한 개를
    id -> Node 딕셔너리로 변환한다.

    sample_dict 예시:
    {
        "relations": [
            [15, 232, 5],
            [167, 33, 1],
            ...
        ],
        "filename": "crohme2019/train/MfrDB2372.inkml"
    }

    주의:
    - 현재 JSON에는 symbol label / bbox 정보가 없다.
    - 따라서 여기서는 label, bbox를 dummy로 세팅하고
      '관계 그래프 구조'만 Node에 담는다.

    나중에 symbol-level JSON이 생기면:
    - Node(label, bbox)를 그 JSON 기준으로 채우고
    - 이 함수는 relations만 연결해주는 쪽으로 수정하면 된다.
    """

    relations = sample_dict.get("relations", [])

    # relations에 등장하는 node id 집합 먼저 모으기
    node_ids = set()
    for src, dst, _rel in relations:
        node_ids.add(src)
        node_ids.add(dst)

    # Node 생성 (dummy label / bbox)
    nodes = {}
    for nid in node_ids:
        # label은 일단 더미: v{nid}
        # bbox는 아직 정보 없으므로 None
        nodes[nid] = Node(
            node_id=nid,
            label=f"v{nid}",
            bbox=None,
        )

    # relation id -> relation type 문자열 매핑 (임시)
    # TODO: 연우/슬라이드 보고 실제 의미에 맞게 수정하기
    relation_id_to_type = {
        1: "right",       # 기본 가로 연결
        2: "above",       # 분자, 위쪽?
        3: "below",       # 분모, 아래쪽?
        4: "attach",      # 괄호 사이, 루트/분수 기준?
        5: "nolink",      # NoRel 느낌의 관계?
        6: "inside",      # 괄호/루트 안쪽?
    }

    # 관계를 Node에 연결하기
    for src, dst, rel_id in relations:
        if src not in nodes or dst not in nodes:
            continue

        rel_type = relation_id_to_type.get(rel_id, "right")

        src_node = nodes[src]
        dst_node = nodes[dst]

        # Node 클래스 안에 이런 메서드가 있다고 가정:
        #   src_node.add_child(relation_type: str, child_node: Node)
        #   그리고 child_node.parents 같은 것도 내부에서 갱신
        src_node.add_child(rel_type, dst_node)

    return nodes
