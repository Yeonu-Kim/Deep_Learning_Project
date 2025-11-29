# tools/formula/graph2latex/nodes.py

from typing import Dict, List, Optional


REL_TYPES = [
    "right",
    "subscript",
    "superscript",
    "numerator",
    "denominator",
    "inside",
    "radicand",
]


class Node:
    """
    수식 그래프의 한 노드(기호)를 표현하는 클래스.

    - id: JSON 상의 심볼 id (int)
    - label: LaTeX 토큰 또는 심볼 이름 (예: "x", "1", "\\alpha")
    - bbox: [x1, y1, x2, y2] 형식의 bounding box (이미지 좌표계) 또는 None
    - children_by_rel: relation 타입별 자식 노드들
    """

    def __init__(
        self,
        node_id: int,
        label: str,
        bbox: Optional[List[float]] = None,
    ) -> None:
        self.id: int = node_id
        self.label: str = label
        self.bbox: Optional[List[float]] = bbox

        # relation 타입별 자식 리스트
        self.children_by_rel: Dict[str, List["Node"]] = {
            rel: [] for rel in REL_TYPES
        }

    @property
    def x_center(self) -> float:
        """
        노드의 x 중심좌표.
        bbox가 없으면 id를 이용해서 임시로 정렬 기준으로 사용.
        """
        if self.bbox is None:
            return float(self.id)
        x1, _, x2, _ = self.bbox
        return 0.5 * (x1 + x2)

    @property
    def y_center(self) -> float:
        """
        노드의 y 중심좌표 (필요하면 relation 분류에 사용).
        """
        if self.bbox is None:
            return 0.0
        _, y1, _, y2 = self.bbox
        return 0.5 * (y1 + y2)

    def add_child(self, rel_type: str, child: "Node") -> None:
        """
        relation 타입에 맞는 자식 노드를 추가.
        rel_type이 미리 정의한 타입이 아니면 그냥 right로 fallback.
        """
        if rel_type not in self.children_by_rel:
            rel_type = "right"
        self.children_by_rel[rel_type].append(child)

    def __repr__(self) -> str:
        return f"Node(id={self.id}, label={self.label!r})"
