# tools/formula/graph2latex/decoder.py

from typing import Dict, Iterable, List, Set, Optional

from .nodes import Node


def render_group(nodes: Iterable[Node]) -> str:
    """
    여러 노드를 x 좌표 기준으로 정렬한 뒤, 순서대로 render 해서 이어붙인다.
    각 노드마다 독립적인 visited 집합을 써서
    한 노드에서의 순환이 다른 노드에 영향을 주지 않도록 한다.
    """
    ordered: List[Node] = sorted(nodes, key=lambda n: n.x_center)
    parts: List[str] = []
    for n in ordered:
        parts.append(render(n, visited=set()))
    return "".join(parts)


def render(node: Node, visited: Optional[Set[int]] = None) -> str:
    """
    단일 노드 및 그 주변 구조를 LaTeX 문자열로 렌더링.

    visited: 재귀 호출 중 현재 경로에서 방문한 노드 id 집합.
             순환(사이클)이 있으면 무한 재귀를 막기 위해 사용.
    """
    if visited is None:
        visited = set()

    # 사이클 방지: 같은 노드를 다시 만나면 더 들어가지 않고 끊어버림
    if node.id in visited:
        # cycle이지만, 최소한 자기 label은 한 번 찍고 끝내자
        return node.label

    visited.add(node.id)

    # 기본 심볼
    base = node.label

    subs = node.children_by_rel["subscript"]
    sups = node.children_by_rel["superscript"]
    nums = node.children_by_rel["numerator"]
    dens = node.children_by_rel["denominator"]
    rads = node.children_by_rel["radicand"]
    rights = node.children_by_rel["right"]
    insides = node.children_by_rel["inside"]

    # 1) 분수 구조: numerator + denominator 가 둘 다 있으면 \frac 생성
    if nums and dens:
        num_str = render_group(nums)
        den_str = render_group(dens)
        # r"\frac"로 해야 실제 문자열은 "\frac" 하나로 나옴
        base = r"\frac{" + num_str + "}{" + den_str + "}"

    # 2) 루트 구조: radicand 가 있으면 \sqrt 생성
    if rads:
        rad_str = render_group(rads)
        base = r"\sqrt{" + rad_str + "}"

    # 3) 괄호 구조: base가 여는 괄호이고 inside에 노드가 있다면 괄호로 감싸기
    if insides and base in ["(", "[", "{"]:
        inside_str = render_group(insides)
        closing = {"(": ")", "[": "]", "{": "}"}.get(base, "")
        base = base + inside_str + closing

    # 4) 아래 첨자 / 위 첨자
    if subs:
        base += "_{" + render_group(subs) + "}"
    if sups:
        base += "^{" + render_group(sups) + "}"

    # 5) 오른쪽으로 이어지는 기호들
    if rights:
        # rights 쪽도 같은 visited를 넘겨서
        # 1→2→3→1 같은 순환이 있으면 더 이상 타지 않게 한다.
        seq = [base] + [
            render(c, visited=visited) for c in sorted(rights, key=lambda n: n.x_center)
        ]
        visited.remove(node.id)
        return "".join(seq)

    visited.remove(node.id)
    return base


def find_roots(nodes: Dict[int, Node]) -> List[Node]:
    """
    그래프에서 '루트 노드들'(어느 노드의 자식도 아닌 노드들)을 찾아 반환.
    여러 루트가 있을 경우 x 좌표 기준으로 정렬.
    """
    all_nodes = set(nodes.values())
    non_roots = set()

    for node in nodes.values():
        for childs in node.children_by_rel.values():
            for c in childs:
                non_roots.add(c)

    roots = list(all_nodes - non_roots)
    return sorted(roots, key=lambda n: n.x_center)


def decode_formula_graph(nodes: Dict[int, Node]) -> str:
    """
    Node 딕셔너리(id -> Node)로 표현된 수식 그래프를
    최종 LaTeX 문자열로 디코딩.
    """
    roots = find_roots(nodes)
    if not roots:
        # 루트를 못 찾는 경우, 전체 노드를 그냥 정렬해서 이어 붙이는 fallback
        return render_group(nodes.values())
    # 여러 루트가 있으면 각 루트를 하나의 블록으로 보고 공백으로 구분
    parts: List[str] = []
    for r in roots:
        parts.append(render(r, visited=set()))
    return " ".join(parts)
