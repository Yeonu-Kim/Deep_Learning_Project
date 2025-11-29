import os
import json
import argparse
import xml.etree.ElementTree as ET

# ====== 이미지/좌표 스케일 설정 (generate_crohme_images.py와 동일하게!) ======
CANVAS_WIDTH = 800
CANVAS_HEIGHT = 600
PADDING = 20


# -------------------- InkML 파싱 & 좌표 변환 -------------------- #

def parse_inkml_traces(inkml_path):
    """
    InkML 파일에서 trace들을 파싱해서
    각 trace를 [(x1, y1), (x2, y2), ...] 리스트로 반환.
    """
    try:
        tree = ET.parse(inkml_path)
    except ET.ParseError as e:
        print(f"[WARN] XML parse error in {inkml_path}: {e} -> 이 샘플은 bbox 계산 스킵")
        return []

    root = tree.getroot()

    # namespace 처리
    ns = ""
    if root.tag.startswith("{"):
        ns = root.tag.split("}")[0] + "}"

    traces = []
    for trace in root.findall(f".//{ns}trace"):
        raw = trace.text
        if raw is None:
            continue

        coords = []
        for piece in raw.strip().split(","):
            piece = piece.strip()
            if not piece:
                continue
            parts = piece.split()
            if len(parts) < 2:
                continue
            try:
                x = float(parts[0])
                y = float(parts[1])
                coords.append((x, y))
            except ValueError:
                continue

        if coords:
            traces.append(coords)

    return traces


def normalize_traces_to_image(traces):
    """
    원래 좌표계 traces를 800x600 이미지 좌표계로 normalize.
    generate_crohme_images.py와 동일 로직.
    """
    all_x = []
    all_y = []
    for tr in traces:
        for x, y in tr:
            all_x.append(x)
            all_y.append(y)

    if len(all_x) == 0:
        return []

    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)

    range_x = max_x - min_x
    range_y = max_y - min_y
    if range_x == 0:
        range_x = 1.0
    if range_y == 0:
        range_y = 1.0

    w = CANVAS_WIDTH - 2 * PADDING
    h = CANVAS_HEIGHT - 2 * PADDING

    norm_traces = []
    for tr in traces:
        pts = []
        for x, y in tr:
            nx = (x - min_x) / range_x
            ny = (y - min_y) / range_y
            px = PADDING + nx * w
            py = PADDING + (1.0 - ny) * h  # y축 뒤집기
            pts.append((px, py))
        norm_traces.append(pts)

    return norm_traces


# -------------------- formula JSON 로딩 -------------------- #

def load_formula_samples(formula_json_path):
    """
    formula_graphs_crohme_*_dummybbox.json 로드.
    리스트 또는 { "samples": [...] } 형태 모두 지원.
    """
    with open(formula_json_path, "r") as f:
        data = json.load(f)

    if isinstance(data, dict) and "samples" in data:
        samples = data["samples"]
        root_obj = data
    elif isinstance(data, list):
        samples = data
        root_obj = samples
    else:
        raise ValueError("알 수 없는 formula JSON 형식입니다. samples 리스트를 찾을 수 없음.")

    return root_obj, samples


# -------------------- sample 안에서 필드 찾기 -------------------- #

def get_inkml_path_from_formula_sample(sample, inkml_root):
    """
    formula sample에서 inkml 상대 경로 필드를 찾아서
    inkml_root와 합쳐 절대 경로 생성.

    아래 키들 중 하나를 사용할 거라고 가정:
      - "inkml_path"
      - "inkml_relpath"
      - "file"
      - "filename"
      - "id"
      - "name"
    필요하면 여기에서 필드명 한 번만 수정해주면 됨.
    """
    candidates = ["inkml_path", "inkml_relpath", "file", "filename", "id", "name"]
    rel = None
    for key in candidates:
        if key in sample:
            rel = sample[key]
            break

    if rel is None:
        raise KeyError("formula sample에서 inkml 경로 필드를 찾을 수 없습니다. get_inkml_path_from_formula_sample() 안의 candidates를 수정하세요.")

    # 혹시 .inkml 확장자가 빠져 있으면 붙여줌
    if not rel.endswith(".inkml"):
        rel = rel + ".inkml"

    inkml_path = os.path.join(inkml_root, rel)
    return inkml_path


def get_nodes_from_formula_sample(sample):
    """
    노드 리스트 필드를 찾음.
    아래 키들 중 하나를 사용할 거라고 가정:
      - "nodes"
      - "symbols"
      - "objects"
    """
    for key in ["nodes", "symbols", "objects"]:
        if key in sample and isinstance(sample[key], list):
            return sample[key]
    raise KeyError("formula sample에서 노드 리스트(nodes/symbols/objects)를 찾을 수 없습니다.")


def get_stroke_ids_from_node(node):
    """
    노드 안에서 stroke id 리스트 필드를 찾아줌.
    아래 키들 중 하나를 사용할 거라고 가정:
      - "strokes"
      - "stroke_ids"
      - "trace_ids"
    """
    for key in ["strokes", "stroke_ids", "trace_ids"]:
        if key in node:
            return node[key]
    return None


# -------------------- bbox 계산 -------------------- #

def compute_bboxes_for_formula_sample(sample, inkml_root):
    """
    formula sample 하나에 대해
    - InkML 읽어서 trace 좌표 normalize
    - 각 노드별로 stroke 묶음의 bbox 계산
    → [ [xmin,ymin,xmax,ymax], ... ] 리스트 반환
    """
    try:
        inkml_path = get_inkml_path_from_formula_sample(sample, inkml_root)
    except Exception as e:
        print(f"[WARN] inkml path 추출 실패: {e}")
        return None

    if not os.path.exists(inkml_path):
        print(f"[WARN] InkML 파일 없음: {inkml_path}")
        return None

    traces = parse_inkml_traces(inkml_path)
    if not traces:
        return None

    norm_traces = normalize_traces_to_image(traces)
    if not norm_traces:
        return None

    # 전체 수식 bbox (fallback용)
    all_x = [x for tr in norm_traces for (x, y) in tr]
    all_y = [y for tr in norm_traces for (x, y) in tr]
    global_bbox = [min(all_x), min(all_y), max(all_x), max(all_y)]

    try:
        nodes = get_nodes_from_formula_sample(sample)
    except KeyError as e:
        print(f"[WARN] 노드 리스트를 찾지 못해 글로벌 bbox만 사용: {e}")
        # 노드 개수는 알 수 없으므로 bboxes 계산 불가 → None
        return None

    bboxes = []
    for node_idx, node in enumerate(nodes):
        stroke_ids = get_stroke_ids_from_node(node)

        xs, ys = [], []
        if stroke_ids is not None:
            for sid in stroke_ids:
                if not isinstance(sid, int):
                    try:
                        sid = int(sid)
                    except ValueError:
                        continue
                if sid < 0 or sid >= len(norm_traces):
                    continue
                for (px, py) in norm_traces[sid]:
                    xs.append(px)
                    ys.append(py)

        if xs and ys:
            xmin, ymin = min(xs), min(ys)
            xmax, ymax = max(xs), max(ys)
            bboxes.append([xmin, ymin, xmax, ymax])
        else:
            # stroke 정보 없거나 비어 있으면 전체 수식 bbox로 fallback
            bboxes.append(global_bbox)

    return bboxes


# -------------------- main -------------------- #

def main():
    parser = argparse.ArgumentParser(description="CROHME formula JSON에 real bbox 추가하기 (formula JSON만 사용)")
    parser.add_argument("--formula_json", type=str, required=True,
                        help="convert_hme_graphs_to_formula_json.py로 만든 *_dummybbox.json 경로")
    parser.add_argument("--inkml_root", type=str, required=True,
                        help="InkML 루트 디렉토리 (예: tools/HME_to_graph/data/crohme/inkml)")
    parser.add_argument("--output_json", type=str, required=True,
                        help="real bbox를 채워서 저장할 output JSON 경로")

    args = parser.parse_args()

    print("=== [Add Real BBoxes from Formula] 시작 ===")
    print(f"formula_json  : {args.formula_json}")
    print(f"inkml_root    : {args.inkml_root}")
    print(f"output_json   : {args.output_json}")
    print()

    root_obj, formula_samples = load_formula_samples(args.formula_json)
    total = len(formula_samples)
    print(f"총 formula 샘플 수: {total}")

    for idx, sample in enumerate(formula_samples, 1):
        bboxes = compute_bboxes_for_formula_sample(sample, args.inkml_root)
        if bboxes is None:
            # 실패 시 기존 bboxes 유지 (dummy 그대로)
            continue

        sample["bboxes"] = bboxes

        if idx % 500 == 0 or idx == total:
            print(f"  - {idx}/{total} 샘플 bbox 갱신 완료")

    print("\nJSON 저장 중...")
    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(root_obj, f)

    print(f"✅ 완료! → {args.output_json}")
    print("=== [Add Real BBoxes from Formula] 종료 ===")


if __name__ == "__main__":
    main()
