# tools/HME_to_graph/add_real_bboxes_from_formula.py

import argparse
import json
import os
import xml.etree.ElementTree as ET


# ------------------------------------------------------------
# 1. JSON 로딩
# ------------------------------------------------------------
def load_formula_samples(json_path):
    """
    formula-style JSON을 읽어서 (root_obj, samples 리스트)를 반환.
    - 최상단이 바로 리스트면 root_obj=None, samples=그 리스트
    - dict인 경우, samples / data 등의 키에서 리스트를 찾아본다.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    # case 1: top-level list
    if isinstance(obj, list):
        return None, obj

    # case 2: top-level dict 안에 "samples" 키
    if isinstance(obj, dict):
        if "samples" in obj and isinstance(obj["samples"], list):
            return obj, obj["samples"]

        # 다른 키 이름 아래에 샘플 리스트가 들어 있을 수 있음
        for key, val in obj.items():
            if isinstance(val, list) and len(val) > 0 and isinstance(val[0], dict):
                return obj, val

    raise ValueError("알 수 없는 formula JSON 형식입니다. samples 리스트를 찾을 수 없음.")


# ------------------------------------------------------------
# 2. InkML 인덱스 만들기 (basename -> full path)
# ------------------------------------------------------------
def build_inkml_index(inkml_root):
    """
    inkml_root 아래를 모두 뒤져서
    파일 이름(basename.inkml) -> 전체 경로 로 매핑한 dict 반환.
    """
    index = {}
    for dirpath, _, filenames in os.walk(inkml_root):
        for fname in filenames:
            if fname.lower().endswith(".inkml"):
                full = os.path.join(dirpath, fname)
                index[fname] = full
    return index


# ------------------------------------------------------------
# 3. formula 샘플에서 InkML 경로 찾기
# ------------------------------------------------------------
def get_inkml_path_from_formula_sample(sample, inkml_root, inkml_index):
    """
    formula 샘플 안에 들어 있는 정보들을 바탕으로
    대응되는 InkML 파일 경로를 찾는다.
    - 가능하면 sample["inkml_path"] / sample["inkml_relpath"] 등 직접 경로 사용
    - 없으면 id / filename 등에서 basename을 추출해서 inkml_index로 찾는다.
    """
    # 1) 직접 경로가 들어 있는 경우들
    direct_keys = [
        "inkml_path",
        "inkml_relpath",
        "inkml",
        "path",
        "file_path",
        "file",
        "filename",
        "image_id",
        "img_id",
        "id",
    ]

    for key in direct_keys:
        if key in sample:
            val = str(sample[key])

            # 이미 .inkml로 끝나는 경우
            if val.lower().endswith(".inkml"):
                # 절대경로로 존재하면 그대로 사용
                if os.path.isabs(val) and os.path.exists(val):
                    return val
                # inkml_root 기준 상대경로로 존재하는지 확인
                cand = os.path.join(inkml_root, val)
                if os.path.exists(cand):
                    return cand
                # basename만 가지고 인덱스에서 찾기
                base = os.path.basename(val)
                if base in inkml_index:
                    return inkml_index[base]

            # ".inkml"가 안 붙어 있으면 붙여서 다시 시도
            base = os.path.basename(val)
            if not base.lower().endswith(".inkml"):
                base = base + ".inkml"

            if base in inkml_index:
                return inkml_index[base]

    # 그래도 못 찾으면 실패
    raise FileNotFoundError("이 formula 샘플에 해당하는 InkML 파일을 찾을 수 없습니다.")


# ------------------------------------------------------------
# 4. InkML 파싱해서 traces 읽기
# ------------------------------------------------------------
def parse_inkml_traces(inkml_path):
    """
    InkML 파일을 파싱해서
    trace_id -> [(x1,y1), (x2,y2), ...] 형태의 dict로 반환.
    """
    tree = ET.parse(inkml_path)
    root = tree.getroot()

    ns = {"ink": root.tag.split("}")[0].strip("{")}

    traces = {}
    # namespace가 붙어 있을 수도 있고, 아닐 수도 있어서 두 가지 모두 시도
    # 1) 네임스페이스 포함
    for t in root.findall(".//{*}trace"):
        tid = t.get("id")
        if tid is None:
            continue
        points = []
        if t.text:
            # trace 텍스트는 "x y, x y, ..." 혹은 줄바꿈으로 구분되어 있음
            raw = t.text.replace(",", " ").split()
            # raw = [x1, y1, x2, y2, ...] 형태라고 가정
            if len(raw) % 2 != 0:
                raw = raw[:-1]
            for i in range(0, len(raw), 2):
                try:
                    x = float(raw[i])
                    y = float(raw[i + 1])
                    points.append((x, y))
                except ValueError:
                    continue
        traces[tid] = points

    return traces


# ------------------------------------------------------------
# 5. 노드 리스트 찾기
# ------------------------------------------------------------
def get_nodes_from_formula_sample(sample):
    """
    formula_json 한 샘플에서 '노드 리스트'를 찾아서 반환한다.
    포맷이 조금씩 다를 수 있어서 여러 키를 순서대로 탐색하고,
    그래도 못 찾으면 heuristic으로 추정한다.
    찾지 못하면 ValueError를 던진다.
    """

    if not isinstance(sample, dict):
        raise ValueError("formula sample 형식이 dict가 아닙니다.")

    # 1) 가장 단순한 형태: sample["nodes"]
    if "nodes" in sample and isinstance(sample["nodes"], list):
        return sample["nodes"]

    # 2) graph 안에 nodes가 있는 형태: sample["graph"]["nodes"]
    if "graph" in sample and isinstance(sample["graph"], dict):
        g = sample["graph"]
        if "nodes" in g and isinstance(g["nodes"], list):
            return g["nodes"]

    # 3) 다른 이름으로 되어 있을 가능성
    for key in ["symbols", "objects", "entities", "graph_nodes"]:
        if key in sample and isinstance(sample[key], list):
            return sample[key]

    # 4) heuristic:
    # dict 안에서 "list of dict" 이면서 label/strokes 관련 키를 가진 애를 노드 후보로 본다
    for v in sample.values():
        if isinstance(v, list) and len(v) > 0 and isinstance(v[0], dict):
            first = v[0]
            if any(
                k in first
                for k in ("label", "sym", "symbol", "stroke_ids", "trace_ids", "strokes")
            ):
                return v

    # 여기까지 왔는데도 못 찾으면 에러
    raise ValueError("formula sample에서 노드 리스트(nodes/symbols/objects)를 찾을 수 없습니다.")


# ------------------------------------------------------------
# 6. 노드별 / 전체 bbox 계산
# ------------------------------------------------------------
def compute_bboxes_for_formula_sample(sample, inkml_root, inkml_index):
    """
    한 formula 샘플에 대해
    - InkML 파일에서 trace 좌표들을 읽고
    - 각 노드에 대해 stroke_ids / trace_ids 기반으로 bbox 계산
    - 전체 formula bbox도 계산하여 반환
    반환값:
        {
          "global_bbox": [xmin, ymin, xmax, ymax],
          "node_bboxes": [ [xmin, ymin, xmax, ymax] or None, ... ]  # 노드 순서와 동일
        }
    """
    # 1) 노드 리스트 가져오기 (실패하면 global bbox만 사용)
    try:
        nodes = get_nodes_from_formula_sample(sample)
    except ValueError as e:
        raise e

    # 2) InkML 경로 찾기
    inkml_path = get_inkml_path_from_formula_sample(sample, inkml_root, inkml_index)
    if not os.path.exists(inkml_path):
        raise FileNotFoundError(f"InkML 파일 없음: {inkml_path}")

    # 3) trace 좌표 읽기
    traces_dict = parse_inkml_traces(inkml_path)

    # trace 전체에서 global bbox 계산
    all_points = []
    for pts in traces_dict.values():
        all_points.extend(pts)

    if not all_points:
        # 포인트가 아예 없으면 bbox 계산 불가
        raise RuntimeError(f"InkML에서 포인트를 찾을 수 없습니다: {inkml_path}")

    xs = [p[0] for p in all_points]
    ys = [p[1] for p in all_points]
    global_bbox = [min(xs), min(ys), max(xs), max(ys)]

    # 4) 각 노드별 bbox 계산
    node_bboxes = []
    for node in nodes:
        # stroke_ids / trace_ids / strokes 등에서 id 리스트 찾기
        stroke_keys = ["stroke_ids", "trace_ids", "strokes", "trace_id_list"]
        stroke_ids = None
        for sk in stroke_keys:
            if sk in node:
                stroke_ids = node[sk]
                break

        if stroke_ids is None:
            # stroke 정보가 없으면 None
            node_bboxes.append(None)
            continue

        # stroke_ids 가 정수 리스트 / 문자열 리스트 등일 수 있으니 문자열화
        if isinstance(stroke_ids, int):
            stroke_ids = [stroke_ids]
        elif isinstance(stroke_ids, str):
            # 콤마/공백 구분 문자열일 수 있음
            parts = stroke_ids.replace(",", " ").split()
            stroke_ids = [p for p in parts]
        elif isinstance(stroke_ids, list):
            stroke_ids = [str(s) for s in stroke_ids]
        else:
            node_bboxes.append(None)
            continue

        pts = []
        for sid in stroke_ids:
            if sid in traces_dict:
                pts.extend(traces_dict[sid])

        if not pts:
            node_bboxes.append(None)
            continue

        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        node_bboxes.append([min(xs), min(ys), max(xs), max(ys)])

    return {
        "global_bbox": global_bbox,
        "node_bboxes": node_bboxes,
    }


# ------------------------------------------------------------
# 7. 메인 루프
# ------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="formula-style JSON에 실제 InkML 기반 bbox를 붙이는 스크립트"
    )
    parser.add_argument(
        "--formula_json", type=str, required=True, help="입력 formula-style JSON 경로"
    )
    parser.add_argument(
        "--inkml_root", type=str, required=True, help="InkML 파일들이 있는 root 디렉토리"
    )
    parser.add_argument(
        "--output_json", type=str, required=True, help="bbox가 추가된 JSON 저장 경로"
    )

    args = parser.parse_args()

    print("=== [Add Real BBoxes from Formula] 시작 ===")
    print(f"formula_json  : {args.formula_json}")
    print(f"inkml_root    : {args.inkml_root}")
    print(f"output_json   : {args.output_json}")
    print()

    root_obj, samples = load_formula_samples(args.formula_json)
    print(f"총 formula 샘플 수: {len(samples)}")

    # InkML 인덱스 미리 만들기
    inkml_index = build_inkml_index(args.inkml_root)

    n_no_inkml = 0
    n_no_nodes = 0
    n_ok = 0

    for i, sample in enumerate(samples):
        try:
            info = compute_bboxes_for_formula_sample(sample, args.inkml_root, inkml_index)
        except FileNotFoundError as e:
            # 해당 InkML이 없으면 샘플은 그대로 두고 넘어감
            print(f"[WARN] {e}")
            n_no_inkml += 1
            continue
        except ValueError as e:
            # 노드 리스트를 못 찾은 경우 (global bbox만 쓰겠다거나…)
            print(f"[WARN] 노드 리스트를 찾지 못해 글로벌 bbox만 사용: '{e}'")
            # InkML 은 찾을 수 있는지 한번 더 시도해서 global bbox만이라도 붙여준다
            try:
                inkml_path = get_inkml_path_from_formula_sample(
                    sample, args.inkml_root, inkml_index
                )
                traces_dict = parse_inkml_traces(inkml_path)
                all_points = []
                for pts in traces_dict.values():
                    all_points.extend(pts)
                if all_points:
                    xs = [p[0] for p in all_points]
                    ys = [p[1] for p in all_points]
                    global_bbox = [min(xs), min(ys), max(xs), max(ys)]
                    # sample 레벨에 global bbox만 저장
                    sample["global_bbox_real"] = global_bbox
            except Exception as e2:
                print(f"[WARN] global bbox 계산도 실패: {e2}")
            n_no_nodes += 1
            continue
        except Exception as e:
            print(f"[WARN] bbox 계산 중 예기치 못한 에러: {e}")
            continue

        # info.global_bbox, info.node_bboxes 를 sample 안에 기록
        global_bbox = info["global_bbox"]
        node_bboxes = info["node_bboxes"]

        sample["global_bbox_real"] = global_bbox

        try:
            nodes = get_nodes_from_formula_sample(sample)
            for node, nb in zip(nodes, node_bboxes):
                if nb is not None:
                    node["bbox"] = nb
                    node["bbox_type"] = "real"  # 나중에 dummy와 구분하고 싶을 때
        except ValueError:
            # 여기까지 왔으면 거의 없겠지만, 혹시 node 탐색 실패하면 global만 유지
            pass

        n_ok += 1

        if (i + 1) % 500 == 0:
            print(f"  - {i+1}/{len(samples)} 샘플 처리 완료")

    print()
    print(f"[요약] bbox 계산 성공: {n_ok}")
    print(f"[요약] InkML 없음: {n_no_inkml}")
    print(f"[요약] 노드 리스트 없음(=global bbox만 시도): {n_no_nodes}")

    # 결과 저장
    print("\nJSON 저장 중...")
    if root_obj is None:
        out_obj = samples
    else:
        # root_obj 안에 samples 리스트가 있었을 수도 있고, 아닐 수도 있음.
        # 일단 root_obj 그대로 두고 samples 부분만 덮어쓴다.
        if "samples" in root_obj and isinstance(root_obj["samples"], list):
            root_obj["samples"] = samples
            out_obj = root_obj
        else:
            # 어떤 키 아래에 있었는지 모르니 그냥 samples만 저장
            out_obj = samples

    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(out_obj, f, ensure_ascii=False)

    print(f"✅ 완료! → {args.output_json}")
    print("=== [Add Real BBoxes from Formula] 종료 ===")


if __name__ == "__main__":
    main()
