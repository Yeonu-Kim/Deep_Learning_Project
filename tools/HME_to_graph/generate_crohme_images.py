import os
import glob
import math
import xml.etree.ElementTree as ET

from PIL import Image, ImageDraw

# ---------- 설정값 ----------
CANVAS_WIDTH = 800
CANVAS_HEIGHT = 600
PADDING = 20         # 이미지 가장자리 여백
LINE_WIDTH = 3       # 펜 두께
BG_COLOR = "white"
FG_COLOR = "black"


def parse_inkml_traces(inkml_path):
    """
    CROHME InkML 파일에서 trace들을 파싱해서
    각 trace를 (x, y) 리스트로 반환.
    """
    try:
        tree = ET.parse(inkml_path)
    except ET.ParseError as e:
        print(f"[WARN] XML parse error in {inkml_path}: {e} -> 이 파일은 건너뜀")
        return []

    root = tree.getroot()

    # InkML namespace 처리 (보통 {http://www.w3.org/2003/InkML}ink 이런 형태)
    ns = ""
    if root.tag.startswith("{"):
        ns = root.tag.split("}")[0] + "}"

    traces = []
    for trace in root.findall(f".//{ns}trace"):
        raw = trace.text
        if raw is None:
            continue

        coords = []
        # 예: "102 345, 110 360, 130 380"
        for piece in raw.strip().split(","):
            piece = piece.strip()
            if not piece:
                continue
            # piece 예: "102 345" or "102 345 0"
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



def normalize_and_render(traces, out_path):
    """
    주어진 trace들을 CANVAS_WIDTH x CANVAS_HEIGHT 캔버스에
    맞게 normalize해서 PNG로 저장.
    """
    # 모든 점들을 모아서 bounding box 계산
    all_x = []
    all_y = []
    for trace in traces:
        for x, y in trace:
            all_x.append(x)
            all_y.append(y)

    if len(all_x) == 0:
        # 비어 있는 경우: 그냥 빈 이미지 저장
        img = Image.new("RGB", (CANVAS_WIDTH, CANVAS_HEIGHT), BG_COLOR)
        img.save(out_path)
        return

    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)

    # 0 division 방지
    range_x = max_x - min_x
    range_y = max_y - min_y
    if range_x == 0:
        range_x = 1.0
    if range_y == 0:
        range_y = 1.0

    # 출력용 이미지
    img = Image.new("RGB", (CANVAS_WIDTH, CANVAS_HEIGHT), BG_COLOR)
    draw = ImageDraw.Draw(img)

    # padding을 감안한 유효 그리기 영역 크기
    w = CANVAS_WIDTH - 2 * PADDING
    h = CANVAS_HEIGHT - 2 * PADDING

    for trace in traces:
        if len(trace) < 2:
            continue
        pts = []
        for x, y in trace:
            # 0~1 로 normalize
            nx = (x - min_x) / range_x
            ny = (y - min_y) / range_y

            # y축은 위가 0, 아래가 1이 되도록 뒤집기(이미지 좌표계에 맞추기)
            px = PADDING + nx * w
            py = PADDING + (1.0 - ny) * h

            pts.append((px, py))

        # 이어진 선으로 그림
        draw.line(pts, fill=FG_COLOR, width=LINE_WIDTH)

    img.save(out_path)


def process_split(inkml_root, image_root, split):
    """
    주어진 split(train/val/test)에 대해
    inkml → png 변환 수행
    """
    in_dir = os.path.join(inkml_root, split)
    out_dir = os.path.join(image_root, split)
    os.makedirs(out_dir, exist_ok=True)

    inkml_files = sorted(glob.glob(os.path.join(in_dir, "*.inkml")))
    print(f"[{split}] InkML 파일 수: {len(inkml_files)}")

    for idx, inkml_path in enumerate(inkml_files, 1):
        fname = os.path.splitext(os.path.basename(inkml_path))[0]
        out_path = os.path.join(out_dir, fname + ".png")

        traces = parse_inkml_traces(inkml_path)
        normalize_and_render(traces, out_path)

        if idx % 100 == 0 or idx == len(inkml_files):
            print(f"  - {idx}/{len(inkml_files)} 완료")


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data", "crohme")

    inkml_root = os.path.join(data_dir, "inkml")
    image_root = os.path.join(data_dir, "images")

    print("=== [CROHME Image Generation] 시작 ===")
    print(f"INKML_ROOT  : {inkml_root}")
    print(f"IMAGE_ROOT  : {image_root}")
    print()

    for split in ["train", "val", "test"]:
        print(f"--- Split: {split} ---")
        process_split(inkml_root, image_root, split)

    print("\n=== [CROHME Image Generation] 종료 ===")


if __name__ == "__main__":
    main()
