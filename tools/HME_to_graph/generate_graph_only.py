from crohme.converter import CROHMEConverter
from util.erase_enter_in_json import compact_json_file
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent  # tools/HME_to_graph
ANN_FILE = BASE_DIR / "debug_data/dummy_ann.txt"   # 우리가 방금 만든 더미 파일
latex_class_file = BASE_DIR / "latex_class.json"

with open(latex_class_file, 'r', encoding='utf-8') as f:
    latex_data = json.load(f)

converter = CROHMEConverter(
    image_size=(800, 600),
    latex_data=latex_data
)

output_dir = BASE_DIR / "output"
output_dir.mkdir(parents=True, exist_ok=True)

# test split 하나만 생성
converter.convert_annotation_to_graph(
    annotation_file=str(ANN_FILE),
    output_file=str(output_dir / "test_graphs.json"),
    split_name='test',
)

compact_json_file(str(output_dir / "test_graphs.json"), overwrite=True)
print("✅ dummy annotation → output/test_graphs.json 생성 완료")
