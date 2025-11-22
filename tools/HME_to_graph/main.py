from crohme.converter import CROHMEConverter
from util.erase_enter_in_json import compact_json_file
import json

latex_class_file = 'latex_class.json'
# latex_class.json 로드
with open(latex_class_file, 'r', encoding='utf-8') as f:
    latex_data = json.load(f)

converter = CROHMEConverter(
    image_size=(800, 600),
    latex_data=latex_data
)
converter.process_crohme_dataset(
    base_dir='data/crohme',
    output_dir='output',
)
compact_json_file('output/test/test_graphs.json', overwrite=True)
compact_json_file('output/train/train_graphs.json', overwrite=True)
compact_json_file('output/valid/valid_graphs.json', overwrite=True)
