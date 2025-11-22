from crohme.converter import CROHMEConverter
from util.erase_enter_in_json import compact_json_file
import json

latex_class_file = 'latex_class.json'

with open(latex_class_file, 'r', encoding='utf-8') as f:
    latex_data = json.load(f)

converter = CROHMEConverter(
    image_size=(800, 600),
    latex_data=latex_data
)

converter.convert_annotation_to_graph(
    annotation_file='data/crohme/annotation/crohme2019_test.txt',
    output_file='output/test_graphs.json',
    split_name='test')
converter.convert_annotation_to_graph(
    annotation_file='data/crohme/annotation/crohme2019_train.txt',
    output_file='output/train_graphs.json',
    split_name='test')
converter.convert_annotation_to_graph(
    annotation_file='data/crohme/annotation/crohme2019_valid.txt',
    output_file='output/valid_graphs.json',
    split_name='test')
# converter.convert_annotation_to_graph(
#     annotation_file='data/hme100k/annotation/train.txt',
#     output_file='output/hme100k_valid_graphs.json',
#     split_name='train')

compact_json_file('output/test_graphs.json', overwrite=True)
compact_json_file('output/train_graphs.json', overwrite=True)
compact_json_file('output/valid_graphs.json', overwrite=True)
# compact_json_file('output/hme100k_valid_graphs.json', overwrite=True)
