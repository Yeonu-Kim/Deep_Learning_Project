import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from tqdm import tqdm

class CROHMEDataset(Dataset):
    """
    Custom dataset for LaTeX symbol graphs
    Assumes annotation JSON has structure:
    {
        "test": {
            "UN19_1032_em_455": {
                "relations": [[103,1,3],[2,200,1],[200,3,4]],
                "symbol_ids": [103,1,2,200,3],
                "filename": "crohme2019/test/UN19_1032_em_455.inkml"
            },
            ...
        },
        "rel_categories": {"__background__":0,"Right":1,...},
        "num_classes": 320,
        "symbol_to_id": {"0":0,"1":1,...}
    }
    """

    def __init__(self, data_folder, feature_extractor, split, debug=False, num_object_queries=100):
        self.data_folder = data_folder
        self.split = split
        self.feature_extractor = feature_extractor
        self.num_object_queries = num_object_queries  # max number of symbols per formula

        # load annotation
        with open(os.path.join(data_folder, f"{split}_graphs.json"), "r") as f:
            self.ann_data = json.load(f)

        self.data = self.ann_data[split]
        self.rel_categories = self.ann_data["rel_categories"]
        self.num_rel_classes = len(self.rel_categories)
        self.ids = list(self.data.keys())

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        item_id = self.ids[idx]
        ann = self.data[item_id]

        # 이미지 읽기
        img_path = os.path.join(self.data_folder, ann["filename"])
        img = Image.open(img_path).convert("RGB")

        # symbol_ids & relations
        symbol_ids = ann["symbol_ids"]
        relations = ann["relations"]  # already in symbol_ids form [[sub_id, obj_id, rel_id], ...]

        # optional: truncate or pad
        if len(symbol_ids) > self.num_object_queries:
            symbol_ids = symbol_ids[: self.num_object_queries]
        else:
            symbol_ids = symbol_ids + [0] * (self.num_object_queries - len(symbol_ids))  # padding 0

        # convert relations to tensor
        rel_tensor = torch.zeros((self.num_object_queries, self.num_object_queries, self.num_rel_classes), dtype=torch.float32)
        for sub_id, obj_id, rel_id in relations:
            if sub_id < self.num_object_queries and obj_id < self.num_object_queries:
                rel_tensor[sub_id, obj_id, rel_id] = 1.0

        target = {
            "symbol_ids": torch.tensor(symbol_ids, dtype=torch.long),
            "rel": rel_tensor,
            "item_id": item_id,
            "filename": ann["filename"]
        }

        encoding = self.feature_extractor(images=img, return_tensors="pt")
        pixel_values = encoding["pixel_values"].squeeze()
        return pixel_values, target

def latex_symbol_graph_get_statistics(dataset, must_exist=True):
    """
    LaTeX symbol graph용 관계 빈도 행렬 생성
    :param dataset: LaTeXGraphDataset 또는 유사 dataset
    :param must_exist: True이면 relations가 비어있으면 무시
    :return:
        fg_matrix: [num_symbols, num_symbols, num_relations] int64
    """
    num_symbols = dataset.ann_data["num_classes"]
    num_relations = len(dataset.rel_categories)

    fg_matrix = np.zeros((num_symbols, num_symbols, num_relations), dtype=np.int64)

    for idx in tqdm(range(len(dataset))):
        item_id = dataset.ids[idx]
        ann = dataset.data[item_id]

        symbol_ids = ann["symbol_ids"]
        relations = ann["relations"]  # [[sub_symbol_id, obj_symbol_id, rel_id], ...]

        if must_exist and len(relations) == 0:
            continue

        for sub_id, obj_id, rel_id in relations:
            # symbol 개수가 max_objects보다 많으면 넘어갈 수도 있음
            if sub_id < num_symbols and obj_id < num_symbols:
                fg_matrix[sub_id, obj_id, rel_id] += 1

    return fg_matrix
