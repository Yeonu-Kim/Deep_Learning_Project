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
        with open(os.path.join(data_folder, f"{split}/{split}_graphs.json"), "r") as f:
            self.ann_data = json.load(f)

        self.data = self.ann_data["test"]
        self.rel_categories = self.ann_data["rel_categories"]
        self.symbol_to_id = self.ann_data["symbol_to_id"]
        self.num_rel_classes = len(self.rel_categories)
        self.ids = list(self.data.keys())
        self.num_classes = self.ann_data["num_classes"]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        item_id = self.ids[idx]
        ann = self.data[item_id]

        # 이미지 읽기
        filename = ann["filename"]
        filename = filename.replace('crohme2019/valid/', 'valid/images/')
        filename = filename.replace('crohme2019/train/', 'train/images/')
        filename = filename.replace('crohme2019/test/', 'test/images/')
        # inkml을 png로 변환
        if filename.endswith('.inkml'):
            filename = filename[:-6] + '.png'
        
        img_path = os.path.join(self.data_folder, filename)
        img = Image.open(img_path).convert("RGB")

        # symbol_ids & relations
        relations = ann.get("relations", [])
        symbol_set = set()
        for rel in relations:
            if len(rel) >= 2:
                symbol_set.add(rel[0])  # subject
                symbol_set.add(rel[1])  # object
        symbol_ids = sorted(list(symbol_set))
        relations = ann["relations"]  # already in symbol_ids form [[sub_id, obj_id, rel_id], ...]

        # optional: truncate or pad
        num_symbols = len(symbol_ids)
        if num_symbols > self.num_object_queries:
            symbol_ids = symbol_ids[:self.num_object_queries]
            num_symbols = self.num_object_queries
        
        symbol_to_idx = {sid: i for i, sid in enumerate(symbol_ids)}
        padded_symbol_ids = symbol_ids + [0] * (self.num_object_queries - len(symbol_ids))

        # convert relations to tensor
        rel_tensor = torch.zeros((self.num_object_queries, self.num_object_queries, self.num_rel_classes), dtype=torch.float32)
        
        for relation in relations:
            if len(relation) != 3:
                continue
            sub_id, obj_id, rel_id = relation
            
            # Map symbol IDs to indices
            if sub_id in self.symbol_to_id and obj_id in self.symbol_to_id:
                sub_idx = self.symbol_to_id[sub_id]
                obj_idx = self.symbol_to_id[obj_id]
                
                if sub_idx < self.num_object_queries and obj_idx < self.num_object_queries:
                    if 0 <= rel_id < self.num_rel_classes:
                        rel_tensor[sub_idx, obj_idx, rel_id] = 1.0

        # EGTR에서 필요한 필드들
        target = {
            "class_labels": torch.tensor(padded_symbol_ids, dtype=torch.long),  # EGTR expects this
            "boxes": torch.zeros((self.num_object_queries, 4), dtype=torch.float32),  # dummy boxes
            "rel": rel_tensor,
            "image_id": torch.tensor([idx], dtype=torch.long),
            "orig_size": torch.tensor([img.height, img.width], dtype=torch.long),
            "size": torch.tensor([img.height, img.width], dtype=torch.long),
            "item_id": item_id,
            "filename": ann["filename"],
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

    for idx in range(len(dataset)):
        item_id = dataset.ids[idx]
        ann = dataset.data[item_id]
        relations = ann["relations"]  # [[sub_symbol_id, obj_symbol_id, rel_id], ...]

        if must_exist and len(relations) == 0:
            continue

        for sub_id, obj_id, rel_id in relations:
            # symbol 개수가 max_objects보다 많으면 넘어갈 수도 있음
            if sub_id < num_symbols and obj_id < num_symbols:
                fg_matrix[sub_id, obj_id, rel_id] += 1

    return fg_matrix
