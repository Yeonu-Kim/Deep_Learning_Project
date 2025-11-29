import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from tqdm import tqdm

import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from tqdm import tqdm

class CROHMEDataset(Dataset):
    """
    Unified CROHME dataset:
    - Loads graph structure (relations)
    - Loads COCO-style bbox annotations
    - Preserves all attributes from original CROHMEDataset
    """

    def __init__(self, data_folder, feature_extractor, split, debug=False, num_object_queries=100):
        self.data_folder = data_folder
        self.split = split
        self.feature_extractor = feature_extractor
        self.num_object_queries = num_object_queries  # max number of symbols
        
        # ---------------------------------------------------------
        # 1) Load Graph Annotation (original ann_data)
        # ---------------------------------------------------------
        graph_json_path = os.path.join(data_folder, split, f"{split}_graphs.json")
        with open(graph_json_path, "r") as f:
            self.ann_data = json.load(f)

        # same naming as the original CROHME dataset
        self.data = self.ann_data.get("test", {})
        self.rel_categories = self.ann_data["rel_categories"]
        self.symbol_to_id = self.ann_data["symbol_to_id"]
        self.num_rel_classes = len(self.rel_categories)
        self.num_classes = self.ann_data["num_classes"]

        # graph keys (original "ids")
        self.ids = list(self.data.keys())

        # ---------------------------------------------------------
        # 2) Load COCO-style BBox JSON
        # ---------------------------------------------------------
        coco_json_path = os.path.join(data_folder, split, f"{split}.json")
        with open(coco_json_path, "r") as f:
            self.coco_data = json.load(f)

        self.images = {img["id"]: img for img in self.coco_data["images"]}
        self.categories = {cat["id"]: cat for cat in self.coco_data["categories"]}

        # image_id list for indexing
        self.image_ids = list(self.images.keys())

        # group annotations by image
        self.image_annotations = {}
        for ann in self.coco_data["annotations"]:
            img_id = ann["image_id"]
            if img_id not in self.image_annotations:
                self.image_annotations[img_id] = []
            self.image_annotations[img_id].append(ann)

        # Debug: shrink dataset
        if debug:
            self.image_ids = self.image_ids[:8]

    # -------------------------------------------------------------
    def __len__(self):
        return len(self.image_ids)

    # -------------------------------------------------------------
    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        img_info = self.images[image_id]

        # ---------------------------------------------------------
        # Load image
        # ---------------------------------------------------------
        img_path = os.path.join(self.data_folder, self.split, "images", img_info["file_name"])
        if not os.path.exists(img_path):
            return self.__getitem__((idx + 1) % len(self))

        img = Image.open(img_path).convert("RGB")

        # ---------------------------------------------------------
        # BBox & Label handling
        # ---------------------------------------------------------
        annotations = self.image_annotations.get(image_id, [])
        num_objects = min(len(annotations), self.num_object_queries)

        boxes, class_labels, areas = [], [], []

        for ann in annotations[:num_objects]:
            x, y, w, h = ann["bbox"]
            boxes.append([x, y, x + w, y + h])
            class_labels.append(ann["category_id"])
            areas.append(ann["area"])

        # pad to num_object_queries
        while len(boxes) < self.num_object_queries:
            boxes.append([0, 0, 0, 0])
            class_labels.append(0)
            areas.append(0)

        boxes = torch.tensor(boxes, dtype=torch.float32)
        class_labels = torch.tensor(class_labels, dtype=torch.long)
        areas = torch.tensor(areas, dtype=torch.float32)

        # normalize [x_min, y_min, x_max, y_max]
        img_w, img_h = img_info["width"], img_info["height"]
        boxes = boxes / torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)

        # convert to (cx, cy, w, h)
        boxes_cxcywh = torch.zeros_like(boxes)
        boxes_cxcywh[:, 0] = (boxes[:, 0] + boxes[:, 2]) / 2
        boxes_cxcywh[:, 1] = (boxes[:, 1] + boxes[:, 3]) / 2
        boxes_cxcywh[:, 2] = boxes[:, 2] - boxes[:, 0]
        boxes_cxcywh[:, 3] = boxes[:, 3] - boxes[:, 1]

        # ---------------------------------------------------------
        # Relation graph (from ann_data)
        # ---------------------------------------------------------
        rel_tensor = torch.zeros(
            (self.num_object_queries, self.num_object_queries, self.num_rel_classes),
            dtype=torch.float32
        )

        item_key = img_info["file_name"].replace(".png", "")
        if item_key in self.data:
            relations = self.data[item_key].get("relations", [])
            for sub_id, obj_id, rel_id in relations:
                if (sub_id < num_objects and obj_id < num_objects
                        and 0 <= rel_id < self.num_rel_classes):
                    rel_tensor[sub_id, obj_id, rel_id] = 1.0

        # ---------------------------------------------------------
        # EGTR target
        # ---------------------------------------------------------
        target = {
            "class_labels": class_labels,
            "boxes": boxes_cxcywh,
            "rel": rel_tensor,
            "area": areas,
            "iscrowd": torch.zeros(self.num_object_queries, dtype=torch.int64),
            "image_id": torch.tensor([image_id]),
            "orig_size": torch.tensor([img_h, img_w]),
            "size": torch.tensor([img_h, img_w]),
            "item_id": item_key,
            "filename": img_info["file_name"],
        }

        # ---------------------------------------------------------
        # Feature extractor
        # ---------------------------------------------------------
        encoding = self.feature_extractor(images=img, return_tensors="pt")
        pixel_values = encoding["pixel_values"].squeeze(0)

        return pixel_values, target



def latex_symbol_graph_get_statistics(dataset, must_overlap=True):
    """
    CROHME dataset용 관계 빈도 행렬 생성
    
    Args:
        dataset: CROHMEDataset
        must_overlap: True이면 bbox가 겹치는 경우만 카운트
    
    Returns:
        fg_matrix: [num_classes, num_classes, num_relations] int64
    """
    num_classes = dataset.num_classes
    num_relations = dataset.num_rel_classes
    
    fg_matrix = np.zeros((num_classes, num_classes, num_relations), dtype=np.int64)
    
    print(f"Computing statistics for {len(dataset)} samples...")
    
    for idx in tqdm(range(len(dataset))):
        try:
            _, target = dataset[idx]
            
            class_labels = target['class_labels'].numpy()
            boxes = target['boxes'].numpy()
            relations = target['rel'].numpy()
            
            # Find valid objects (non-padding)
            valid_mask = class_labels > 0
            valid_indices = np.where(valid_mask)[0]
            
            if len(valid_indices) == 0:
                continue
            
            # Count relations
            for i in valid_indices:
                for j in valid_indices:
                    if i == j:
                        continue
                    
                    # Check if there's a relation
                    rel_types = np.where(relations[i, j, :] > 0)[0]
                    
                    if len(rel_types) > 0:
                        sub_class = class_labels[i]
                        obj_class = class_labels[j]
                        
                        # Check bbox overlap if required
                        if must_overlap:
                            box1 = boxes[i]
                            box2 = boxes[j]
                            
                            # Simple overlap check (can be refined)
                            overlap = not (box1[0] > box2[2] or box1[2] < box2[0] or
                                         box1[1] > box2[3] or box1[3] < box2[1])
                            
                            if not overlap:
                                continue
                        
                        # Count each relation type
                        for rel_type in rel_types:
                            fg_matrix[sub_class, obj_class, rel_type] += 1
        
        except Exception as e:
            print(f"Error processing index {idx}: {e}")
            continue
    
    return fg_matrix
