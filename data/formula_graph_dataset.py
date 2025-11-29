# data/formula_graph_dataset.py
#
# CROHME/HME 스타일 "수식 그래프 JSON"을 EGTR에서 사용할 수 있는
# PyTorch Dataset 형태로 바꿔주는 클래스.
#
# JSON 예시:
# {
#   "train": [
#     {
#       "filename": "crohme2019/train/MfrDB2372.inkml",
#       "labels": [15, 232, 167, ...],
#       "relations": [
#         [0, 1, 5],
#         [2, 3, 1],
#         ...
#       ],
#       "bboxes": [
#         [x_min, y_min, x_max, y_max],
#         ...
#       ]
#     },
#     ...
#   ],
#   "val": [...],
#   "test": [...]
# }
#
# 이미지 파일은 예를 들어:
#   tools/HME_to_graph/data/crohme/images/crohme2019/train/MfrDB2372.png
# 처럼 있다고 가정하고,
#   images_root = "tools/HME_to_graph/data/crohme/images"
# 라고 넘겨주면 됨.

import json
import os
from typing import List, Dict, Any

import torch
from torch.utils.data import Dataset
from PIL import Image


class FormulaGraphDataset(Dataset):
    """
    수식 그래프(JSON) + 이미지 파일을 EGTR 학습용으로 변환하는 Dataset.
    """

    def __init__(
        self,
        json_path: str,
        split: str,
        images_root: str,
        feature_extractor,
    ):
        """
        Args:
            json_path: formula_graphs_*.json 경로
            split: "train", "val", "test" 중 하나
            images_root: 이미지 루트 디렉토리
                        (예: tools/HME_to_graph/data/crohme/images)
            feature_extractor: DeformableDetrFeatureExtractor (train/val용)
        """
        super().__init__()
        self.json_path = json_path
        self.split = split
        self.images_root = images_root
        self.feature_extractor = feature_extractor

        # ----- JSON 로드 -----
        with open(json_path, "r") as f:
            data = json.load(f)

        # data가 {"train": [...], "val": [...]} 형태라고 가정
        if isinstance(data, dict):
            if split not in data:
                raise KeyError(
                    f"[FormulaGraphDataset] JSON 안에 split='{split}' 키가 없습니다. "
                    f"존재하는 키: {list(data.keys())}"
                )
            self.samples: List[Dict[str, Any]] = data[split]
        elif isinstance(data, list):
            # 혹시라도 리스트 형태면 전체를 split 상관 없이 사용
            print(
                f"[FormulaGraphDataset] 경고: JSON이 리스트 형태입니다. "
                f"split='{split}'을 무시하고 전체 {len(data)}개 샘플 사용."
            )
            self.samples = data
        else:
            raise ValueError(
                f"[FormulaGraphDataset] 지원하지 않는 JSON 구조입니다: type={type(data)}"
            )

        if len(self.samples) == 0:
            print(
                f"[FormulaGraphDataset] 경고: split='{split}'에 해당하는 샘플이 0개입니다."
            )

        # ----- 클래스 개수(num_classes) 추정 -----
        all_labels = []
        for sample in self.samples:
            all_labels.extend(sample.get("labels", []))
        if all_labels:
            self.num_classes = max(all_labels) + 1
        else:
            # 라벨이 하나도 없을 경우 fallback (거의 없겠지만)
            print(
                "[FormulaGraphDataset] WARNING: labels가 비어 있어서 num_classes=1로 설정합니다."
            )
            self.num_classes = 1

        # ----- relation 카테고리 추출 -----
        # relations: [ [sub_idx, obj_idx, rel_type], ... ]
        rel_value_set = set()
        for sample in self.samples:
            for s, o, p in sample.get("relations", []):
                rel_value_set.add(p)

        # 예: p 값이 {1,2,4,5} 이런 식이면 정렬해서 [1,2,4,5]
        self.rel_values: List[int] = sorted(rel_value_set)
        # p 원본 값 -> 0 ~ (num_rel-1) 로 매핑
        self.rel_id_map: Dict[int, int] = {
            p_val: i for i, p_val in enumerate(self.rel_values)
        }
        # EGTR 쪽에서 이름만 필요하므로 문자열로
        self.rel_categories: List[str] = [str(p_val) for p_val in self.rel_values]

        if not self.rel_categories:
            # relation이 하나도 없는 경우도 대비
            print(
                "[FormulaGraphDataset] WARNING: relations가 비어 있습니다. rel_categories도 비어 있음."
            )

        print(
            f"[FormulaGraphDataset] Loaded {len(self.samples)} samples from '{json_path}' (split='{split}')"
        )
        print(f"[FormulaGraphDataset] num_classes = {self.num_classes}")
        print(f"[FormulaGraphDataset] #rel_categories = {len(self.rel_categories)}")

    def __len__(self):
        return len(self.samples)

    def _build_image_path(self, filename: str) -> str:
        """
        JSON의 filename (예: 'crohme2019/train/MfrDB2372.inkml') 을
        실제 이미지 경로로 변환.

        images_root = 'tools/HME_to_graph/data/crohme/images'
        이고,
        실제 파일이
          tools/HME_to_graph/data/crohme/images/crohme2019/train/MfrDB2372.png
        라고 있을 때를 가정.
        """
        # inkml -> png 로 확장자 변경
        if filename.lower().endswith(".inkml"):
            rel_path = filename[:-6] + ".png"
        else:
            # 이미 .png 라면 그대로 사용
            rel_path = filename

        image_path = os.path.join(self.images_root, rel_path)
        return image_path

    def __getitem__(self, idx: int):
        sample = self.samples[idx]

        # ----- 이미지 로드 -----
        filename = sample["filename"]  # 예: 'crohme2019/train/MfrDB2372.inkml'

        # 1) .inkml → .png로 확장자만 변경
        img_rel_path = filename
        if img_rel_path.endswith(".inkml"):
            img_rel_path = img_rel_path[:-6] + ".png"  # ".inkml" 길이 6

        # 2) 실제 이미지 폴더 구조에 맞게 prefix 정리
        #    JSON은 'crohme2019/train/...' 인데,
        #    실제 폴더는 'train/...' 만 있다고 가정
        if img_rel_path.startswith("crohme2019/"):
            img_rel_path = img_rel_path[len("crohme2019/"):]  # 앞의 'crohme2019/' 잘라내기

        # 3) 최종 경로 조합
        img_path = os.path.join(self.images_root, img_rel_path)

        if not os.path.exists(img_path):
            raise FileNotFoundError(
                f"[FormulaGraphDataset] 이미지 파일을 찾을 수 없습니다: {img_path} "
                f"(json filename='{filename}')"
            )


        image = Image.open(img_path).convert("RGB")
        width, height = image.size

        # feature_extractor로 전처리
        encoding = self.feature_extractor(images=image, return_tensors="pt")
        # [1, C, H, W] -> [C, H, W]
        pixel_values = encoding["pixel_values"].squeeze(0)

        # ----- GT 라벨 구성 -----
        labels = sample.get("labels", [])
        bboxes = sample.get("bboxes", [])
        relations = sample.get("relations", [])

        num_objs = len(labels)
        if len(bboxes) != num_objs:
            # 이 경우는 데이터 오류. 일단 assert로 잡자.
            raise ValueError(
                f"[FormulaGraphDataset] labels 개수({len(labels)})와 "
                f"bboxes 개수({len(bboxes)})가 다릅니다. idx={idx}, filename={filename}"
            )

        # boxes: [num_objs, 4] (x_min, y_min, x_max, y_max) 형식으로 가정
        # JSON이 dummy인 경우 (0,0,1,1)이지만 형식만 맞으면 DE-TR는 학습은 됨.
        boxes = torch.tensor(bboxes, dtype=torch.float32)  # [N, 4]
        class_labels = torch.tensor(labels, dtype=torch.long)  # [N]

        # rel: [num_objs, num_objs, num_rel_categories]
        num_rel = len(self.rel_categories)
        rel_tensor = torch.zeros(
            (num_objs, num_objs, num_rel), dtype=torch.float32
        )

        for s_idx, o_idx, p_val in relations:
            if s_idx < 0 or s_idx >= num_objs or o_idx < 0 or o_idx >= num_objs:
                # 잘못된 index 방어
                continue
            if p_val not in self.rel_id_map:
                # 정의되지 않은 relation value 방어
                continue
            p_idx = self.rel_id_map[p_val]
            rel_tensor[s_idx, o_idx, p_idx] = 1.0

        # EGTR가 기대하는 target 포맷
        target = {
            "boxes": boxes,                     # [N, 4]
            "class_labels": class_labels,       # [N]
            "rel": rel_tensor,                  # [N, N, num_rel]
            "orig_size": torch.tensor([height, width], dtype=torch.long),
            "size": torch.tensor([height, width], dtype=torch.long),
            "image_id": torch.tensor(idx, dtype=torch.long),
        }

        return pixel_values, target
