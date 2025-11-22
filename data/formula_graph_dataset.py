# data/formula_graph_dataset.py

import json
import os
from typing import Any, Dict, List, Tuple

import torch
from torch.utils.data import Dataset
from PIL import Image


class FormulaGraphDataset(Dataset):
    """
    Crohme/HME 수식 그래프용 Dataset.

    JSON 포맷(목표 예시):

    {
      "train": {
        "UN19_1032_em_455": {
          "filename": "crohme2019/train/UN19_1032_em_455.png",
          "boxes": [[x0,y0,x1,y1], ...],   # 심볼 bbox (image 좌표계)
          "labels": [class_id, ...],       # symbol_to_id 기반
          "relations": [[i,j,rel_id], ...],# i, j는 boxes/labels 인덱스
          "latex": "\\frac{a+b}{c^2}"      # (옵션)
        },
        ...
      },

      "val": { ... },
      "test": { ... },

      "symbol_to_id": {...},
      "rel_categories": {...},
      "num_classes": 320
    }

    지금 네가 가진 test_graphs.json은 일단 "test"만 있고
    boxes / labels는 아직 없는 상태라,
    실제 학습 전에 JSON을 한 번 손봐야 한다는 점만 기억해 두면 됨.
    """

    def __init__(
        self,
        json_path: str,
        split: str,
        images_root: str,
        feature_extractor=None,
        transforms=None,
    ):
        super().__init__()
        assert split in ["train", "val", "test"], f"Unknown split: {split}"

        # JSON 로드
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if split not in data:
            raise KeyError(
                f"Split '{split}' not found in {json_path}. "
                f"지금 JSON 안에 '{split}' 키가 없는 것 같아. "
                f"현재 구조를 한번 확인해봐야 해."
            )

        # 수식별 그래프 정보
        self.graphs: Dict[str, Dict[str, Any]] = data[split]

        # 전역 메타 정보
        self.symbol_to_id = data.get("symbol_to_id", {})
        self.rel_categories = data.get("rel_categories", {})
        self.num_classes = data.get("num_classes", None)

        # id 리스트 (키 순서 고정)
        self.ids: List[str] = list(self.graphs.keys())

        self.images_root = images_root
        self.feature_extractor = feature_extractor
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        gid = self.ids[idx]
        ginfo = self.graphs[gid]

        # -------------------------
        # 1. 이미지 로드
        # -------------------------
        # ginfo["filename"] 예시: "crohme2019/test/UN19_1032_em_455.png"
        img_path = os.path.join(self.images_root, ginfo["filename"])
        if not os.path.exists(img_path):
            raise FileNotFoundError(
                f"이미지 파일을 찾을 수 없음: {img_path}\n"
                f"images_root='{self.images_root}', filename='{ginfo['filename']}'"
            )

        image = Image.open(img_path).convert("RGB")
        w, h = image.size

        # -------------------------
        # 2. GT boxes / labels / relations
        # -------------------------
        if "boxes" not in ginfo or "labels" not in ginfo:
            # 아직 bbox/label이 안 붙은 단계라면 여기서 바로 친절하게 에러
            raise KeyError(
                "현재 JSON 엔트리에 'boxes' 또는 'labels' 키가 없습니다.\n"
                "→ 학습 전에 test_graphs.json을 확장해서 "
                "'boxes' / 'labels'를 추가해줘야 합니다."
            )

        boxes = torch.as_tensor(ginfo["boxes"], dtype=torch.float32)   # [N,4], x0,y0,x1,y1
        labels = torch.as_tensor(ginfo["labels"], dtype=torch.int64)   # [N]
        relations = torch.as_tensor(ginfo["relations"], dtype=torch.int64)  # [M,3] (sub, obj, rel_id)

        target: Dict[str, torch.Tensor] = {
            "boxes": boxes,
            "labels": labels,
            "relations": relations,
            "image_id": torch.tensor([idx], dtype=torch.int64),
            "orig_size": torch.tensor([h, w], dtype=torch.int64),
        }

        # latex는 optional (있으면 string으로만 보관)
        if "latex" in ginfo:
            target["latex"] = ginfo["latex"]  # string 그대로 둬도 되고, 나중에 tokenizer에서 처리

        # -------------------------
        # 3. transforms (이미지/박스 augmentation)
        # -------------------------
        if self.transforms is not None:
            image, target = self.transforms(image, target)

        # -------------------------
        # 4. feature_extractor (DETR용 전처리)
        # -------------------------
        if self.feature_extractor is not None:
            encoded = self.feature_extractor(
                images=image,
                return_tensors="pt",
            )
            # encoded["pixel_values"]: [1, C, H, W]
            pixel_values = encoded["pixel_values"].squeeze(0)
        else:
            # 만약 feature_extractor를 안 쓰는 경우가 있다면, 여기서 직접 ToTensor
            import numpy as np

            arr = np.array(image).astype("float32") / 255.0  # [H,W,C]
            arr = arr.transpose(2, 0, 1)                     # [C,H,W]
            pixel_values = torch.from_numpy(arr)

        return pixel_values, target
