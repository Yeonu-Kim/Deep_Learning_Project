# data/formula/formula_graph_dataset.py

import json
import os
from typing import Any, Dict, List, Tuple, Union

import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np


class FormulaGraphDataset(Dataset):
    """
    Crohme/HME 수식 그래프용 Dataset.

    ✅ 지원 JSON 포맷 1 (dict):

    {
      "train": {
        "UN19_1032_em_455": {
          "filename": "crohme2019/train/UN19_1032_em_455.png",
          "boxes": [[x0,y0,x1,y1], ...],   # 또는 "bboxes"
          "labels": [class_id, ...],
          "relations": [[i,j,rel_id], ...],
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

    ✅ 지원 JSON 포맷 2 (list, EGTR formula용):

    {
      "test": [
        {
          "filename": "DUMMY_001.inkml",  # 또는 "DUMMY_001"
          "bboxes": [[x0,y0,x1,y1], ...],
          "labels": [...],
          "relations": [[i,j,rel_id], ...]
        },
        ...
      ],
      "rel_categories": {...},
      "num_classes": 320,
      "symbol_to_id": {...}
    }

    ➕ train/val이 없으면 train/val 요청 시 자동으로 test split을 대신 사용.
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

        # -------------------------
        # 1. JSON 로드
        # -------------------------
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # split이 없으면 test로 fallback
        if split not in data:
            if "test" in data:
                print(
                    f"[FormulaGraphDataset] Split '{split}' not found in {json_path}. "
                    f"대신 'test' split을 사용합니다 (sanity check 용)."
                )
                split_data = data["test"]
            else:
                raise KeyError(
                    f"Split '{split}' not found in {json_path}. "
                    f"지금 JSON 안에 '{split}' 키가 없는 것 같아. "
                    f"현재 구조를 한번 확인해봐야 해."
                )
        else:
            split_data = data[split]

        # dict / list 형식 자동 감지
        if isinstance(split_data, dict):
            # {"id": {...}, ...}
            self.mode = "dict"
            self.graphs: Dict[str, Dict[str, Any]] = split_data
            self.ids: List[Union[str, int]] = list(self.graphs.keys())
        elif isinstance(split_data, list):
            # [{...}, {...}, ...]
            self.mode = "list"
            self.samples: List[Dict[str, Any]] = split_data
            self.ids = list(range(len(self.samples)))
        else:
            raise TypeError(
                f"Unsupported split data type: {type(split_data)} "
                f"(dict 또는 list만 지원합니다)"
            )

        # 전역 메타 정보
        self.symbol_to_id = data.get("symbol_to_id", {})
        self.rel_categories = data.get("rel_categories", {})
        self.num_classes = data.get("num_classes", None)

        self.images_root = images_root
        self.feature_extractor = feature_extractor
        self.transforms = transforms

    # ----------------------------------------------------
    # 🔥 이미지 경로 자동 해결: 확장자 / 경로 유연 처리
    # ----------------------------------------------------
    def _resolve_image_path(self, filename_field: str) -> str:
        """
        JSON의 filename 필드를 이용해 실제 이미지 파일 경로를 찾는다.

        1) images_root/ + filename_field 그대로 시도
        2) 안 되면 base name만 뽑아서
           base + [.png, .jpg, .jpeg, .bmp] 순서로 탐색
        """
        # 1) 그대로 시도
        direct_path = os.path.join(self.images_root, filename_field)
        if os.path.exists(direct_path):
            return direct_path

        # 2) base name + 다양한 확장자
        base = os.path.basename(filename_field)
        base_no_ext = os.path.splitext(base)[0]

        candidate_exts = [".png", ".jpg", ".jpeg", ".bmp"]
        tried_paths = [direct_path]

        for ext in candidate_exts:
            cand = os.path.join(self.images_root, base_no_ext + ext)
            tried_paths.append(cand)
            if os.path.exists(cand):
                return cand

        raise FileNotFoundError(
            "[FormulaGraphDataset] 이미지 파일을 찾을 수 없습니다.\n"
            f"  filename_field = '{filename_field}'\n"
            f"  images_root    = '{self.images_root}'\n"
            "  시도한 경로들:\n    - "
            + "\n    - ".join(tried_paths)
        )

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, Any]]:
        # -------------------------
        # 0. 현재 샘플 ginfo 가져오기
        # -------------------------
        if self.mode == "dict":
            gid = self.ids[idx]
            ginfo = self.graphs[gid]
        else:  # "list"
            gid = idx
            ginfo = self.samples[idx]

        # -------------------------
        # 1. 이미지 로드
        # -------------------------
        if "filename" not in ginfo:
            raise KeyError(
                f"Sample {gid} 에 'filename' 필드가 없습니다. "
                f"JSON 생성 시 filename을 꼭 넣어줘야 합니다."
            )

        img_path = self._resolve_image_path(ginfo["filename"])
        image = Image.open(img_path).convert("RGB")
        w, h = image.size

        # -------------------------
        # 2. GT boxes / labels / relations
        # -------------------------
        # boxes / bboxes 둘 다 지원
        if "boxes" in ginfo:
            boxes_data = ginfo["boxes"]
        elif "bboxes" in ginfo:
            boxes_data = ginfo["bboxes"]
        else:
            raise KeyError(
                f"Sample {gid} 에 'boxes' 또는 'bboxes' 필드가 없습니다.\n"
                f"→ 학습용 JSON에는 반드시 bbox 정보가 들어가야 합니다."
            )

        if "labels" not in ginfo:
            raise KeyError(
                f"Sample {gid} 에 'labels' 필드가 없습니다.\n"
                f"→ 각 심볼의 class id 리스트가 필요합니다."
            )

        boxes = torch.as_tensor(boxes_data, dtype=torch.float32)      # [N,4]
        labels = torch.as_tensor(ginfo["labels"], dtype=torch.int64)  # [N]

        rel_list = ginfo.get("relations", [])
        if rel_list is None:
            rel_list = []
        relations = torch.as_tensor(rel_list, dtype=torch.int64)      # [M,3] 또는 [0,3]

        target: Dict[str, Any] = {
            "boxes": boxes,
            "labels": labels,
            "relations": relations,
            "image_id": torch.tensor([idx], dtype=torch.int64),
            "orig_size": torch.tensor([h, w], dtype=torch.int64),
        }

        if "latex" in ginfo:
            target["latex"] = ginfo["latex"]

        # -------------------------
        # 3. transforms (augmentation)
        # -------------------------
        if self.transforms is not None:
            image, target = self.transforms(image, target)

        # -------------------------
        # 4. feature_extractor (DETR/EGTR 전처리)
        # -------------------------
        if self.feature_extractor is not None:
            encoded = self.feature_extractor(
                images=image,
                return_tensors="pt",
            )
            # encoded["pixel_values"]: [1, C, H, W]
            pixel_values = encoded["pixel_values"].squeeze(0)
        else:
            # feature_extractor를 안 쓰는 경우: 직접 ToTensor
            arr = np.array(image).astype("float32") / 255.0  # [H,W,C]
            arr = arr.transpose(2, 0, 1)                     # [C,H,W]
            pixel_values = torch.from_numpy(arr)

        return pixel_values, target
