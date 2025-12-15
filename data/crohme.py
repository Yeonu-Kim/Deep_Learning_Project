# References:
# - https://github.com/pytorch/vision/blob/13b35ff/references/detection/coco_utils.py
# - https://github.com/suprosanna/relationformer/blob/scene_graph/datasets/get_dataset_counts.py

import json
import os

import numpy as np
import torch
import torchvision
from tqdm import tqdm


class CROHMEDetection(torchvision.datasets.CocoDetection):
    """CROHME 데이터세트를 위한 기본 Detection 클래스"""
    
    def __init__(self, data_folder, feature_extractor, split, debug=False):
        """
        Args:
            data_folder: 데이터셋 폴더 경로 (dataset/)
            feature_extractor: DETR feature extractor
            split: 'train', 'valid', 'test'
            debug: 디버그 모드 (작은 subset만 사용)
        """
        ann_file = os.path.join(data_folder, f"{split}.json")
        img_folder = os.path.join(data_folder, "images")
        super(CROHMEDetection, self).__init__(img_folder, ann_file)
        self.feature_extractor = feature_extractor
        self.split = split
        self.debug = debug

        if self.debug:
            self.ids = self.ids[:8]

    def __getitem__(self, idx):
        # PIL 이미지와 COCO 형식 target 읽기
        img, target = super(CROHMEDetection, self).__getitem__(idx)

        # 이미지와 target 전처리 (DETR 형식으로 변환, 리사이징 + 정규화)
        image_id = self.ids[idx]
        target = {"image_id": image_id, "annotations": target}
        encoding = self.feature_extractor(
            images=img, annotations=target, return_tensors="pt"
        )
        pixel_values = encoding["pixel_values"].squeeze()  # batch dimension 제거
        target = encoding["labels"][0]  # batch dimension 제거
        
        return pixel_values, target

    def __len__(self):
        return len(self.ids)
           


class CROHMEDataset(CROHMEDetection):
    """관계 정보를 포함한 CROHME 데이터세트 클래스"""
    
    def __init__(
        self, 
        data_folder, 
        feature_extractor, 
        split, 
        num_object_queries=100, 
        debug=False
    ):
        """
        Args:
            data_folder: 데이터셋 폴더 경로 (dataset/)
            feature_extractor: DETR feature extractor
            split: 'train', 'valid', 'test'
            num_object_queries: 최대 객체 쿼리 수 (DETR의 num_queries)
            debug: 디버그 모드
        """
        super(CROHMEDataset, self).__init__(data_folder, feature_extractor, split, debug)
        
        # 관계 데이터 로드
        rel_file = os.path.join(data_folder, "rel.json")
        with open(rel_file, "r") as f:
            rel = json.load(f)
        
        self.rel = rel[split]
        self.rel_categories = {i: name for i, name in enumerate(rel["rel_categories"])}
        self.num_object_queries = num_object_queries
        self.num_rel_labels = len(self.rel_categories)
        self.num_classes = len(self.coco.cats)
        
        # id2class 매핑 생성 (VG와 유사)
        # COCO cats는 {id: {'id': int, 'name': str, ...}} 형태
        self.id2label = {cat['id']: cat['name'] for cat in self.coco.cats.values()}
        
        # class2id 매핑도 추가 (편의성)
        self.label2id = {cat['name']: cat['id'] for cat in self.coco.cats.values()}
        
        # rel_id2name 매핑
        self.rel_id2name = {i: name for i, name in enumerate(rel["rel_categories"])}
        self.rel_name2id = {name: i for i, name in enumerate(rel["rel_categories"])}
        
        print(f"CROHME {split} dataset initialized:")
        print(f"  - Images: {len(self.ids)}")
        print(f"  - Symbol classes: {len(self.coco.cats)}")
        print(f"  - Relation types: {self.num_rel_labels}")
        if len(self.rel_categories) <= 20:
            print(f"  - Relation categories: {self.rel_categories}")
        else:
            print(f"  - Relation categories (first 10): {self.rel_categories[:10]}...")

    # -------------------------------------------------------------
    def __getitem__(self, idx):
        # PIL 이미지와 COCO 형식 target 읽기
        img, target = super(CROHMEDetection, self).__getitem__(idx)

        # 이미지 ID와 관계 데이터 가져오기
        image_id = self.ids[idx]
        ann_id_to_idx = {ann['id']: i for i, ann in enumerate(target)}
        
        # 관계 리스트 가져오기
        rel_list = self.rel.get(str(image_id), [])
        
        # 이미지와 target 전처리
        target_dict = {"image_id": image_id, "annotations": target}
        encoding = self.feature_extractor(
            images=img, annotations=target_dict, return_tensors="pt"
        )
        pixel_values = encoding["pixel_values"].squeeze()
        target = encoding["labels"][0]
        
        # 관계 텐서 생성
        if rel_list:
            rel = np.array(rel_list)
            target["rel"] = self._get_rel_tensor(rel, ann_id_to_idx)
        else:
            target["rel"] = torch.zeros(
                [self.num_object_queries, self.num_object_queries, self.num_rel_labels]
            )
        
        return pixel_values, target

    def _get_rel_tensor(self, rel_tensor, ann_id_to_idx):
        """
        관계 리스트를 텐서로 변환
        
        Args:
            rel_tensor: [[subj_ann_id, obj_ann_id, rel_id], ...] 형태의 numpy array
            ann_id_to_idx: annotation ID -> 배열 인덱스 매핑 딕셔너리
        
        Returns:
            rel: [num_object_queries, num_object_queries, num_rel_labels] 텐서
        """
        rel = torch.zeros(
            [self.num_object_queries, self.num_object_queries, self.num_rel_labels]
        )
        
        for subj_ann_id, obj_ann_id, rel_id in rel_tensor:
            # Annotation ID를 배열 인덱스로 변환
            if subj_ann_id not in ann_id_to_idx or obj_ann_id not in ann_id_to_idx:
                continue
            
            subj_idx = ann_id_to_idx[subj_ann_id]
            obj_idx = ann_id_to_idx[obj_ann_id]
            
            # 인덱스가 num_object_queries 범위 내인지 확인
            if subj_idx < self.num_object_queries and obj_idx < self.num_object_queries:
                rel[subj_idx, obj_idx, rel_id] = 1.0
        
        return rel
    
    def get_symbol_name(self, category_id):
        """Category ID로 심볼 이름 가져오기"""
        return self.id2label.get(category_id, f"Unknown({category_id})")
    
    def get_relation_name(self, relation_id):
        """Relation ID로 관계 이름 가져오기"""
        return self.rel_id2name.get(relation_id, f"Unknown({relation_id})")


def crohme_get_statistics(train_data, must_overlap=True):
    """
    모든 관계의 빈도수 계산. P(rel | o1, o2)를 직접 모델링하는 데 사용
    
    Args:
        train_data: CROHMEDataset 인스턴스
        must_overlap: 사용되지 않음 (VG 호환성을 위해 유지)
    
    Returns:
        fg_matrix: [num_classes+1, num_classes+1, num_predicates] 형태의 빈도 행렬
    """
    num_classes = len(train_data.coco.cats)
    num_predicates = len(train_data.rel_categories)

    print(f"\nComputing statistics...")
    print(f"  - Number of symbol classes: {num_classes}")
    print(f"  - Number of relation types: {num_predicates}")

    # Foreground matrix 초기화
    fg_matrix = np.zeros(
        (num_classes + 1, num_classes + 1, num_predicates),
        dtype=np.int64,
    )

    rel = train_data.rel
    
    for idx in tqdm(range(len(train_data)), desc="Computing relation statistics"):
        image_id = train_data.ids[idx]

        # Ground truth annotations 로드
        target = train_data.coco.loadAnns(train_data.coco.getAnnIds(image_id))
        
        if not target:
            continue
        
        # Annotation ID → 배열 인덱스 매핑 생성
        ann_id_to_idx = {ann['id']: i for i, ann in enumerate(target)}
        
        # Category IDs 추출
        gt_classes = np.array([ann['category_id'] for ann in target])
        
        # 관계 리스트 가져오기
        rel_list = rel.get(str(image_id), [])
        
        if not rel_list:
            continue
        
        # 관계 인덱스: [subject_ann_id, object_ann_id, relation_id]
        gt_relations = np.array(rel_list, dtype="int64")
        
        # 각 관계에 대해 빈도 카운트
        for subj_ann_id, obj_ann_id, rel_id in gt_relations:
            # Annotation ID를 배열 인덱스로 변환
            if subj_ann_id not in ann_id_to_idx or obj_ann_id not in ann_id_to_idx:
                continue
            
            subj_idx = ann_id_to_idx[subj_ann_id]
            obj_idx = ann_id_to_idx[obj_ann_id]
            
            # Category ID 가져오기
            subj_category = gt_classes[subj_idx]
            obj_category = gt_classes[obj_idx]
            
            # 빈도 카운트
            fg_matrix[subj_category, obj_category, rel_id] += 1

    print(f"\nStatistics computed:")
    print(f"  - Total relations: {fg_matrix.sum()}")
    print(f"  - Non-zero entries: {(fg_matrix > 0).sum()}")
    
    # 가장 빈번한 관계 출력
    top_relations = []
    for s in range(num_classes + 1):
        for o in range(num_classes + 1):
            for r in range(num_predicates):
                if fg_matrix[s, o, r] > 0:
                    top_relations.append((fg_matrix[s, o, r], s, o, r))
    
    if top_relations:
        top_relations.sort(reverse=True)
        print(f"\n  Top 10 most frequent relations:")
        for count, s, o, r in top_relations[:10]:
            subj_name = train_data.get_symbol_name(s) if s < num_classes else "Unknown"
            obj_name = train_data.get_symbol_name(o) if o < num_classes else "Unknown"
            rel_name = train_data.get_relation_name(r)
            print(f"    {subj_name} --[{rel_name}]--> {obj_name}: {count}")

    return fg_matrix
