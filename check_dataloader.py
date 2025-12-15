import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
import numpy as np

from data.crohme import CROHMEDataset
from model.deformable_detr import (
    DeformableDetrConfig,
    DeformableDetrFeatureExtractor,
    DeformableDetrFeatureExtractorWithAugmentorNoCrop,
)

feature_extractor = DeformableDetrFeatureExtractor.from_pretrained(
    "SenseTime/deformable-detr", size=800, max_size=1333
)
feature_extractor_train = (
    DeformableDetrFeatureExtractorWithAugmentorNoCrop.from_pretrained(
        "SenseTime/deformable-detr", size=800, max_size=1333
    )
)

train_dataset = CROHMEDataset(
    data_folder="dataset/crohme",
    feature_extractor=feature_extractor_train,
    split="train",
    num_object_queries=100,
    debug=False,
)

def visualize_dataset_sample(dataset, idx=0):
    """
    DETR Feature Extractor를 통과한 데이터셋의 샘플을 역변환하여 시각화합니다.
    """
    # 1. 데이터셋에서 샘플 로드 (Tensor, Dict)
    pixel_values, target = dataset[idx]
    
    # -------------------------------
    # 2. 이미지 역정규화 (Denormalization)
    # -------------------------------
    # Deformable DETR 기본 ImageNet Mean/Std
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    
    # (Normalization 해제) image = (pixel_values * std) + mean
    img_tensor = pixel_values.detach().cpu() * std + mean
    
    # (C, H, W) -> (H, W, C) 순서 변경 및 Numpy 변환
    img = img_tensor.permute(1, 2, 0).numpy()
    
    # 값 범위를 [0, 1]로 클리핑 (시각화 안전장치)
    img = np.clip(img, 0, 1)
    
    # 이미지 실제 크기 (픽셀 단위 좌표 계산용)
    h, w, _ = img.shape
    
    # -------------------------------
    # 3. Target 정보 추출
    # -------------------------------
    boxes = target['boxes'].detach().cpu()           # (cx, cy, w, h) normalized
    labels = target['class_labels'].detach().cpu()   # Class IDs
    
    # 관계 정보가 있다면 (CROHMEDataset)
    # rel = target.get('rel', None) 
    
    print(f"Sample Index: {idx}")
    print(f"Image Shape: {img.shape}")
    print(f"Number of Objects: {len(boxes)}")
    
    # -------------------------------
    # 4. 시각화 (Matplotlib)
    # -------------------------------
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    ax.imshow(img)
    
    # 색상 팔레트 (클래스별로 다른 색을 쓰면 좋지만 여기선 빨강 통일)
    box_color = 'red'
    text_color = 'black'
    text_bg = 'yellow'
    
    for box, label in zip(boxes, labels):
        # DETR Box Format: (center_x, center_y, width, height) [0 ~ 1]
        cx, cy, bw, bh = box.tolist()
        
        # 상대 좌표 -> 절대 픽셀 좌표 변환
        # 좌상단(x_min, y_min) 좌표 계산: (center - width/2) * image_size
        x_min = (cx - bw / 2) * w
        y_min = (cy - bh / 2) * h
        abs_w = bw * w
        abs_h = bh * h
        
        # Bbox 그리기
        rect = patches.Rectangle(
            (x_min, y_min), abs_w, abs_h, 
            linewidth=2, edgecolor=box_color, facecolor='none'
        )
        ax.add_patch(rect)
        
        # Label 이름 가져오기
        # dataset.id2label에 매핑 정보가 있습니다.
        label_id = label.item()
        label_text = dataset.id2label.get(label_id, str(label_id))
        
        # 텍스트 추가
        ax.text(
            x_min, y_min - 5, 
            label_text, 
            fontsize=11, 
            color=text_color,
            fontweight='bold',
            bbox=dict(facecolor=text_bg, alpha=0.5, edgecolor='none', pad=1)
        )
    
    plt.title(f"Check BBox & Label (Image ID: {target['image_id']})")
    plt.axis('off')
    plt.show()

# ==========================================================
# 실행 (데이터셋 인스턴스 train_dataset이 있다고 가정)
# ==========================================================
# 첫 번째 샘플 확인
visualize_dataset_sample(train_dataset, idx=0)

# 랜덤한 다른 샘플 확인 (예: 5번째)
if len(train_dataset) > 5:
    visualize_dataset_sample(train_dataset, idx=5)