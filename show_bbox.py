#!/usr/bin/env python3
"""
CROHME 데이터셋 시각화 도구
"""

import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from pycocotools.coco import COCO
import random
import os


def visualize_single_image(coco, rel_data, image_id, save_path=None, show_relations=True):
    """단일 이미지의 bbox와 라벨 시각화"""
    
    # 이미지 정보 가져오기
    img_info = coco.imgs[image_id]
    img_path = os.path.join('dataset/crohme/images', img_info['file_name'])
    
    # 이미지 로드
    img = Image.open(img_path)
    
    # Figure 생성
    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(img)
    
    # Annotations 가져오기
    ann_ids = coco.getAnnIds(imgIds=image_id)
    anns = coco.loadAnns(ann_ids)
    
    # Annotation ID → 인덱스 매핑
    ann_id_to_idx = {ann['id']: i for i, ann in enumerate(anns)}
    
    # 색상 맵
    colors = plt.cm.rainbow(range(len(anns)))
    
    # Bbox와 라벨 그리기
    for i, ann in enumerate(anns):
        # Bbox 추출 (COCO 형식: [x, y, width, height])
        x, y, w, h = ann['bbox']
        
        # Category 이름
        category_id = ann['category_id']
        category_name = coco.cats[category_id]['name']
        
        # Rectangle 그리기
        rect = patches.Rectangle(
            (x, y), w, h,
            linewidth=2,
            edgecolor=colors[i],
            facecolor='none'
        )
        ax.add_patch(rect)
        
        # 라벨 텍스트
        ax.text(
            x, y - 5,
            f"{i}: {category_name}",
            color=colors[i],
            fontsize=10,
            fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7)
        )
    
    # 관계 시각화 (옵션)
    if show_relations and rel_data:
        relations = rel_data.get(str(image_id), [])
        
        for subj_ann_id, obj_ann_id, rel_id in relations:
            if subj_ann_id not in ann_id_to_idx or obj_ann_id not in ann_id_to_idx:
                continue
            
            subj_idx = ann_id_to_idx[subj_ann_id]
            obj_idx = ann_id_to_idx[obj_ann_id]
            
            # Subject와 Object의 중심점 계산
            subj_ann = anns[subj_idx]
            obj_ann = anns[obj_idx]
            
            subj_x = subj_ann['bbox'][0] + subj_ann['bbox'][2] / 2
            subj_y = subj_ann['bbox'][1] + subj_ann['bbox'][3] / 2
            obj_x = obj_ann['bbox'][0] + obj_ann['bbox'][2] / 2
            obj_y = obj_ann['bbox'][1] + obj_ann['bbox'][3] / 2
            
            # 화살표 그리기
            ax.annotate(
                '',
                xy=(obj_x, obj_y),
                xytext=(subj_x, subj_y),
                arrowprops=dict(
                    arrowstyle='->',
                    color='red',
                    lw=1.5,
                    alpha=0.6
                )
            )
    
    # 제목
    ax.set_title(
        f"Image {image_id}: {img_info['file_name']}\n"
        f"{len(anns)} symbols, {len(relations) if show_relations and rel_data else 0} relations",
        fontsize=14,
        fontweight='bold'
    )
    ax.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
    
    plt.show()
    plt.close()


def visualize_random_samples(split='train', n_samples=5, show_relations=True, save_dir=None):
    """랜덤 샘플 여러 개 시각화"""
    
    print(f"Loading {split} dataset...")
    coco = COCO(f'dataset/crohme/{split}.json')
    
    # rel.json 로드
    rel_data = None
    if show_relations:
        try:
            with open('dataset/crohme/rel.json', 'r') as f:
                rel_json = json.load(f)
                rel_data = rel_json.get(split, {})
        except Exception as e:
            print(f"Warning: Could not load rel.json: {e}")
    
    # 랜덤 이미지 선택
    image_ids = random.sample(list(coco.imgs.keys()), min(n_samples, len(coco.imgs)))
    
    # 저장 디렉토리 생성
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    
    # 각 이미지 시각화
    for i, image_id in enumerate(image_ids):
        print(f"\nVisualizing image {i+1}/{len(image_ids)}: {image_id}")
        
        save_path = None
        if save_dir:
            save_path = os.path.join(save_dir, f"{split}_sample_{i+1}_id_{image_id}.png")
        
        visualize_single_image(coco, rel_data, image_id, save_path, show_relations)


def print_image_details(split='train', image_id=None, n_random=3):
    """이미지 상세 정보 텍스트로 출력"""
    
    print(f"\n{'='*60}")
    print(f"Image Details - {split}")
    print(f"{'='*60}")
    
    coco = COCO(f'dataset/crohme/{split}.json')
    
    # rel.json 로드
    try:
        with open('dataset/crohme/rel.json', 'r') as f:
            rel_json = json.load(f)
            rel_data = rel_json.get(split, {})
            rel_categories = rel_json.get('rel_categories', [])
    except Exception as e:
        print(f"Warning: Could not load rel.json: {e}")
        rel_data = {}
        rel_categories = []
    
    # 이미지 ID 선택
    if image_id is None:
        image_ids = random.sample(list(coco.imgs.keys()), min(n_random, len(coco.imgs)))
    else:
        image_ids = [image_id]
    
    for img_id in image_ids:
        print(f"\n{'='*60}")
        img_info = coco.imgs[img_id]
        print(f"Image ID: {img_id}")
        print(f"Filename: {img_info['file_name']}")
        print(f"Size: {img_info['width']} x {img_info['height']}")
        
        # Annotations
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        
        print(f"\nAnnotations ({len(anns)} symbols):")
        print(f"{'Idx':<5} {'Ann ID':<8} {'Symbol':<15} {'BBox (x,y,w,h)':<30}")
        print("-" * 60)
        
        ann_id_to_idx = {ann['id']: i for i, ann in enumerate(anns)}
        
        for i, ann in enumerate(anns):
            category_name = coco.cats[ann['category_id']]['name']
            bbox_str = f"({ann['bbox'][0]:.1f}, {ann['bbox'][1]:.1f}, {ann['bbox'][2]:.1f}, {ann['bbox'][3]:.1f})"
            
            print(f"{i:<5} {ann['id']:<8} {category_name:<15} {bbox_str:<30}")
        
        # Relations
        relations = rel_data.get(str(img_id), [])
        if relations:
            print(f"\nRelations ({len(relations)}):")
            print(f"{'#':<5} {'Subject':<15} {'Relation':<15} {'Object':<15}")
            print("-" * 60)
            
            for i, (subj_id, obj_id, rel_id) in enumerate(relations[:20]):  # 처음 20개만
                if subj_id not in ann_id_to_idx or obj_id not in ann_id_to_idx:
                    continue
                
                subj_cat = coco.cats[anns[ann_id_to_idx[subj_id]]['category_id']]['name']
                obj_cat = coco.cats[anns[ann_id_to_idx[obj_id]]['category_id']]['name']
                rel_name = rel_categories[rel_id] if rel_id < len(rel_categories) else f"rel_{rel_id}"
                
                print(f"{i+1:<5} {subj_cat:<15} {rel_name:<15} {obj_cat:<15}")
            
            if len(relations) > 20:
                print(f"... and {len(relations) - 20} more relations")


def check_dataset_quality(split='train'):
    """데이터셋 품질 체크"""
    
    print(f"\n{'='*60}")
    print(f"Dataset Quality Check - {split}")
    print(f"{'='*60}")
    
    coco = COCO(f'dataset/crohme/{split}.json')
    
    # 통계
    total_images = len(coco.imgs)
    total_annotations = len(coco.anns)
    total_categories = len(coco.cats)
    
    print(f"\nBasic Statistics:")
    print(f"  Images: {total_images}")
    print(f"  Annotations: {total_annotations}")
    print(f"  Categories: {total_categories}")
    print(f"  Avg annotations per image: {total_annotations/total_images:.2f}")
    
    # Bbox 품질 체크
    invalid_bboxes = 0
    bbox_sizes = []
    
    for ann_id, ann in coco.anns.items():
        x, y, w, h = ann['bbox']
        
        if w <= 0 or h <= 0 or x < 0 or y < 0:
            invalid_bboxes += 1
        else:
            bbox_sizes.append((w, h))
    
    print(f"\nBbox Quality:")
    print(f"  Invalid bboxes: {invalid_bboxes} ({invalid_bboxes/total_annotations*100:.2f}%)")
    
    if bbox_sizes:
        import numpy as np
        widths = [s[0] for s in bbox_sizes]
        heights = [s[1] for s in bbox_sizes]
        
        print(f"  Width - min: {min(widths):.1f}, max: {max(widths):.1f}, avg: {np.mean(widths):.1f}")
        print(f"  Height - min: {min(heights):.1f}, max: {max(heights):.1f}, avg: {np.mean(heights):.1f}")
    
    # Category 분포
    from collections import Counter
    category_counts = Counter(ann['category_id'] for ann in coco.anns.values())
    
    print(f"\nTop 10 Most Frequent Symbols:")
    for cat_id, count in category_counts.most_common(10):
        cat_name = coco.cats[cat_id]['name']
        print(f"  {cat_name}: {count}")
    
    # Relations 체크
    try:
        with open('dataset/crohme/rel.json', 'r') as f:
            rel_json = json.load(f)
            rel_data = rel_json.get(split, {})
            rel_categories = rel_json.get('rel_categories', [])
        
        total_relations = sum(len(rels) for rels in rel_data.values())
        images_with_relations = len([img_id for img_id, rels in rel_data.items() if len(rels) > 0])
        
        print(f"\nRelation Statistics:")
        print(f"  Total relations: {total_relations}")
        print(f"  Images with relations: {images_with_relations}/{total_images}")
        print(f"  Avg relations per image: {total_relations/total_images:.2f}")
        print(f"  Relation types: {len(rel_categories)}")
        
        # Relation 분포
        rel_type_counts = Counter()
        for rels in rel_data.values():
            for _, _, rel_id in rels:
                rel_type_counts[rel_id] += 1
        
        print(f"\nRelation Type Distribution:")
        for rel_id, count in rel_type_counts.most_common():
            rel_name = rel_categories[rel_id] if rel_id < len(rel_categories) else f"rel_{rel_id}"
            print(f"  {rel_name}: {count}")
        
    except Exception as e:
        print(f"\nWarning: Could not analyze relations: {e}")


def interactive_viewer():
    """대화형 뷰어"""
    
    print("="*60)
    print("CROHME Dataset Interactive Viewer")
    print("="*60)
    
    while True:
        print("\nOptions:")
        print("  1. View random samples")
        print("  2. View specific image")
        print("  3. Print image details")
        print("  4. Check dataset quality")
        print("  5. Exit")
        
        choice = input("\nEnter choice (1-5): ").strip()
        
        if choice == '1':
            split = input("Split (train/valid/test) [train]: ").strip() or 'train'
            n = input("Number of samples [5]: ").strip()
            n = int(n) if n else 5
            show_rel = input("Show relations? (y/n) [y]: ").strip().lower() != 'n'
            save = input("Save to directory? (path or empty): ").strip() or None
            
            visualize_random_samples(split, n, show_rel, save)
        
        elif choice == '2':
            split = input("Split (train/valid/test) [train]: ").strip() or 'train'
            img_id = input("Image ID: ").strip()
            
            if img_id:
                coco = COCO(f'dataset/{split}.json')
                with open('dataset/rel.json', 'r') as f:
                    rel_json = json.load(f)
                    rel_data = rel_json.get(split, {})
                
                visualize_single_image(coco, rel_data, int(img_id), show_relations=True)
        
        elif choice == '3':
            split = input("Split (train/valid/test) [train]: ").strip() or 'train'
            img_id = input("Image ID (empty for random): ").strip()
            
            if img_id:
                print_image_details(split, int(img_id))
            else:
                n = input("Number of random images [3]: ").strip()
                n = int(n) if n else 3
                print_image_details(split, n_random=n)
        
        elif choice == '4':
            split = input("Split (train/valid/test) [train]: ").strip() or 'train'
            check_dataset_quality(split)
        
        elif choice == '5':
            print("Goodbye!")
            break
        
        else:
            print("Invalid choice!")


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1:
        # 커맨드라인 모드
        command = sys.argv[1]
        
        if command == 'random':
            split = sys.argv[2] if len(sys.argv) > 2 else 'train'
            n = int(sys.argv[3]) if len(sys.argv) > 3 else 5
            visualize_random_samples(split, n, show_relations=True)
        
        elif command == 'details':
            split = sys.argv[2] if len(sys.argv) > 2 else 'train'
            print_image_details(split)
        
        elif command == 'quality':
            split = sys.argv[2] if len(sys.argv) > 2 else 'train'
            check_dataset_quality(split)
        
        else:
            print("Usage:")
            print("  python visualize_crohme.py random [split] [n_samples]")
            print("  python visualize_crohme.py details [split]")
            print("  python visualize_crohme.py quality [split]")
    else:
        # 대화형 모드
        interactive_viewer()