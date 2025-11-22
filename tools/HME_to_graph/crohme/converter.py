"""
CROHME Converter Module
InkML 파싱, 이미지 렌더링, 그래프 생성을 통합 관리하는 클래스
"""

import xml.etree.ElementTree as ET
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np

from .generator.image_generator import InkMLImageRenderer
from .generator.graph_generator import GraphGenerator


class CROHMEConverter:
    """
    CROHME 데이터셋의 InkML 파일을 처리하는 통합 클래스
    - InkML 파싱
    - 이미지 렌더링 (InkMLImageRenderer 사용)
    - 그래프 생성 (GraphGenerator 사용)
    """
    
    def __init__(self,
                 latex_data,
                 inkml_dir: Optional[str] = None,
                 image_size: Tuple[int, int] = (400, 200),
                 line_width: int = 2,
                 padding: int = 10,
                 ):
        """
        Args:
            latex_data: 수식 관계 카테고리 (필수 입력)
            inkml_dir: InkML 파일들이 있는 디렉토리 (선택)
            image_size: 렌더링할 이미지 크기
            line_width: 획 두께
            padding: 이미지 여백
        """
        self.inkml_dir = Path(inkml_dir) if inkml_dir else None
        self.inkml_files = []
        
        if self.inkml_dir and self.inkml_dir.exists():
            self.inkml_files = list(self.inkml_dir.glob('*.inkml'))
            print(f"총 {len(self.inkml_files)}개의 InkML 파일 발견")
        
        # 이미지 렌더러 초기화
        self.image_renderer = InkMLImageRenderer(
            default_image_size=image_size,
            default_line_width=line_width,
            default_padding=padding
        )
        
        # 그래프 생성기 초기화
        self.graph_generator = GraphGenerator(latex_data=latex_data)
        
        # 네임스페이스
        self.ns = {
            'ink': 'http://www.w3.org/2003/InkML',
            'math': 'http://www.w3.org/1998/Math/MathML'
        }
    
    def parse_inkml(self, inkml_path: str) -> Dict:
        """
        InkML 파일을 파싱하여 모든 정보를 추출합니다.
        
        Args:
            inkml_path: InkML 파일 경로
        
        Returns:
            {
                'strokes': List[List[Tuple[float, float]]],
                'annotations': Dict,
                'symbol_annotations': List[Dict],
                'mathml': str,
                'file_name': str
            }
        """
        inkml_path = Path(inkml_path)
        tree = ET.parse(inkml_path)
        root = tree.getroot()
        
        # 1. Annotations 추출
        annotations = {}
        for annot in root.findall('ink:annotation', self.ns):
            annot_type = annot.get('type')
            annotations[annot_type] = annot.text
        
        # 2. Traces (획) 추출
        strokes = []
        trace_dict = {}
        
        for i, trace in enumerate(root.findall('ink:trace', self.ns)):
            trace_id = trace.get('id')
            coords_text = trace.text.strip() if trace.text else ""
            
            points = []
            for coord_pair in coords_text.split(','):
                coord_pair = coord_pair.strip()
                if coord_pair:
                    parts = coord_pair.split()
                    if len(parts) == 2:
                        x, y = map(float, parts)
                        points.append((x, y))
            
            strokes.append(points)
            trace_dict[trace_id] = i
        
        # 3. Symbol-level Ground Truth 추출
        symbol_annotations = []
        
        for trace_group in root.findall('.//ink:traceGroup', self.ns):
            for symbol_group in trace_group.findall('ink:traceGroup', self.ns):
                symbol_info = {}
                
                truth_annot = symbol_group.find('ink:annotation[@type="truth"]', self.ns)
                if truth_annot is not None:
                    symbol_info['label'] = truth_annot.text
                
                trace_refs = []
                for trace_view in symbol_group.findall('ink:traceView', self.ns):
                    trace_ref = trace_view.get('traceDataRef')
                    if trace_ref in trace_dict:
                        trace_refs.append(trace_dict[trace_ref])
                
                symbol_info['trace_indices'] = trace_refs
                
                annot_xml = symbol_group.find('ink:annotationXML', self.ns)
                if annot_xml is not None:
                    symbol_info['mathml_ref'] = annot_xml.get('href')
                
                if symbol_info:
                    symbol_annotations.append(symbol_info)
        
        # 4. MathML 구조 추출
        mathml_root = root.find('.//ink:annotationXML[@type="truth"]', self.ns)
        mathml_str = ""
        if mathml_root is not None:
            mathml_element = mathml_root.find('math:math', self.ns)
            if mathml_element is not None:
                mathml_str = ET.tostring(mathml_element, encoding='unicode')
        
        return {
            'strokes': strokes,
            'annotations': annotations,
            'symbol_annotations': symbol_annotations,
            'mathml': mathml_str,
            'file_name': inkml_path.name
        }
    
    def convert_single_inkml(self,
                            inkml_path: str,
                            output_image_path: Optional[str] = None,
                            image_size: Optional[Tuple[int, int]] = None) -> Dict:
        """
        단일 InkML 파일을 파싱하고 이미지로 렌더링합니다.
        
        Args:
            inkml_path: InkML 파일 경로
            output_image_path: 이미지 저장 경로 (None이면 저장 안함)
            image_size: 이미지 크기 (None이면 기본값)
        
        Returns:
            파싱 결과 딕셔너리 + 'image' 키 추가
        """
        data = self.parse_inkml(inkml_path)
        
        # 이미지 렌더링
        img = self.image_renderer.render_to_pil(
            data['strokes'],
            image_size=image_size
        )
        
        data['image'] = img
        
        # 이미지 저장
        if output_image_path and img:
            self.image_renderer.save_image(
                data['strokes'],
                output_image_path,
                image_size=image_size
            )
        
        return data
    
    def convert_dataset_to_images(self,
                                  output_dir: str,
                                  image_size: Optional[Tuple[int, int]] = None,
                                  save_metadata: bool = True) -> List[Dict]:
        """
        디렉토리의 모든 InkML을 이미지로 변환합니다.
        
        Args:
            output_dir: 출력 디렉토리
            image_size: 이미지 크기
            save_metadata: 메타데이터 JSON 저장 여부
        
        Returns:
            메타데이터 리스트
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        metadata_list = []
        
        for i, inkml_path in enumerate(self.inkml_files):
            try:
                data = self.parse_inkml(inkml_path)
                
                # 이미지 생성
                img_filename = f"{inkml_path.stem}.png"
                img_path = output_dir / img_filename
                
                success = self.image_renderer.save_image(
                    data['strokes'],
                    str(img_path),
                    image_size=image_size
                )
                
                if not success:
                    continue
                
                # 메타데이터
                metadata = {
                    'image_file': img_filename,
                    'inkml_file': inkml_path.name,
                    'formula': data['annotations'].get('truth', ''),
                    'writer': data['annotations'].get('writer', ''),
                    'num_strokes': len(data['strokes']),
                    'num_symbols': len(data['symbol_annotations']),
                    'symbols': [s['label'] for s in data['symbol_annotations']]
                }
                metadata_list.append(metadata)
                
                if (i + 1) % 100 == 0:
                    print(f"진행중: {i+1}/{len(self.inkml_files)}")
            
            except Exception as e:
                print(f"오류 ({inkml_path.name}): {e}")
                continue
        
        # 메타데이터 JSON 저장
        if save_metadata:
            metadata_path = output_dir / 'metadata.json'
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata_list, f, indent=2, ensure_ascii=False)
            print(f"\n메타데이터 저장: {metadata_path}")
        
        print(f"완료! {len(metadata_list)}개 파일 변환")
        return metadata_list
    
    def convert_annotation_to_graph(self,
                                    annotation_file: str,
                                    output_file: str,
                                    split_name: str = 'train',
                                    output_format: str = 'detailed') -> Dict:
        """
        어노테이션 파일을 그래프로 변환합니다.
        
        Args:
            annotation_file: 입력 어노테이션 파일
            output_file: 출력 JSON 파일
            split_name: 데이터셋 스플릿 이름
            output_format: 'simple' 또는 'detailed'
        
        Returns:
            변환된 그래프 데이터
        """
        self.graph_generator.save_graph_json(
            annotation_file,
            output_file,
            split_name,
            output_format
        )
        
        with open(output_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def analyze_annotation_statistics(self, annotation_file: str):
        """
        어노테이션 파일의 통계를 분석하고 출력합니다.
        
        Args:
            annotation_file: 어노테이션 파일 경로
        """
        stats = self.graph_generator.analyze_graph_statistics(annotation_file)
        self.graph_generator.print_statistics(stats)
        return stats
    
    def process_crohme_dataset(self,
                              base_dir: str,
                              output_dir: str,
                              splits: Optional[List[str]] = None,
                              image_size: Optional[Tuple[int, int]] = None):
        """
        CROHME 표준 데이터셋 구조를 자동으로 처리합니다.
        
        지원하는 구조:
        1. base_dir/inkml/{split}/*.inkml + base_dir/annotation/{prefix}_{split}.txt
        2. base_dir/{split}/*.inkml + base_dir/{split}/{split}.txt
        
        Args:
            base_dir: 데이터셋 루트 폴더
            output_dir: 출력 폴더
            splits: 처리할 split 리스트 (None이면 자동 감지)
            image_size: 이미지 크기
        
        Example:
            # 구조 1: data/crohme/inkml/train/*.inkml + data/crohme/annotation/crohme2019_train.txt
            converter.process_crohme_dataset(
                base_dir='data/crohme',
                output_dir='output',
                splits=['train', 'valid', 'test']
            )
        """
        base_dir = Path(base_dir)
        output_dir = Path(output_dir)
        
        if not base_dir.exists():
            raise FileNotFoundError(f"데이터셋 폴더를 찾을 수 없습니다: {base_dir}")
        
        # Split 자동 감지
        if splits is None:
            splits = self._detect_splits(base_dir)
            if not splits:
                raise ValueError(f"데이터셋 구조를 자동 감지할 수 없습니다: {base_dir}")
            print(f"자동 감지된 splits: {splits}")
        
        # 각 split 처리
        for split in splits:
            print(f"\n{'='*60}")
            print(f"Processing {split.upper()} set...")
            print('='*60)
            
            # InkML 폴더 찾기
            inkml_dir = self._find_inkml_dir(base_dir, split)
            if not inkml_dir:
                print(f"⚠️  {split} InkML 폴더를 찾을 수 없습니다. 건너뜁니다.")
                continue
            
            # Annotation 폴더 찾기
            annotation_dir = self._find_annotation_dir(base_dir, split)
            if not annotation_dir:
                print(f"⚠️  {split} 어노테이션 폴더를 찾을 수 없습니다. 건너뜁니다.")
                continue
            
            # 출력 폴더
            split_output_dir = output_dir / split
            
            # 파이프라인 실행
            try:
                self.process_full_pipeline(
                    inkml_dir=str(inkml_dir),
                    annotation_dir=str(annotation_dir),
                    output_dir=str(split_output_dir),
                    split_name=split,
                    image_size=image_size
                )
            except Exception as e:
                print(f"❌ {split} 처리 중 오류: {e}")
                continue
        
        print(f"\n{'='*60}")
        print(f"모든 split 처리 완료!")
        print(f"출력 폴더: {output_dir}")
        print('='*60)
    
    def _detect_splits(self, base_dir: Path) -> List[str]:
        """데이터셋에서 사용 가능한 split을 자동 감지합니다."""
        splits = set()
        
        # inkml 폴더에서 감지
        inkml_base = base_dir / 'inkml'
        if inkml_base.exists():
            for subdir in inkml_base.iterdir():
                if subdir.is_dir():
                    splits.add(subdir.name)
        
        # annotation 폴더에서 감지
        annotation_base = base_dir / 'annotation'
        if annotation_base.exists():
            for txt_file in annotation_base.glob('*.txt'):
                # crohme2019_train.txt -> train
                name = txt_file.stem
                for possible_split in ['train', 'valid', 'val', 'test']:
                    if possible_split in name.lower():
                        # valid/val 통일
                        if possible_split == 'val':
                            splits.add('valid')
                        elif possible_split == 'valid':
                            splits.add('valid')
                        else:
                            splits.add(possible_split)
        
        # 루트 레벨에서 감지
        for subdir in base_dir.iterdir():
            if subdir.is_dir() and subdir.name in ['train', 'valid', 'val', 'test']:
                if subdir.name == 'val':
                    splits.add('valid')
                else:
                    splits.add(subdir.name)
        
        return sorted(list(splits))
    
    def _find_inkml_dir(self, base_dir: Path, split: str) -> Optional[Path]:
        """InkML 폴더를 찾습니다."""
        # split 변형 (valid <-> val)
        split_variations = [split]
        if split == 'valid':
            split_variations.append('val')
        elif split == 'val':
            split_variations.append('valid')
        
        # 가능한 경로들
        possible_paths = []
        for split_var in split_variations:
            possible_paths.extend([
                base_dir / 'inkml' / split_var,
                base_dir / split_var,
                base_dir / 'handwriting' / split_var,
                base_dir / 'data' / split_var,
            ])
        
        for path in possible_paths:
            if path.exists() and path.is_dir():
                # InkML 파일이 있는지 확인
                if list(path.glob('*.inkml')):
                    return path
        
        return None
    
    def _find_annotation_dir(self, base_dir: Path, split: str) -> Optional[Path]:
        """Annotation 폴더를 찾습니다."""
        # 가능한 경로들
        possible_paths = [
            base_dir / 'annotation',
            base_dir / 'annotations',
            base_dir / 'gt',
            base_dir / 'groundtruth',
            base_dir / 'labels',
            base_dir / split,  # split 폴더 안에 있을 수도
        ]
        
        for path in possible_paths:
            if path.exists() and path.is_dir():
                # .txt 파일이 있는지 확인
                if list(path.glob('*.txt')):
                    return path
        
        return None
    
    def process_full_pipeline(self,
                             inkml_dir: str,
                             annotation_dir: str,
                             output_dir: str,
                             split_name: str = 'train',
                             image_size: Optional[Tuple[int, int]] = None):
        """
        전체 파이프라인: 손글씨(InkML) + 정답(Annotation) → 이미지 + 그래프 JSON
        
        Args:
            inkml_dir: 손글씨 InkML 파일들이 있는 폴더
            annotation_dir: 정답 어노테이션 파일들이 있는 폴더
            output_dir: 출력 폴더
            split_name: 데이터셋 스플릿 이름 (train/val/test)
            image_size: 이미지 크기 (None이면 기본값)
        """
        inkml_dir = Path(inkml_dir)
        annotation_dir = Path(annotation_dir)
        output_dir = Path(output_dir)
        
        # 입력 폴더 확인
        if not inkml_dir.exists():
            raise FileNotFoundError(f"손글씨 폴더를 찾을 수 없습니다: {inkml_dir}")
        if not annotation_dir.exists():
            raise FileNotFoundError(f"정답 폴더를 찾을 수 없습니다: {annotation_dir}")
        
        # 출력 폴더 생성
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"=== CROHME 전체 변환 파이프라인 시작 ===")
        print(f"손글씨 폴더: {inkml_dir}")
        print(f"정답 폴더: {annotation_dir}")
        print(f"출력 폴더: {output_dir}")
        print(f"Split: {split_name}")
        print()
        
        # InkML 파일 목록 로드
        inkml_files = list(inkml_dir.glob('*.inkml'))
        if not inkml_files:
            print(f"경고: {inkml_dir}에 InkML 파일이 없습니다.")
            return
        
        print(f"발견된 InkML 파일: {len(inkml_files)}개")
        
        # 어노테이션 파일 찾기
        # 다양한 네이밍 패턴 지원
        split_name_variations = [
            split_name,                          # 'train'
            split_name.replace('val', 'valid'),  # 'valid' 
            split_name.replace('valid', 'val'),  # 'val'
        ]
        
        possible_annotation_files = []
        for split_var in split_name_variations:
            possible_annotation_files.extend([
                annotation_dir / f'{split_var}.txt',                    # train.txt
                annotation_dir / f'crohme2019_{split_var}.txt',        # crohme2019_train.txt
                annotation_dir / f'{split_var}_annotation.txt',        # train_annotation.txt
                annotation_dir / f'{split_var}_annotations.txt',       # train_annotations.txt
            ])
        
        # 일반적인 파일명도 추가
        possible_annotation_files.extend([
            annotation_dir / 'annotation.txt',
            annotation_dir / 'annotations.txt',
        ])
        
        annotation_file = None
        for possible_file in possible_annotation_files:
            if possible_file.exists():
                annotation_file = possible_file
                break
        
        # 어노테이션 파일이 없으면 split_name을 포함하는 .txt 파일 찾기
        if annotation_file is None:
            txt_files = list(annotation_dir.glob('*.txt'))
            for txt_file in txt_files:
                # 파일명에 split_name이 포함되어 있으면 사용
                filename_lower = txt_file.stem.lower()
                for split_var in split_name_variations:
                    if split_var.lower() in filename_lower:
                        annotation_file = txt_file
                        print(f"매칭된 어노테이션 파일 사용: {annotation_file.name}")
                        break
                if annotation_file:
                    break
        
        # 여전히 없으면 첫 번째 .txt 파일 사용
        if annotation_file is None:
            txt_files = list(annotation_dir.glob('*.txt'))
            if txt_files:
                annotation_file = txt_files[0]
                print(f"⚠️  정확한 매칭 실패, 첫 번째 .txt 파일 사용: {annotation_file.name}")
            else:
                raise FileNotFoundError(f"정답 폴더에 어노테이션 파일(.txt)이 없습니다: {annotation_dir}")
        else:
            print(f"어노테이션 파일: {annotation_file.name}")
        
        print()
        
        # 1. 이미지 생성
        print("[1/3] 이미지 생성 중...")
        # 임시로 inkml_files 설정 (기존 __init__에서 설정된 것 덮어쓰기)
        original_inkml_files = self.inkml_files
        self.inkml_files = inkml_files
        
        image_dir = output_dir / 'images'
        self.convert_dataset_to_images(
            str(image_dir),
            image_size=image_size
        )
        
        # 원래 상태로 복원
        self.inkml_files = original_inkml_files
        print()
        
        # 2. 그래프 생성
        print("[2/3] 그래프 생성 중...")
        graph_file = output_dir / f'{split_name}_graphs.json'
        self.convert_annotation_to_graph(
            str(annotation_file),
            str(graph_file),
            split_name,
            output_format='detailed'
        )
        print()
        
        print(f"=== 완료 ===")
        print(f"출력 폴더: {output_dir}")
        print(f"  - 이미지: {image_dir}")
        print(f"  - 그래프: {graph_file}")
        print()
