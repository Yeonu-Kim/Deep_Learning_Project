import numpy as np
from PIL import Image, ImageDraw
from typing import List, Tuple, Optional


class InkMLImageRenderer:
    """InkML 손글씨를 이미지로 렌더링하는 클래스"""
    
    def __init__(self, 
                 default_image_size: Tuple[int, int] = (400, 200),
                 default_line_width: int = 2,
                 default_padding: int = 10,
                 background_color: str = 'white',
                 stroke_color: str = 'black'):
        """
        Args:
            default_image_size: 기본 이미지 크기 (width, height)
            default_line_width: 기본 선 두께
            default_padding: 기본 여백
            background_color: 배경색
            stroke_color: 획 색상
        """
        self.default_image_size = default_image_size
        self.default_line_width = default_line_width
        self.default_padding = default_padding
        self.background_color = background_color
        self.stroke_color = stroke_color
    
    def render_to_pil(self, 
                      strokes: List[List[Tuple[float, float]]],
                      image_size: Optional[Tuple[int, int]] = None,
                      line_width: Optional[int] = None,
                      padding: Optional[int] = None,
                      background: Optional[str] = None,
                      stroke_color: Optional[str] = None) -> Optional[Image.Image]:
        """
        획 데이터를 PIL Image로 렌더링합니다.
        
        Args:
            strokes: 획 리스트 [[(x1, y1), (x2, y2), ...], ...]
            image_size: 출력 이미지 크기 (None이면 기본값 사용)
            line_width: 선 두께 (None이면 기본값 사용)
            padding: 여백 (None이면 기본값 사용)
            background: 배경색 (None이면 기본값 사용)
            stroke_color: 획 색상 (None이면 기본값 사용)
        
        Returns:
            PIL.Image 객체 또는 None (실패 시)
        """
        if not strokes or not any(strokes):
            return None
        
        # 기본값 적용
        image_size = image_size or self.default_image_size
        line_width = line_width or self.default_line_width
        padding = padding or self.default_padding
        background = background or self.background_color
        stroke_color = stroke_color or self.stroke_color
        
        # 모든 좌표의 범위 계산
        all_points = [point for stroke in strokes for point in stroke if stroke]
        if not all_points:
            return None
        
        xs = [p[0] for p in all_points]
        ys = [p[1] for p in all_points]
        
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        
        # 원본 크기
        orig_width = max_x - min_x
        orig_height = max_y - min_y
        
        # 종횡비를 유지하면서 스케일 계산
        scale_x = (image_size[0] - 2 * padding) / orig_width if orig_width > 0 else 1
        scale_y = (image_size[1] - 2 * padding) / orig_height if orig_height > 0 else 1
        scale = min(scale_x, scale_y)
        
        # 이미지 생성
        img = Image.new('RGB', image_size, background)
        draw = ImageDraw.Draw(img)
        
        # 각 획 그리기
        for stroke in strokes:
            if not stroke:
                continue
            
            # 좌표 정규화 및 스케일링
            normalized_stroke = []
            for x, y in stroke:
                nx = (x - min_x) * scale + padding
                ny = (y - min_y) * scale + padding
                normalized_stroke.append((nx, ny))
            
            # 선 그리기
            if len(normalized_stroke) > 1:
                draw.line(normalized_stroke, fill=stroke_color, width=line_width)
            elif len(normalized_stroke) == 1:
                # 점 하나만 있는 경우
                x, y = normalized_stroke[0]
                r = line_width // 2
                draw.ellipse([x-r, y-r, x+r, y+r], fill=stroke_color)
        
        return img
    
    def render_to_array(self,
                       strokes: List[List[Tuple[float, float]]],
                       image_size: Optional[Tuple[int, int]] = None,
                       line_width: Optional[int] = None,
                       padding: Optional[int] = None,
                       normalize: bool = True) -> Optional[np.ndarray]:
        """
        획 데이터를 numpy array로 렌더링합니다 (딥러닝 모델 입력용).
        
        Args:
            strokes: 획 리스트
            image_size: 출력 이미지 크기
            line_width: 선 두께
            padding: 여백
            normalize: 0-1 범위로 정규화할지 여부
        
        Returns:
            numpy array (H, W) - grayscale 이미지
        """
        img = self.render_to_pil(
            strokes, 
            image_size=image_size,
            line_width=line_width,
            padding=padding,
            background='white',
            stroke_color='black'
        )
        
        if img is None:
            return None
        
        # Grayscale로 변환
        img_array = np.array(img.convert('L'))
        
        if normalize:
            # 0-1 범위로 정규화 (선은 0, 배경은 1)
            img_array = img_array.astype(np.float32) / 255.0
        
        return img_array
    
    def save_image(self,
                   strokes: List[List[Tuple[float, float]]],
                   output_path: str,
                   image_size: Optional[Tuple[int, int]] = None,
                   line_width: Optional[int] = None,
                   padding: Optional[int] = None) -> bool:
        """
        획 데이터를 이미지 파일로 저장합니다.
        
        Args:
            strokes: 획 리스트
            output_path: 저장할 파일 경로
            image_size: 출력 이미지 크기
            line_width: 선 두께
            padding: 여백
        
        Returns:
            성공 여부
        """
        img = self.render_to_pil(
            strokes,
            image_size=image_size,
            line_width=line_width,
            padding=padding
        )
        
        if img is None:
            return False
        
        try:
            img.save(output_path)
            return True
        except Exception as e:
            print(f"이미지 저장 실패: {e}")
            return False
    
    def get_bounding_box(self, strokes: List[List[Tuple[float, float]]]) -> Optional[Tuple[float, float, float, float]]:
        """
        획들의 바운딩 박스를 계산합니다.
        
        Args:
            strokes: 획 리스트
        
        Returns:
            (min_x, min_y, max_x, max_y) 또는 None
        """
        if not strokes or not any(strokes):
            return None
        
        all_points = [point for stroke in strokes for point in stroke if stroke]
        if not all_points:
            return None
        
        xs = [p[0] for p in all_points]
        ys = [p[1] for p in all_points]
        
        return (min(xs), min(ys), max(xs), max(ys))
    
    def calculate_image_size(self,
                            strokes: List[List[Tuple[float, float]]],
                            max_width: int = 800,
                            max_height: int = 600,
                            padding: Optional[int] = None) -> Tuple[int, int]:
        """
        획의 종횡비를 유지하면서 적절한 이미지 크기를 계산합니다.
        
        Args:
            strokes: 획 리스트
            max_width: 최대 너비
            max_height: 최대 높이
            padding: 여백
        
        Returns:
            (width, height)
        """
        padding = padding or self.default_padding
        bbox = self.get_bounding_box(strokes)
        
        if bbox is None:
            return (max_width, max_height)
        
        min_x, min_y, max_x, max_y = bbox
        orig_width = max_x - min_x
        orig_height = max_y - min_y
        
        if orig_width <= 0 or orig_height <= 0:
            return (max_width, max_height)
        
        # 종횡비 계산
        aspect_ratio = orig_width / orig_height
        
        # 패딩을 고려한 최대 크기
        avail_width = max_width - 2 * padding
        avail_height = max_height - 2 * padding
        
        # 종횡비를 유지하면서 크기 결정
        if avail_width / avail_height > aspect_ratio:
            # 높이 기준
            height = avail_height
            width = int(height * aspect_ratio)
        else:
            # 너비 기준
            width = avail_width
            height = int(width / aspect_ratio)
        
        return (width + 2 * padding, height + 2 * padding)
