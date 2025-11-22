"""
JSON 압축 유틸리티
불필요한 줄바꿈 제거하고 공백 최소화
"""

import json
from pathlib import Path
from typing import Optional, Union


def compact_json_file(input_file: str, 
                      output_file: Optional[str] = None,
                      indent: Optional[int] = None,
                      ensure_ascii: bool = False,
                      overwrite: bool = False) -> dict:
    """
    JSON 파일을 압축 형식으로 변환
    
    Args:
        input_file: 입력 JSON 파일 경로
        output_file: 출력 파일 (None이면 '_compact' 접미사 추가)
        indent: 들여쓰기 (None=한줄, 0=줄바꿈만, 2=2칸 등)
        ensure_ascii: True=ASCII만, False=유니코드 허용
        overwrite: True면 원본 파일 덮어쓰기 (output_file 무시)
    
    Returns:
        로드된 JSON 데이터
    
    예시:
        # 완전 압축 (한 줄)
        compact_json_file('train_graphs.json')
        
        # 줄바꿈은 유지, 공백만 최소화
        compact_json_file('train_graphs.json', indent=0)
        
        # 적당한 가독성 (2칸 들여쓰기)
        compact_json_file('train_graphs.json', indent=2)
        
        # 원본 파일 덮어쓰기
        compact_json_file('train_graphs.json', overwrite=True)
    """
    input_path = Path(input_file)
    
    # 출력 파일명 결정
    if overwrite:
        output_path = input_path
    elif output_file is None:
        output_path = input_path.parent / f"{input_path.stem}_compact{input_path.suffix}"
    else:
        output_path = Path(output_file)
    
    # JSON 로드
    print(f"읽는 중: {input_file}")
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    original_size = input_path.stat().st_size
    
    # 압축 저장
    print(f"저장 중: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        if indent is None:
            # 완전 압축 (한 줄, 공백 없음)
            json.dump(data, f, separators=(',', ':'), ensure_ascii=ensure_ascii)
        else:
            # 들여쓰기 있지만 공백은 최소화
            json.dump(data, f, indent=indent, separators=(',', ':'), ensure_ascii=ensure_ascii)
    
    compressed_size = output_path.stat().st_size
    reduction = (1 - compressed_size / original_size) * 100
    
    print(f"✓ 완료!")
    print(f"  원본 크기: {original_size:,} bytes")
    print(f"  압축 크기: {compressed_size:,} bytes")
    print(f"  감소율: {reduction:.1f}%")
    
    return data

