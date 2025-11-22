import json
from collections import defaultdict
from typing import List, Dict, Tuple, Optional
from pathlib import Path


class GraphGenerator:
    """어노테이션을 그래프 형태로 변환하는 클래스 (latex_class.json 사용)"""
    
    def __init__(self, latex_data):
        """
        Args:
            latex_class_file: latex_class.json 파일 경로
            latex_data: 수식 관계 데이터
        """
        
        # symbol2id 매핑 로드
        symbols = latex_data["symbols"]
        self.symbol_to_id = symbols.get('symbol2id', {})
        self.id_to_symbol = symbols.get('id2symbol', {})
        
        # 관계 카테고리 로드
        if "relations" not in latex_data:
            raise ValueError("JSON 파일에 'relations' 항목이 없습니다.")

        relations = latex_data['relations']
        
        # key → int, value → str
        self.rel_id_to_name = {int(k): v for k, v in relations.items()}

        # 역매핑: 관계명 → ID
        self.rel_to_id = {v: k for k, v in self.rel_id_to_name.items()}
    
    def parse_annotation_line(self, line: str) -> Tuple[Optional[str], Optional[List[str]]]:
        """
        어노테이션 라인을 파싱합니다.
        
        입력 예시:
        "crohme2019/test/UN19_1032.inkml\t4 Right n Right - Right 4"
        
        Returns:
            (파일명, 토큰 리스트) 또는 (None, None)
        """
        parts = line.strip().split('\t')
        if len(parts) != 2:
            return None, None
        
        filename = parts[0]
        tokens = parts[1].split()
        
        return filename, tokens
    
    def tokens_to_graph(self, tokens: List[str]) -> Tuple[List[int], List[List[int]]]:
        """
        토큰 시퀀스를 (기호 ID 리스트, 관계 리스트)로 변환
        relations에 symbol_ids 사용
        """
        symbol_ids = []
        relations = []
        unknown_symbols = set()
        
        i = 0
        while i < len(tokens):
            token = tokens[i]
            
            # 현재 토큰이 기호
            if token not in self.rel_to_id:
                if token in self.symbol_to_id:
                    symbol_id = self.symbol_to_id[token]
                else:
                    symbol_id = -1
                    unknown_symbols.add(token)
                
                symbol_ids.append(symbol_id)
                current_idx = len(symbol_ids) - 1
                
                # 다음이 관계 키워드인지 확인
                if i + 1 < len(tokens) and tokens[i + 1] in self.rel_to_id:
                    relation = tokens[i + 1]
                    
                    if i + 2 < len(tokens):
                        # 다음 기호가 올 것으로 예상
                        next_idx = len(symbol_ids)  # 아직 추가 전
                        next_token = tokens[i + 2]
                        
                        if next_token in self.symbol_to_id:
                            next_symbol_id = self.symbol_to_id[next_token]
                        else:
                            next_symbol_id = -1
                            unknown_symbols.add(next_token)
                        
                        # NoRel이 아닌 경우만 관계 추가
                        if relation != "NoRel":
                            relation_id = self.rel_to_id[relation]
                            relations.append([symbol_id, next_symbol_id, relation_id])
                        
                        i += 1  # relation 토큰 건너뛰기
                i += 1
            else:
                # 관계만 나온 경우
                i += 1
        
        if unknown_symbols:
            print(f"Warning: Unknown symbols found: {unknown_symbols}")
        
        return relations
    
    def create_graph_from_file(self, 
                               annotation_file: str,
                               file_id: Optional[str] = None) -> Dict:
        """
        단일 어노테이션 파일을 읽어 그래프로 변환합니다.
        
        Args:
            annotation_file: 어노테이션 파일 경로
            file_id: 파일 ID (None이면 파일명에서 추출)
        
        Returns:
            {
                'symbol_ids': List[int],  # latex_class.json의 ID
                'relations': List[List[int]],
                'filename': str,
                'file_id': str
            }
        """
        with open(annotation_file, 'r', encoding='utf-8') as f:
            line = f.readline()
        
        filename, tokens = self.parse_annotation_line(line)
        
        if filename is None or tokens is None:
            return None
        
        relations = self.tokens_to_graph(tokens)
        
        if file_id is None:
            file_id = Path(filename).stem
        
        return {
            'relations': relations,
            'filename': filename,
            'file_id': file_id
        }
    
    def create_graph_batch(self, 
                          annotation_file: str,
                          output_format: str = 'detailed') -> Dict:
        """
        어노테이션 파일의 모든 항목을 그래프로 변환합니다.
        
        Args:
            annotation_file: 어노테이션 파일 경로
            output_format: 'simple' 또는 'detailed'
        
        Returns:
            output_format이 'simple'인 경우:
            {
                file_id: [[s, o, r], ...],
                ...
            }
            
            output_format이 'detailed'인 경우:
            {
                file_id: {
                    'relations': [[s, o, r], ...],
                    'filename': str
                },
                ...
            }
        """
        result = {}
        
        with open(annotation_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                filename, tokens = self.parse_annotation_line(line)
                
                if filename is None or tokens is None:
                    continue
                
                file_id = Path(filename).stem
                relations = self.tokens_to_graph(tokens)
                
                if output_format == 'simple':
                    result[file_id] = relations
                else:  # detailed
                    result[file_id] = {
                        'relations': relations,
                        'filename': filename
                    }
        
        return result
    
    def save_graph_json(self,
                       annotation_file: str,
                       output_file: str,
                       split_name: str = 'train',
                       output_format: str = 'detailed'):
        """
        그래프를 JSON 파일로 저장합니다.
        
        Args:
            annotation_file: 입력 어노테이션 파일
            output_file: 출력 JSON 파일
            split_name: 데이터셋 스플릿 이름
            output_format: 'simple' 또는 'detailed'
        """
        graphs = self.create_graph_batch(annotation_file, output_format)
        
        result = {
            split_name: graphs,
            'rel_categories': self.rel_to_id,
            'num_classes': len(self.symbol_to_id),
            'symbol_to_id': self.symbol_to_id
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        print(f"그래프 저장 완료: {output_file}")
        print(f"  - 샘플 수: {len(graphs)}")
        print(f"  - 관계 카테고리: {len(self.rel_to_id)}개")
        print(f"  - 기호 클래스: {len(self.symbol_to_id)}개")
