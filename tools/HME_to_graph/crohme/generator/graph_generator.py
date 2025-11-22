import json
from collections import defaultdict
from typing import List, Dict, Tuple, Optional
from pathlib import Path


class GraphGenerator:
    """어노테이션을 그래프 형태로 변환하는 클래스 (latex_class.json 사용)"""
    
    def __init__(self, latex_data):
        """
        Args:
            latex_data: 수식 관계 데이터 (latex_class.json을 json.load 한 dict)
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
        토큰 시퀀스를 (심볼 ID 리스트, 관계 리스트)로 변환.
        
        - 심볼 ID 리스트: latex_class.json의 symbol2id에 따라 매핑된 정수 ID
        - 관계 리스트: [from_node_idx, to_node_idx, rel_id] 형태
        
        node index는 "토큰 시퀀스에서 관계 토큰이 아닌 심볼의 등장 순서"를 그대로 사용.
        예를 들어 토큰이 [4, Right, n, Right, -] 라면
        심볼 시퀀스는 [4, n, -] → 노드 인덱스 [0, 1, 2]
        """
        symbol_tokens: List[str] = []
        token_pos_to_node_idx: Dict[int, int] = {}
        relations: List[List[int]] = []
        unknown_symbols = set()
        
        # 1-pass: 심볼 토큰만 모으고, 토큰 위치→노드 인덱스 매핑
        node_idx = 0
        for i, tok in enumerate(tokens):
            if tok not in self.rel_to_id:  # 관계 키워드가 아니면 심볼
                token_pos_to_node_idx[i] = node_idx
                symbol_tokens.append(tok)
                node_idx += 1
        
        # 2-pass: 관계 토큰을 보면서 앞/뒤 심볼을 엣지로 연결
        for i, tok in enumerate(tokens):
            if tok in self.rel_to_id:
                rel_name = tok
                if rel_name == "NoRel":
                    continue  # NoRel은 엣지 만들지 않음
                
                # 관계 토큰 양 옆에 심볼이 있는 경우만 사용
                if (i - 1) in token_pos_to_node_idx and (i + 1) in token_pos_to_node_idx:
                    src_idx = token_pos_to_node_idx[i - 1]
                    dst_idx = token_pos_to_node_idx[i + 1]
                    rel_id = self.rel_to_id[rel_name]
                    relations.append([src_idx, dst_idx, rel_id])
        
        # 심볼 토큰을 ID로 변환 (labels 역할)
        symbol_ids: List[int] = []
        for tok in symbol_tokens:
            if tok in self.symbol_to_id:
                symbol_ids.append(self.symbol_to_id[tok])
            else:
                symbol_ids.append(-1)
                unknown_symbols.add(tok)
        
        if unknown_symbols:
            print(f"Warning: Unknown symbols found: {unknown_symbols}")
        
        return symbol_ids, relations
    
    def create_graph_from_file(self, 
                               annotation_file: str,
                               file_id: Optional[str] = None) -> Optional[Dict]:
        """
        단일 어노테이션 파일을 읽어 그래프로 변환합니다.
        
        Returns:
            {
                'labels': List[int],      # 노드별 심볼 클래스 ID
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
        
        symbol_ids, relations = self.tokens_to_graph(tokens)
        
        if file_id is None:
            file_id = Path(filename).stem
        
        return {
            'labels': symbol_ids,
            'relations': relations,
            'filename': filename,
            'file_id': file_id
        }
    
    def create_graph_batch(self, 
                           annotation_file: str,
                           output_format: str = 'detailed') -> Dict:
        """
        어노테이션 파일의 모든 항목을 그래프로 변환합니다.
        
        output_format:
            'simple'  → file_id: [[src, dst, rel_id], ...]
            'detailed'→ file_id: {
                             'labels': [...],
                             'relations': [[src, dst, rel_id], ...],
                             'filename': str
                          }
        """
        result: Dict[str, Dict] = {}
        
        with open(annotation_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                filename, tokens = self.parse_annotation_line(line)
                if filename is None or tokens is None:
                    continue
                
                file_id = Path(filename).stem
                symbol_ids, relations = self.tokens_to_graph(tokens)
                
                if output_format == 'simple':
                    result[file_id] = relations
                else:
                    result[file_id] = {
                        'labels': symbol_ids,
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
        
        result (detailed 기준):
        {
          "train": {
            "file_id": {
              "labels": [...],
              "relations": [[src, dst, rel_id], ...],
              "filename": "crohme2019/train/XXX.inkml"
            },
            ...
          },
          "rel_categories": { "Right": 1, ... },
          "num_classes": 320,
          "symbol_to_id": {...}
        }
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
