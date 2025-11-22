import json
from collections import OrderedDict


class LaTeXSymbolMapper:
    """LaTeX 기호와 클래스 ID를 매핑하는 클래스"""
    
    def __init__(self):
        self.symbol2id = OrderedDict()
        self.id2symbol = {}
        self._build_mapping()
    
    def _build_mapping(self):
        """모든 LaTeX 기호 매핑 생성"""
        
        # 숫자 (0-9)
        digits = [str(i) for i in range(10)]
        
        # 영문 소문자 (a-z)
        lowercase = [chr(i) for i in range(ord('a'), ord('z') + 1)]
        
        # 영문 대문자 (A-Z)
        uppercase = [chr(i) for i in range(ord('A'), ord('Z') + 1)]
        
        # 그리스 문자 (소문자)
        greek_lowercase = [
            '\\alpha', '\\beta', '\\gamma', '\\delta', '\\epsilon', '\\varepsilon',
            '\\zeta', '\\eta', '\\theta', '\\vartheta', '\\iota', '\\kappa',
            '\\lambda', '\\mu', '\\nu', '\\xi', '\\pi', '\\varpi',
            '\\rho', '\\varrho', '\\sigma', '\\varsigma', '\\tau', '\\upsilon',
            '\\phi', '\\varphi', '\\chi', '\\psi', '\\omega'
        ]
        
        # 그리스 문자 (대문자)
        greek_uppercase = [
            '\\Gamma', '\\Delta', '\\Theta', '\\Lambda', '\\Xi', '\\Pi',
            '\\Sigma', '\\Upsilon', '\\Phi', '\\Psi', '\\Omega'
        ]
        
        # 기본 연산자
        basic_operators = [
            '+', '-', '\\times', '/', '\\div', '\\pm', '\\mp',
            '=', '\\neq', '\\equiv', '\\approx', '\\cong',
            '<', '>', '\\leq', '\\geq', '\\ll', '\\gg',
            '\\cdot', '\\ast', '\\star', '\\circ', '\\bullet',
            '\\oplus', '\\ominus', '\\otimes', '\\oslash', '\\lt', '\\gt', '\\%'
        ]
        
        # 집합 연산자
        set_operators = [
            '\\in', '\\notin', '\\subset', '\\subseteq', '\\supset', '\\supseteq',
            '\\cup', '\\cap', '\\setminus', '\\emptyset', '\\varnothing'
        ]
        
        # 논리 연산자
        logic_operators = [
            '\\land', '\\lor', '\\lnot', '\\neg', '\\implies', '\\iff',
            '\\forall', '\\exists', '\\nexists'
        ]
        
        # 화살표
        arrows = [
            '\\rightarrow', '\\leftarrow', '\\leftrightarrow',
            '\\Rightarrow', '\\Leftarrow', '\\Leftrightarrow',
            '\\mapsto', '\\to', '\\longrightarrow', '\\longleftarrow',
            '\\uparrow', '\\downarrow', '\\updownarrow',
            '\\xrightarrow', '\\xleftarrow'
        ]
        
        # 괄호류
        brackets = [
            '(', ')', '[', ']', '\\{', '\\}',
            '\\langle', '\\rangle', '\\lfloor', '\\rfloor',
            '\\lceil', '\\rceil', '|', '\\|', '\\mid',
            '\\lvert', '\\rvert', '\\lVert', '\\rVert'
        ]
        
        # 미적분 및 해석
        calculus = [
            '\\int', '\\iint', '\\iiint', '\\oint',
            '\\partial', '\\nabla', '\\infty',
            '\\lim', '\\limsup', '\\liminf',
            '\\sum', '\\prod', '\\coprod',
            '\\frac', '\\sqrt', '\\int', 'd', '\\limits'  # 'd' for differential
        ]
        
        # 함수
        functions = [
            '\\sin', '\\cos', '\\tan', '\\cot', '\\sec', '\\csc',
            '\\arcsin', '\\arccos', '\\arctan',
            '\\sinh', '\\cosh', '\\tanh', '\\coth',
            '\\exp', '\\log', '\\ln', '\\lg',
            '\\max', '\\min', '\\sup', '\\inf',
            '\\det', '\\dim', '\\ker', '\\deg',
            '\\arg', '\\gcd', '\\lcm'
        ]
        
        # 특수 문자
        special = [
            '\_', '\-',
            '\\prime', '\\dagger', '\\ddagger',
            '\\Re', '\\Im', '\\wp', '\\aleph',
            '\\hbar', '\\ell', '\\imath', '\\jmath',
            '\\angle', '\\triangle', '\\square', '\\Box',
            '\\Diamond', '\\bot', '\\top', '\\perp', '\\parallel', '\\vert', '\\bigcirc'
        ]
        
        # 점 및 강조
        accents = [
            '\\dot', '\\ddot', '\\hat', '\\tilde', '\\bar',
            '\\vec', '\\overline', '\\underline',
            '\\widehat', '\\widetilde', '\\overrightarrow'
        ]
        
        # 공백 및 구조
        structure = [
            ',', '.', ':', ';', '!', '?',
            '\\ldots', '\\cdots', '\\vdots', '\\ddots',
            '\\quad', '\\qquad', '\\,', '\\;', '\\:',
        ]
        
        # 상수
        constants = [
            'e', 'i', '\\pi', '\\hbar', '\\infty'
        ]
        
        # 행렬 관련
        matrix_symbols = [
            '\\matrix', '\\pmatrix', '\\bmatrix', '\\vmatrix',
            '\\begin', '\\end', '&', '\\\\'
        ]
        
        # 추가 수학 기호
        additional_math = [
            '\\mathbb', '\\mathcal', '\\mathfrak', '\\mathbf',
            '\\text', '\\mathrm',
            '\\left', '\\right', '\\big', '\\Big', '\\bigg', '\\Bigg',
            '\\propto', '\\sim', '\\simeq', '\\asymp',
            '\\prec', '\\succ', '\\preceq', '\\succeq',
            '\\vee', '\\wedge', '\\bigvee', '\\bigwedge',
            '\\binom', '\\choose', '\\because', '\\therefore', '\\boxed', '\\textcircled', '\\xlongequal', '\\rightleftharpoons',
        ]
        
        # 모든 기호 통합
        all_symbols = (
            digits +
            lowercase +
            uppercase +
            greek_lowercase +
            greek_uppercase +
            basic_operators +
            set_operators +
            logic_operators +
            arrows +
            brackets +
            calculus +
            functions +
            special +
            accents +
            structure +
            constants +
            matrix_symbols +
            additional_math
        )
        
        # 중복 제거 및 ID 할당
        unique_symbols = []
        seen = set()
        for symbol in all_symbols:
            if symbol not in seen:
                unique_symbols.append(symbol)
                seen.add(symbol)
        
        # 매핑 생성
        for idx, symbol in enumerate(unique_symbols):
            self.symbol2id[symbol] = idx
            self.id2symbol[idx] = symbol
        
        print(f"총 {len(self.symbol2id)}개의 고유 LaTeX 기호가 매핑되었습니다.")
    
    def get_id(self, symbol):
        """기호 → 클래스 ID"""
        return self.symbol2id.get(symbol, -1)  # -1: unknown
    
    def get_symbol(self, class_id):
        """클래스 ID → 기호"""
        return self.id2symbol.get(class_id, '<UNK>')
    
    def add_symbol(self, symbol):
        """새로운 기호 추가"""
        if symbol not in self.symbol2id:
            new_id = len(self.symbol2id)
            self.symbol2id[symbol] = new_id
            self.id2symbol[new_id] = symbol
            return new_id
        return self.symbol2id[symbol]
    
    def save_mapping(self, filepath):
        """매핑을 JSON 파일로 저장"""
        mapping_data = {
            'symbol2id': dict(self.symbol2id),
            'id2symbol': {str(k): v for k, v in self.id2symbol.items()},
            'num_classes': len(self.symbol2id)
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(mapping_data, f, ensure_ascii=False, indent=2)
        print(f"매핑이 {filepath}에 저장되었습니다.")
    
    def load_mapping(self, filepath):
        """JSON 파일에서 매핑 로드"""
        with open(filepath, 'r', encoding='utf-8') as f:
            mapping_data = json.load(f)
        
        self.symbol2id = OrderedDict(mapping_data['symbol2id'])
        self.id2symbol = {int(k): v for k, v in mapping_data['id2symbol'].items()}
        print(f"{filepath}에서 {len(self.symbol2id)}개 기호를 로드했습니다.")
    
    def get_num_classes(self):
        """총 클래스 수 반환"""
        return len(self.symbol2id)
    
    def print_mapping(self, max_display=50):
        """매핑 정보 출력"""
        print("\n" + "="*80)
        print("LaTeX 기호 → 클래스 ID 매핑")
        print("="*80)
        print(f"총 클래스 수: {self.get_num_classes()}")
        print()
        
        display_count = min(max_display, len(self.symbol2id))
        print(f"처음 {display_count}개 매핑:")
        for i, (symbol, idx) in enumerate(list(self.symbol2id.items())[:display_count]):
            print(f"  [{idx:3d}] {symbol}")
        
        if len(self.symbol2id) > max_display:
            print(f"  ... 외 {len(self.symbol2id) - max_display}개")
    
    def convert_symbols_to_ids(self, symbols):
        """기호 리스트를 ID 리스트로 변환"""
        return [self.get_id(symbol) for symbol in symbols]
    
    def convert_ids_to_symbols(self, ids):
        """ID 리스트를 기호 리스트로 변환"""
        return [self.get_symbol(idx) for idx in ids]


def create_relation_mapping():
    """수식 구조 관계 타입 매핑 생성"""
    relation_types = OrderedDict([
        (0, '__background__'),
        (1, 'Right'),
        (2, 'Below'),
        (3, 'Above'),
        (4, 'Inside'),
        (5, 'Sup'),
        (6, 'Sub'),
        (7, 'NoRel'),
    ])
    
    return relation_types

def save_all_mappings(output_file, mapper, relation_types):
    """LaTeX 기호 매핑 + 수식 구조 관계도를 하나의 JSON 파일로 저장"""

    output = {
        "symbols": {
            "symbol2id": dict(mapper.symbol2id),
            "id2symbol": {str(k): v for k, v in mapper.id2symbol.items()},
            "num_classes": len(mapper.symbol2id)
        },
        "relations": {str(k): v for k, v in relation_types.items()}
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"전체 매핑(기호 + 관계)이 {output_file}에 저장되었습니다.")


def map_latex_symbol(output_file: str):
    """메인 함수"""

    # LaTeX 기호 매퍼 생성
    mapper = LaTeXSymbolMapper()

    # 매핑 정보 출력
    mapper.print_mapping(max_display=100)

    # 관계 타입 출력
    print("\n" + "="*80)
    print("수식 구조 관계 타입")
    print("="*80)
    relation_types = create_relation_mapping()
    for rel_id, rel_name in relation_types.items():
        print(f"  [{rel_id:2d}] {rel_name}")

    # 하나의 JSON 파일로 저장
    save_all_mappings(output_file, mapper, relation_types)

