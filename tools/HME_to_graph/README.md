# HME_to_graph

CROHME 손글씨 수식 → 이미지 + 그래프 변환기

## 📁 구조

```
data/crohme/
├── annotation/           # 정답 파일
│   ├── crohme2019_train.txt
│   ├── crohme2019_test.txt
│   └── crohme2019_valid.txt
└── inkml/               # 손글씨 파일
    ├── train/*.inkml
    ├── test/*.inkml
    └── valid/*.inkml

output/                  # 변환 결과
├── train/
│   ├── images/*.png
│   └── train_graphs.json
├── valid/
│   ├── images/*.png
│   └── valid_graphs.json
└── test/
    ├── images/*.png
    └── test_graphs.json
```

## 🚀 사용법

```bash
python main.py
```

## 📊 출력

### 이미지
- 경로: `output/{split}/images/*.png`
- 크기: 800x600

### 그래프 JSON
```json
{
  "train": {
    "file_001": {
      "symbols": ["\\alpha", "+", "\\beta"],
      "relations": [[0, 1, 1]],
      "filename": "file_001.inkml"
    }
  }
}
```
