# 🔄 KoBackTrans — NER-Aware Korean Back-Translation

> **한국어** | [**English**](#english)

---

## 한국어

### 개요

한국어 데이터 증강을 위한 **NER 기반 역번역(Back-Translation) 파이프라인**입니다.

구글 번역을 이용한 역번역(`한국어 → 일본어 → 한국어`)은 문장 표현을 다양하게 만들어 NLP 모델 학습 데이터를 늘리는 데 효과적입니다. 하지만 고유명사(인명, 기관명 등)가 의도치 않게 번역되는 문제가 있습니다. 이를 해결하기 위해 번역 전 **NER로 개체명을 마스킹**하고, 번역 후 원래 표현으로 **복원**하는 방식을 사용합니다.

또한 한국어 조사 처리 로직이 포함되어 있어, 개체명에 조사(`은/는/이/가/을/를` 등)가 붙어 있어도 올바르게 마스킹·복원됩니다.

### 주요 기능

- **NER 기반 개체명 보호** — 번역 중 고유명사 변형 방지
- **한국어 조사 처리** — 어절 단위 분리 및 조사 보존
- **중간 언어 역번역** — 한국어 → 일본어 → 한국어
- **GPU 자동 감지** — CUDA 사용 가능 시 자동 적용
- **실행 시간 로깅** — 시작/종료/소요 시간 출력

### 설치

```bash
pip install transformers googletrans==4.0.0rc1 torch
```

### 사용법

`back-trans.py` 하단의 `__main__` 블록에서 경로와 옵션을 설정합니다.

```python
input_file_path  = 'input.txt'   # 입력 파일 경로
output_file_path = 'output.txt'  # 출력 파일 경로
model_name       = "monologg/koelectra-base-v3-naver-ner"  # NER 모델

# 마스킹할 개체명 태그 (사용 모델의 태그셋에 맞게 조정)
entities_to_mask = ['ORG-B', 'PER-B', 'CVL-B']
```

설정 후 실행:

```bash
python back-trans.py
```

입력 파일은 **한 줄에 한 문장** 형식의 UTF-8 텍스트 파일입니다.

### 지원 NER 모델

| 모델 | 설명 |
|------|------|
| `monologg/koelectra-base-v3-naver-ner` | 기본값, 네이버 NER 데이터 기반 |
| `Leo97/KoELECTRA-small-v3-modu-ner` | 경량 모델, 모두의 말뭉치 기반 |

모델에 따라 `entities_to_mask`의 태그명이 달라질 수 있습니다.

---

<a name="english"></a>

## English

### Overview

A **NER-aware back-translation pipeline** for Korean text data augmentation.

Back-translation via Google Translate (`Korean → Japanese → Korean`) is a powerful technique to diversify sentence expressions for NLP training data. However, a known issue is that proper nouns (names, organizations, etc.) tend to get incorrectly translated. This pipeline solves that by **masking named entities before translation** and **restoring them afterward** using a Korean NER model.

Korean postposition handling is also built in, so entity tokens with attached particles (e.g., `은/는/이/가`) are correctly masked and restored at the eojeol (word unit) level.

### Features

- **NER-based entity protection** — prevents proper noun corruption during translation
- **Korean postposition handling** — eojeol-level masking with particle preservation
- **Intermediate language back-translation** — Korean → Japanese → Korean
- **Automatic GPU detection** — uses CUDA when available
- **Execution time logging** — prints start, end, and elapsed time

### Installation

```bash
pip install transformers googletrans==4.0.0rc1 torch
```

### Usage

Configure the paths and options in the `__main__` block at the bottom of `back-trans.py`:

```python
input_file_path  = 'input.txt'   # path to input file
output_file_path = 'output.txt'  # path to output file
model_name       = "monologg/koelectra-base-v3-naver-ner"  # NER model

# entity tags to mask (adjust to match your model's tagset)
entities_to_mask = ['ORG-B', 'PER-B', 'CVL-B']
```

Then run:

```bash
python back-trans.py
```

The input file should be a UTF-8 plain text file with **one sentence per line**.

### Supported NER Models

| Model | Description |
|-------|-------------|
| `monologg/koelectra-base-v3-naver-ner` | Default; trained on Naver NER dataset |
| `Leo97/KoELECTRA-small-v3-modu-ner` | Lightweight; trained on Modu corpus |

Note: `entities_to_mask` tag names differ depending on the model's tagset.

### How It Works

```
Input sentence
     │
     ▼
[NER] Detect & mask named entities (e.g., 이순신 → <@_0>이)
     │
     ▼
[Google Translate] Korean → Japanese
     │
     ▼
[Google Translate] Japanese → Korean
     │
     ▼
[Restore] Replace masks with original entities
     │
     ▼
Output sentence (paraphrased, proper nouns preserved)
```

---

## License

MIT
