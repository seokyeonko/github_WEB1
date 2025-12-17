# Document Text Extractor

다양한 문서 파일 형식에서 텍스트를 자동으로 추출하는 Python 스크립트입니다.

## 📋 프로그램 개요

이 스크립트는 지정된 폴더(및 하위 폴더) 내의 다양한 문서 파일에서 텍스트를 추출하여 개별 텍스트 파일(.txt)로 저장하는 도구입니다.

## ✨ 주요 기능

### 지원 파일 형식
- **한글 (.hwp)** : hwp5txt.exe를 자동으로 탐지하여 텍스트 추출
- **엑셀 (.xlsx, .xls)** : 시트별 텍스트 추출
- **파워포인트 (.pptx)** : 슬라이드별 텍스트 추출
- **PDF (.pdf)** : pdfplumber 및 PyPDF2를 병행하여 텍스트 추출
- **워드 (.docx)** : 문단 및 표 데이터 추출

### 주요 특징
- 폴더 경로를 입력받아 재귀적으로 파일 탐색
- tqdm 라이브러리를 이용한 진행률(Progress Bar) 표시
- 추출된 텍스트는 원본 파일 위치의 'txt output' 폴더에 자동 저장
- 인코딩 자동 감지 및 에러 처리 포함

## 🔧 설치 및 실행 방법

### 1. Python 버전 요구사항
- **Python 3.8 이상** 권장

### 2. 가상환경 생성

#### Windows
```bash
# 가상환경 생성
python -m venv venv

# 가상환경 활성화
venv\Scripts\activate
```

#### macOS / Linux
```bash
# 가상환경 생성
python3 -m venv venv

# 가상환경 활성화
source venv/bin/activate
```

### 3. 필수 모듈 설치

가상환경이 활성화된 상태에서 다음 명령어를 실행하세요:

```bash
pip install -r requirements.txt
```

또는 개별 설치:

```bash
pip install pandas python-pptx PyPDF2 python-docx pdfplumber chardet tqdm openpyxl
```

### 4. HWP 파일 지원 (선택사항)

HWP 파일 추출을 위해서는 `hwp5txt.exe`가 필요합니다:

```bash
pip install hwp5
```

설치 후 `hwp5txt.exe`는 자동으로 탐지됩니다.

### 5. 프로그램 실행

```bash
python text3(final).py
```

## 📖 사용 방법

1. 프로그램을 실행합니다
2. 콘솔에 텍스트를 추출할 대상 폴더 경로를 입력합니다
3. 진행 상황을 확인합니다 (진행률 바 표시)
4. 각 파일의 원본 위치에 생성된 `txt output` 폴더에서 결과를 확인합니다

### 실행 예시

```
텍스트를 추출할 '폴더' 경로를 입력하세요: C:\Documents\MyFiles

총 15개 파일에서 텍스트 추출을 시작합니다.

파일 처리 중: 100%|████████████████████| 15/15 [00:23<00:00,  1.56s/it]

모든 처리가 완료되었습니다.
```

## 📁 출력 구조

```
원본폴더/
├── document1.pdf
├── document2.hwp
└── txt output/          # 자동 생성됨
    ├── document1_추출결과.txt
    └── document2_추출결과.txt
```

## 🔍 주요 함수 설명

### `detect_hwp5txt()`
- hwp5txt.exe 실행 파일을 자동으로 탐지
- venv/Scripts 폴더 또는 시스템 PATH에서 검색

### `extract_text_from_*()` 시리즈
- 각 파일 형식별 텍스트 추출 함수
- 에러 처리 및 인코딩 자동 감지 포함

### `find_files(folder, exts)`
- 지정된 폴더에서 지원 파일 형식을 재귀적으로 탐색

### `save_text_to_output_folder(orig_file_path, text)`
- 추출된 텍스트를 원본 파일 위치의 'txt output' 폴더에 저장

## ⚠️ 주의사항

- HWP 파일 처리를 위해서는 `hwp5` 패키지가 설치되어 있어야 합니다
- 대용량 파일 처리 시 시간이 소요될 수 있습니다
- PDF 파일의 경우 이미지 기반 PDF는 텍스트 추출이 제한적일 수 있습니다

## 🐛 문제 해결

### HWP 파일이 처리되지 않는 경우
```bash
pip install hwp5
```

### 인코딩 오류가 발생하는 경우
- `chardet` 라이브러리가 자동으로 인코딩을 감지하지만, 일부 파일은 수동 확인이 필요할 수 있습니다

### 모듈 import 오류
- 가상환경이 활성화되어 있는지 확인하세요
- `pip install -r requirements.txt`를 다시 실행하세요

## 📝 라이선스

이 프로젝트는 자유롭게 사용 가능합니다.

## 👤 기여

버그 리포트 및 기능 제안은 Issues를 통해 제출해 주세요.
