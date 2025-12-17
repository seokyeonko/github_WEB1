# -*- coding: utf-8 -*-
"""
value 텍스트 분량 계산기
- 동작: 폴더 선택 → 재귀적으로 모든 .json 탐색 → 'value' 키의 문자열만 집계
- 산출: 글자수(공백 제외), 글자수(공백 포함), 어절수(띄어쓰기 기준)
- 출력: 파일 경로별 합계 및 전체 합계 프린트
"""
import os
import json
import re
import sys

# ---------- 유틸: 폴더 선택(가능하면 GUI, 실패 시 콘솔 입력) ----------
def choose_folder():
    try:
        # Tk가 없는 환경도 있으므로 동적 import & 예외 처리
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        path = filedialog.askdirectory(title="분석할 폴더를 선택하세요")
        if path:
            return path
    except Exception:
        pass

    # 콘솔 입력 fallback
    while True:
        path = input("분석할 폴더 경로를 입력하세요: ").strip('"').strip()
        if os.path.isdir(path):
            return path
        print("유효한 폴더 경로가 아닙니다. 다시 입력해 주세요.\n")

# ---------- 토큰/글자수 계산 ----------
_WS = re.compile(r"\s+", flags=re.UNICODE)

def char_count_including_spaces(text: str) -> int:
    return len(text)

def char_count_excluding_spaces(text: str) -> int:
    # 모든 공백(스페이스/탭/개행 등) 제거
    return len(_WS.sub("", text))

def eojeol_count(text: str) -> int:
    text = text.strip()
    if not text:
        return 0
    return len([tok for tok in _WS.split(text) if tok])

# ---------- JSON 탐색: 'value' 키의 문자열만 추출 ----------
def iter_values(obj):
    """
    임의의 JSON 구조(dict/list/… )에서
    키 이름이 정확히 'value'이고 값이 문자열인 항목만 yield
    """
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == "value" and isinstance(v, str):
                yield v
            # 하위 구조 재귀 탐색
            yield from iter_values(v)
    elif isinstance(obj, list):
        for it in obj:
            yield from iter_values(it)
    # 숫자/불리언/None/문자열 등은 상위에서 처리하므로 여기선 패스

# ---------- 파일 단위 처리 ----------
def process_json_file(path):
    """
    파일 하나를 읽어 value 문자열을 모두 합산하여
    (chars_no_space, chars_with_space, eojeols) 합계를 반환
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"[WARN] JSON 파싱 실패: {path} ({e})")
        return (0, 0, 0, 0)  # (개수, 공백제외, 공백포함, 어절)

    values = list(iter_values(data))
    total_no_space = 0
    total_with_space = 0
    total_eojeol = 0

    for s in values:
        total_no_space += char_count_excluding_spaces(s)
        total_with_space += char_count_including_spaces(s)
        total_eojeol += eojeol_count(s)

    return (len(values), total_no_space, total_with_space, total_eojeol)

# ---------- 메인 ----------
def main():
    root_dir = choose_folder()
    print("\n[폴더] ", root_dir)
    print("모든 하위 폴더까지 재귀적으로 .json 파일을 검색합니다.\n")

    # 표 헤더
    header = f"{'파일 경로':<60} | {'value개수':>8} | {'글자수-공백제외':>14} | {'글자수-공백포함':>14} | {'어절수':>8}"
    sep = "-" * len(header)
    print(header)
    print(sep)

    grand_values = 0
    grand_chars_no_space = 0
    grand_chars_with_space = 0
    grand_eojeols = 0
    file_count = 0

    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            if not fname.lower().endswith(".json"):
                continue
            file_count += 1
            fpath = os.path.join(dirpath, fname)
            n_values, c_no, c_with, e_cnt = process_json_file(fpath)

            grand_values += n_values
            grand_chars_no_space += c_no
            grand_chars_with_space += c_with
            grand_eojeols += e_cnt

            # 파일별 라인 출력
            print(f"{fpath:<60} | {n_values:>8} | {c_no:>14} | {c_with:>14} | {e_cnt:>8}")

    print(sep)
    print(
        f"{'합계(TOTAL)':<60} | {grand_values:>8} | {grand_chars_no_space:>14} | {grand_chars_with_space:>14} | {grand_eojeols:>8}"
    )
    if file_count == 0:
        print("\n[정보] 처리할 .json 파일을 찾지 못했습니다.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n중단되었습니다.", file=sys.stderr)
