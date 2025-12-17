# -*- coding: utf-8 -*-
# 특정 폴더를 지정하면 json내에 value 값(텍스트)을 카운트하는 코드(문자수, 어절 수)
"""
value 텍스트 분량 계산기 (CSV 자동저장 + 평균 포함)
- 폴더 선택 → 재귀적으로 모든 .json 탐색
- 'value' 키의 문자열만 대상으로 글자수/어절수 계산
- 결과를 콘솔 표 + CSV 파일로 저장 (바탕화면)
"""
import os
import json
import re
import sys
import csv
from datetime import datetime

# ---------- 폴더 선택 ----------
def choose_folder():
    while True:
        path = input("분석할 폴더 경로를 입력하세요: ").strip('"').strip()
        if os.path.isdir(path):
            return path
        print("유효한 폴더 경로가 아닙니다. 다시 입력해 주세요.\n")

# ---------- 계산 함수 ----------
_WS = re.compile(r"\s+", flags=re.UNICODE)

def char_count_including_spaces(text: str) -> int:
    return len(text)

def char_count_excluding_spaces(text: str) -> int:
    return len(_WS.sub("", text))

def eojeol_count(text: str) -> int:
    text = text.strip()
    if not text:
        return 0
    return len([tok for tok in _WS.split(text) if tok])

# ---------- JSON에서 'value'만 추출 ----------
def iter_values(obj):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == "value" and isinstance(v, str):
                yield v
            yield from iter_values(v)
    elif isinstance(obj, list):
        for it in obj:
            yield from iter_values(it)

# ---------- 파일 단위 처리 ----------
def process_json_file(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"[WARN] JSON 파싱 실패: {path} ({e})")
        return (0, 0, 0, 0)

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
    print(f"\n[폴더 선택됨] {root_dir}")
    print("하위 폴더 포함 모든 .json 파일을 분석합니다...\n")

    results = []
    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            if not fname.lower().endswith(".json"):
                continue
            fpath = os.path.join(dirpath, fname)
            n_values, c_no, c_with, e_cnt = process_json_file(fpath)
            results.append({
                "파일경로": fpath,
                "value개수": n_values,
                "글자수(공백제외)": c_no,
                "글자수(공백포함)": c_with,
                "어절수": e_cnt
            })

    # 콘솔 출력 요약
    print(f"{'파일 경로':<60} | {'value개수':>8} | {'공백제외':>10} | {'공백포함':>10} | {'어절수':>8}")
    print("-" * 105)
    for r in results:
        print(f"{r['파일경로']:<60} | {r['value개수']:>8} | {r['글자수(공백제외)']:>10} | {r['글자수(공백포함)']:>10} | {r['어절수']:>8}")

    total_values = sum(r["value개수"] for r in results)
    total_no_space = sum(r["글자수(공백제외)"] for r in results)
    total_with_space = sum(r["글자수(공백포함)"] for r in results)
    total_eojeol = sum(r["어절수"] for r in results)
    file_count = len(results)

    print("-" * 105)
    print(f"{'총합계(TOTAL)':<60} | {total_values:>8} | {total_no_space:>10} | {total_with_space:>10} | {total_eojeol:>8}")

    # ---------- CSV 저장 ----------
    desktop = os.path.join(os.path.expanduser("~"), "Desktop")
    filename = f"value_count_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    save_path = os.path.join(desktop, filename)

    # 평균 계산
    avg_values = total_values / file_count if file_count else 0
    avg_no_space = total_no_space / file_count if file_count else 0
    avg_with_space = total_with_space / file_count if file_count else 0
    avg_eojeol = total_eojeol / file_count if file_count else 0

    with open(save_path, "w", newline="", encoding="utf-8-sig") as f:
        fieldnames = ["파일경로", "value개수", "글자수(공백제외)", "글자수(공백포함)", "어절수"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
        writer.writerow({})
        writer.writerow({
            "파일경로": "총합계(TOTAL)",
            "value개수": total_values,
            "글자수(공백제외)": total_no_space,
            "글자수(공백포함)": total_with_space,
            "어절수": total_eojeol
        })
        writer.writerow({
            "파일경로": "평균(Average)",
            "value개수": f"{avg_values:.2f}",
            "글자수(공백제외)": f"{avg_no_space:.2f}",
            "글자수(공백포함)": f"{avg_with_space:.2f}",
            "어절수": f"{avg_eojeol:.2f}"
        })

    print(f"\n[완료] 결과가 CSV 파일로 저장되었습니다:")
    print(f"📁 {save_path}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n사용자에 의해 중단됨.", file=sys.stderr)
