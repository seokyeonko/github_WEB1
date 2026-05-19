#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
from typing import Any

OLD_NAME = "AI 파운데이션 모델(LLM/LAM) 사후학습용 데이터"
NEW_NAME = "AI 파운데이션 모델(LLM/LAM/LMM) 학습용 데이터"


def remove_llm_prefix_from_id(value: Any) -> Any:
    """
    문자열 값이 'llm-'로 시작하면 앞의 'llm-'만 제거한다.
    예: llm-reference-0000001 -> reference-0000001
    """
    if isinstance(value, str) and value.startswith("llm-"):
        return value[len("llm-"):]
    return value


def update_json_data(data: dict) -> bool:
    """
    JSON 객체에서 다음을 수정한다.
    1. 최상위 id 값에서 'llm-' 제거
    2. metadata.name 값 변경

    반환값:
        True  -> 변경 발생
        False -> 변경 없음
    """
    changed = False

    # 1) id 수정
    if "id" in data:
        new_id = remove_llm_prefix_from_id(data["id"])
        if new_id != data["id"]:
            data["id"] = new_id
            changed = True

    # 2) metadata.name 수정
    metadata = data.get("metadata")
    if isinstance(metadata, dict):
        if metadata.get("name") == OLD_NAME:
            metadata["name"] = NEW_NAME
            changed = True

    return changed


def process_file(file_path: str) -> str:
    """
    파일 하나를 읽어서 JSON 수정 후 덮어쓴다.
    반환값:
        'updated' / 'skipped' / 'error'
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, dict):
            print(f"[SKIP] JSON 객체(dict) 형식이 아님: {os.path.basename(file_path)}")
            return "skipped"

        changed = update_json_data(data)

        if not changed:
            print(f"[SKIP] 변경 사항 없음: {os.path.basename(file_path)}")
            return "skipped"

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        print(f"[OK] 수정 완료: {os.path.basename(file_path)}")
        return "updated"

    except json.JSONDecodeError:
        print(f"[ERROR] JSON 파일이 아님 또는 형식 오류: {os.path.basename(file_path)}")
        return "error"
    except Exception as e:
        print(f"[ERROR] 처리 실패: {os.path.basename(file_path)} | {e}")
        return "error"


def main():
    print("=== JSON id / metadata.name 일괄 수정 스크립트 ===")
    folder_path = input("처리할 폴더 경로를 입력하세요: ").strip().strip('"').strip("'")

    if not os.path.isdir(folder_path):
        print(f"[ERROR] 유효한 폴더가 아닙니다: {folder_path}")
        return

    files = os.listdir(folder_path)

    updated_count = 0
    skipped_count = 0
    error_count = 0

    for filename in files:
        file_path = os.path.join(folder_path, filename)

        if not os.path.isfile(file_path):
            continue

        result = process_file(file_path)

        if result == "updated":
            updated_count += 1
        elif result == "skipped":
            skipped_count += 1
        else:
            error_count += 1

    print("\n=== 작업 완료 ===")
    print(f"수정 완료: {updated_count}")
    print(f"건너뜀:   {skipped_count}")
    print(f"오류:     {error_count}")


if __name__ == "__main__":
    main()