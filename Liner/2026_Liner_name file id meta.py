#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
JSON + 파일명 일괄 수정 스크립트

기능
- 실행 시 대상 폴더 경로를 입력받음 (GUI 없음)
- 지정한 폴더 하위의 모든 .json 파일을 재귀적으로 탐색
- 아래 항목들을 일괄 수정
  1. 파일명에서 맨 앞 'llm-' 제거
     예: llm-reference-0000001.json -> reference-0000001.json
  2. 최상위 id 값에서 'llm-' 제거
     예: llm-reference-0000001 -> reference-0000001
  3. metadata.name 값 변경
     "AI 파운데이션 모델(LLM/LAM) 사후학습용 데이터"
     -> "AI 파운데이션 모델(LLM/LAM/LMM) 학습용 데이터"
  4. metadata.category 값 변경
     "출처를 포함한 질의 응답"
     -> "출처를 포함한 질의응답"

저장 방식
- 원본은 수정하지 않음
- 입력 폴더 아래 output 폴더를 생성하여 저장
- 원본 폴더 구조를 유지하여 output 안에 저장
- 저장 시 파일명도 변경 규칙을 적용
- 동일 파일명 존재 시 덮어쓰지 않고 오류 처리

예시
원본:
  /data/llm-a.json
  /data/sub/llm-b.json

결과:
  /data/output/a.json
  /data/output/sub/b.json
"""

import json
from pathlib import Path
from typing import Any


OLD_NAME = "AI 파운데이션 모델(LLM/LAM) 사후학습용 데이터"
NEW_NAME = "AI 파운데이션 모델(LLM/LAM/LMM) 학습용 데이터"

OLD_CATEGORY = "출처를 포함한 질의 응답"
NEW_CATEGORY = "출처를 포함한 질의응답"


def remove_llm_prefix(value: Any) -> Any:
    """
    문자열 값이 'llm-'로 시작하면 맨 앞의 'llm-'만 제거한다.
    """
    if isinstance(value, str) and value.startswith("llm-"):
        return value[len("llm-"):]
    return value


def rename_filename(filename: str) -> str:
    """
    파일명이 'llm-'로 시작하면 맨 앞의 'llm-'를 제거한다.
    예: llm-reference-0000001.json -> reference-0000001.json
    """
    if filename.startswith("llm-"):
        return filename.replace("llm-", "", 1)
    return filename


def find_json_files(root_dir: Path):
    """지정 폴더 하위의 모든 JSON 파일을 찾는다."""
    return list(root_dir.rglob("*.json"))


def update_json_data(data: dict) -> bool:
    """
    JSON 객체에서 아래 항목들을 수정한다.
    1. 최상위 id 값에서 'llm-' 제거
    2. metadata.name 값 변경
    3. metadata.category 값 변경

    반환값:
        True  -> 변경 발생
        False -> 변경 없음
    """
    changed = False

    if "id" in data:
        new_id = remove_llm_prefix(data["id"])
        if new_id != data["id"]:
            data["id"] = new_id
            changed = True

    metadata = data.get("metadata")
    if isinstance(metadata, dict):
        if metadata.get("name") == OLD_NAME:
            metadata["name"] = NEW_NAME
            changed = True

        if metadata.get("category") == OLD_CATEGORY:
            metadata["category"] = NEW_CATEGORY
            changed = True

    return changed


def process_json_file(src_path: Path, root_dir: Path, output_dir: Path):
    """
    JSON 파일을 읽어 수정 후 output 폴더에 저장한다.
    저장 시 파일명에 대해서도 'llm-' 제거 규칙을 적용한다.
    """
    rel_path = src_path.relative_to(root_dir)
    new_filename = rename_filename(rel_path.name)
    dst_path = output_dir / rel_path.parent / new_filename
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with src_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        return {
            "file": str(src_path),
            "status": "read_error",
            "message": f"JSON 형식 오류: {e}"
        }
    except Exception as e:
        return {
            "file": str(src_path),
            "status": "read_error",
            "message": f"읽기 실패: {e}"
        }

    if not isinstance(data, dict):
        return {
            "file": str(src_path),
            "status": "skipped",
            "message": "최상위 JSON 객체(dict) 형식이 아님"
        }

    content_changed = update_json_data(data)
    filename_changed = (new_filename != rel_path.name)

    if dst_path.exists():
        return {
            "file": str(src_path),
            "status": "write_error",
            "message": f"저장 대상 파일이 이미 존재함: {dst_path}"
        }

    try:
        with dst_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            f.write("\n")
    except Exception as e:
        return {
            "file": str(src_path),
            "status": "write_error",
            "message": f"저장 실패: {e}"
        }

    if content_changed or filename_changed:
        return {
            "file": str(src_path),
            "output": str(dst_path),
            "status": "changed",
            "message": "파일명 또는 내용 수정 완료"
        }

    return {
        "file": str(src_path),
        "output": str(dst_path),
        "status": "copied",
        "message": "변경 대상 없음, 그대로 저장"
    }


def main():
    print("=" * 72)
    print("JSON 내용 + 파일명 일괄 수정기")
    print("=" * 72)
    print("[1] 파일명 변경")
    print('  - "llm-파일명.json" -> "파일명.json"')
    print("[2] JSON 최상위 id 변경")
    print('  - "llm-..." -> "..."')
    print("[3] metadata.name 변경")
    print(f'  - "{OLD_NAME}"')
    print(f'  -> "{NEW_NAME}"')
    print("[4] metadata.category 변경")
    print(f'  - "{OLD_CATEGORY}"')
    print(f'  -> "{NEW_CATEGORY}"')
    print()

    input_path_str = input("처리할 폴더 경로를 입력하세요: ").strip().strip('"').strip("'")
    if not input_path_str:
        print("폴더 경로가 입력되지 않았습니다.")
        return

    root_dir = Path(input_path_str).expanduser().resolve()

    if not root_dir.exists():
        print(f"오류: 경로가 존재하지 않습니다.\n- {root_dir}")
        return

    if not root_dir.is_dir():
        print(f"오류: 폴더가 아닙니다.\n- {root_dir}")
        return

    output_dir = root_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    json_files = find_json_files(root_dir)

    # output 폴더 내부 파일은 다시 처리하지 않도록 제외
    json_files = [p for p in json_files if output_dir not in p.parents and p != output_dir]

    if not json_files:
        print("JSON 파일을 찾지 못했습니다.")
        return

    total = len(json_files)
    changed_count = 0
    copied_count = 0
    skipped_count = 0
    error_count = 0

    print()
    print(f"총 {total}개의 JSON 파일을 처리합니다.")
    print()

    for idx, file_path in enumerate(json_files, start=1):
        result = process_json_file(file_path, root_dir, output_dir)
        status = result["status"]

        if status == "changed":
            changed_count += 1
            print(f"[{idx}/{total}] 수정됨      - {result['file']}")
            print(f"           저장: {result['output']}")
        elif status == "copied":
            copied_count += 1
            print(f"[{idx}/{total}] 그대로저장  - {result['file']}")
            print(f"           저장: {result['output']}")
        elif status == "skipped":
            skipped_count += 1
            print(f"[{idx}/{total}] 건너뜀      - {result['file']}")
            print(f"           {result['message']}")
        else:
            error_count += 1
            print(f"[{idx}/{total}] 오류        - {result['file']}")
            print(f"           {result['message']}")

    print()
    print("=" * 72)
    print("처리 완료")
    print("=" * 72)
    print(f"총 파일 수        : {total}")
    print(f"수정된 파일 수    : {changed_count}")
    print(f"그대로 저장 수   : {copied_count}")
    print(f"건너뛴 파일 수    : {skipped_count}")
    print(f"오류 파일 수      : {error_count}")
    print(f"저장 폴더         : {output_dir}")


if __name__ == "__main__":
    main()