#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
JSON 메타데이터 수정 스크립트

기능
- 실행 시 대상 폴더 경로를 입력받음 (GUI 없음)
- 지정한 폴더 하위의 모든 .json 파일을 재귀적으로 탐색
- metadata.category 값이
  "출처를 포함한 질의 응답"
  인 경우
  "출처를 포함한 질의응답"
  으로 변경
- 원본은 건드리지 않고, 입력한 폴더 아래 output 폴더를 만들어 저장
- 원본 폴더 구조를 최대한 유지하여 output 안에 저장

예시
원본:
  /data/a.json
  /data/sub/b.json

결과:
  /data/output/a.json
  /data/output/sub/b.json
"""

import json
from pathlib import Path


OLD_CATEGORY = "출처를 포함한 질의 응답"
NEW_CATEGORY = "출처를 포함한 질의응답"


def find_json_files(root_dir: Path):
    """지정 폴더 하위의 모든 JSON 파일을 찾는다."""
    return list(root_dir.rglob("*.json"))


def process_json_file(src_path: Path, root_dir: Path, output_dir: Path):
    """
    JSON 파일을 읽어 metadata.category를 수정한 뒤
    output 폴더에 같은 상대경로로 저장한다.
    """
    rel_path = src_path.relative_to(root_dir)
    dst_path = output_dir / rel_path
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with src_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        return {
            "file": str(src_path),
            "status": "read_error",
            "message": f"읽기 실패: {e}"
        }

    changed = False

    if isinstance(data, dict):
        metadata = data.get("metadata")
        if isinstance(metadata, dict):
            if metadata.get("category") == OLD_CATEGORY:
                metadata["category"] = NEW_CATEGORY
                changed = True

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

    return {
        "file": str(src_path),
        "output": str(dst_path),
        "status": "changed" if changed else "copied",
        "message": "수정 완료" if changed else "변경 대상 없음, 그대로 저장"
    }


def main():
    print("=" * 60)
    print("JSON metadata.category 일괄 수정기")
    print("=" * 60)
    print(f'변경 전: "{OLD_CATEGORY}"')
    print(f'변경 후: "{NEW_CATEGORY}"')
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
    error_count = 0

    print()
    print(f"총 {total}개의 JSON 파일을 처리합니다.")
    print()

    for idx, file_path in enumerate(json_files, start=1):
        result = process_json_file(file_path, root_dir, output_dir)

        status = result["status"]
        if status == "changed":
            changed_count += 1
            print(f"[{idx}/{total}] 수정됨   - {result['file']}")
        elif status == "copied":
            copied_count += 1
            print(f"[{idx}/{total}] 그대로저장 - {result['file']}")
        else:
            error_count += 1
            print(f"[{idx}/{total}] 오류     - {result['file']}")
            print(f"           {result['message']}")

    print()
    print("=" * 60)
    print("처리 완료")
    print("=" * 60)
    print(f"총 파일 수      : {total}")
    print(f"수정된 파일 수  : {changed_count}")
    print(f"그대로 저장 수 : {copied_count}")
    print(f"오류 파일 수    : {error_count}")
    print(f"저장 폴더       : {output_dir}")


if __name__ == "__main__":
    main()