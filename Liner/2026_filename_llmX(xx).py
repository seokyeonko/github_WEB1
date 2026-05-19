#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

def rename_files(target_dir: str):
    if not os.path.isdir(target_dir):
        print(f"[ERROR] 유효한 폴더가 아닙니다: {target_dir}")
        return

    files = os.listdir(target_dir)
    renamed_count = 0

    for filename in files:
        # 조건: 'llm-'로 시작하는 파일만 처리
        if filename.startswith("llm-"):
            old_path = os.path.join(target_dir, filename)

            # 'llm-' 제거
            new_filename = filename.replace("llm-", "", 1)
            new_path = os.path.join(target_dir, new_filename)

            # 동일 파일명 존재 시 스킵 (덮어쓰기 방지)
            if os.path.exists(new_path):
                print(f"[SKIP] 이미 존재: {new_filename}")
                continue

            try:
                os.rename(old_path, new_path)
                print(f"[OK] {filename} → {new_filename}")
                renamed_count += 1
            except Exception as e:
                print(f"[ERROR] {filename} 처리 실패: {e}")

    print(f"\n총 변경된 파일 수: {renamed_count}")


if __name__ == "__main__":
    print("=== 파일명 'llm-' 제거 스크립트 ===")
    target_dir = input("변경할 폴더 경로를 입력하세요: ").strip()

    rename_files(target_dir)