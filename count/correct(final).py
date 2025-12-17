#json내에 meassages를 messages로 일괄 수정하는 코드드


import os
import re
import json
import csv
import sys
from pathlib import Path
from datetime import datetime
from typing import Tuple, Any, Union

def read_text_best_effort(path: Path) -> Tuple[str, str]:
    """
    파일을 최선의 인코딩으로 읽어 텍스트를 반환.
    return: (text, used_encoding)
    """
    encodings = ["utf-8", "utf-8-sig", "cp949", "euc-kr", "latin-1"]
    last_err = None
    for enc in encodings:
        try:
            return path.read_text(encoding=enc), enc
        except Exception as e:
            last_err = e
    raise last_err  # 모두 실패한 경우 예외

def write_text(path: Path, text: str, encoding: str = "utf-8") -> None:
    path.write_text(text, encoding=encoding, newline="\n")

def rename_key_in_json_obj(obj: Any, old_key: str, new_key: str) -> Tuple[Any, int]:
    """
    파싱된 JSON 객체에서 키 old_key -> new_key로 재귀적으로 변경.
    반환: (수정된 객체, 변경된 키 개수)
    """
    changed = 0
    if isinstance(obj, dict):
        new_dict = {}
        for k, v in obj.items():
            nk = new_key if k == old_key else k
            if k == old_key:
                changed += 1
            nv, c = rename_key_in_json_obj(v, old_key, new_key)
            changed += c
            new_dict[nk] = nv
        return new_dict, changed
    elif isinstance(obj, list):
        new_list = []
        for item in obj:
            ni, c = rename_key_in_json_obj(item, old_key, new_key)
            changed += c
            new_list.append(ni)
        return new_list, changed
    else:
        return obj, changed

def regex_safe_key_replace(json_text: str, old_key: str, new_key: str) -> Tuple[str, int]:
    """
    JSON 텍스트에서 '키:값' 형태의 키 이름만 안전하게 치환.
    값 문자열 내부의 'meassages'는 변경하지 않음.
    """
    pattern = re.compile(r'(?P<q>["\'])' + re.escape(old_key) + r'(?P=q)\s*:')
    # count를 알기 위해 subn 사용
    replaced_text, n = pattern.subn(r'\g<q>' + new_key + r'\g<q>:', json_text)
    return replaced_text, n

def process_single_json(path: Path, old_key: str = "meassages", new_key: str = "messages") -> Tuple[bool, int, Union[str, None]]:
    """
    단일 JSON 파일 처리.
    반환: (modified, num_changes, error_message)
    """
    try:
        text, enc = read_text_best_effort(path)
    except Exception as e:
        return False, 0, f"READ_ERROR: {e}"

    # 1) JSON 파싱 후 안전 변경 시도
    try:
        data = json.loads(text)
        new_data, count_changes = rename_key_in_json_obj(data, old_key, new_key)
        if count_changes > 0:
            # 보기 좋은 포맷으로 덮어쓰기 (ensure_ascii=False, indent=2)
            try:
                dumped = json.dumps(new_data, ensure_ascii=False, indent=2)
                write_text(path, dumped, encoding="utf-8")
                return True, count_changes, None
            except Exception as e:
                return False, 0, f"WRITE_ERROR(JSON_DUMP): {e}"
        else:
            # 2) 파싱 성공했지만 바꿀 키 없음 → 수정 없음
            return False, 0, None
    except Exception:
        # 3) 파싱 실패 시, 정규식으로 '키:값' 형태만 치환
        try:
            replaced_text, n = regex_safe_key_replace(text, old_key, new_key)
            if n > 0:
                try:
                    write_text(path, replaced_text, encoding="utf-8")
                    return True, n, None
                except Exception as e:
                    return False, 0, f"WRITE_ERROR(REGEX): {e}"
            else:
                return False, 0, None
        except Exception as e:
            return False, 0, f"REGEX_ERROR: {e}"

def get_desktop_path() -> Path:
    # 대부분 환경에서 동작. 특수한 경우는 환경변수 USERPROFILE 등을 활용.
    desktop = Path.home() / "Desktop"
    return desktop if desktop.exists() else Path.home()

def main():
    print("=== JSON 키 이름 일괄 수정기: 'meassages' → 'messages' ===")
    root_input = input("대상 최상위 폴더 경로를 입력하세요: ").strip().strip('"')
    if not root_input:
        print("경로가 입력되지 않았습니다. 종료합니다.")
        sys.exit(1)

    root = Path(root_input)
    if not root.exists() or not root.is_dir():
        print(f"유효하지 않은 폴더 경로입니다: {root}")
        sys.exit(1)

    json_files = list(root.rglob("*.json"))
    if not json_files:
        print("대상 폴더 하위에서 JSON 파일을 찾지 못했습니다.")
        sys.exit(0)

    print(f"발견된 JSON 파일 수: {len(json_files)}")
    modified_records = []  # (file_path, changes)
    errors = []            # (file_path, error)

    for idx, path in enumerate(json_files, 1):
        modified, count_changes, err = process_single_json(path)
        if modified:
            modified_records.append((str(path), count_changes))
            print(f"[{idx}/{len(json_files)}] 수정됨: {path} (변경된 키 수: {count_changes})")
        else:
            if err:
                errors.append((str(path), err))
                print(f"[{idx}/{len(json_files)}] 오류: {path} ({err})")
            else:
                print(f"[{idx}/{len(json_files)}] 변경 없음: {path}")

    # 리포트 CSV (수정된 파일만)
    desktop = get_desktop_path()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = desktop / f"json_key_fix_report_{timestamp}.csv"

    try:
        with report_path.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(["file_path", "changed_key_count"])
            for fp, cnt in modified_records:
                writer.writerow([fp, cnt])
        print(f"\n수정된 파일 리포트를 저장했습니다: {report_path}")
        if errors:
            # 오류도 참고용으로 별도 CSV 저장
            err_path = desktop / f"json_key_fix_errors_{timestamp}.csv"
            with err_path.open("w", newline="", encoding="utf-8-sig") as f:
                writer = csv.writer(f)
                writer.writerow(["file_path", "error"])
                for fp, e in errors:
                    writer.writerow([fp, e])
            print(f"오류 리포트를 저장했습니다: {err_path}")
    except Exception as e:
        print(f"리포트 저장 중 오류가 발생했습니다: {e}")

    print("\n=== 작업 완료 ===")
    print(f"- 총 JSON 파일: {len(json_files)}")
    print(f"- 수정된 파일: {len(modified_records)}")
    print(f"- 오류 발생 파일: {len(errors)}")

if __name__ == "__main__":
    main()
