#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
출처 포함 QA 데이터 통합 검사기 (최적화 + 오류 분리 처리 버전)

검사 항목
1. citation_exists:
   - assistant text value 안의 참조 표기 [33:33], [30:30, 34:34] 등이
     tool text의 <passage id=...>에 실제 존재하는지 기계적으로 검사

2. llm_combined_check:
   - grounded_on_tool: assistant text value가 tool text 내용에 기반해 작성되었는지
   - answer_to_question: assistant text value가 user text value(질문)에 적절히 답했는지

최적화 내용
- LLM 호출 2회 -> 1회 통합
- 인용된 passage 우선 사용
- passage별 최대 길이 제한
- reason 길이 제한
- 병렬 처리 지원
- question/answer 전체 저장 여부 옵션 제공

중요 수정 사항
- gpt-5-nano 호출 시 temperature 제거
- citation 검사와 LLM 검사 예외 처리 분리
- LLM 실패 시에도 citation 결과와 실제 passage_count 유지
"""

import os
import re
import json
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from tqdm import tqdm

# =========================
# 설정
from dotenv import load_dotenv
load_dotenv()
API_KEY = os.environ.get("OPENAI_API_KEY")
MODEL = "gpt-5-nano"
OPENAI_TIMEOUT = 60

# 병렬 처리
MAX_WORKERS = 4

# 결과 저장 옵션
SAVE_FULL_TEXT = False
QUESTION_PREVIEW_CHARS = 500
ANSWER_PREVIEW_CHARS = 700

# 근거 passage 선택/압축 옵션
MAX_EVIDENCE_PASSAGES = 8
PASSAGE_PER_ITEM_MAX_CHARS = 1000
FALLBACK_PASSAGES_WHEN_NO_CITATION = 5

# reason 길이 제한
REASON_MAX_CHARS = 250

print("Python executable:", sys.executable)
print("Python version:", sys.version)

try:
    from openai import OpenAI
except Exception as e:
    print(f"❌ openai 패키지 import 실패: {e}")
    print("먼저 아래를 실행해 보세요:")
    print("pip install -U openai pandas tqdm")
    sys.exit(1)


# =========================
# 데이터 구조
# =========================
@dataclass
class ParsedFile:
    question: Optional[str]
    answer: Optional[str]
    passages: Dict[int, str]
    tool_raw_texts: List[str]


# =========================
# 유틸
# =========================
def ask_path(prompt: str) -> str:
    p = input(prompt).strip().strip('"').strip("'")
    return os.path.abspath(p)


def read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def shorten_text(text: Optional[str], limit: int) -> str:
    text = text or ""
    if len(text) <= limit:
        return text
    return text[:limit] + " ...[truncated]"


def openai_client() -> OpenAI:
    if not API_KEY or not API_KEY.startswith("sk-"):
        raise RuntimeError("API_KEY가 비어 있거나 형식이 올바르지 않습니다.")
    return OpenAI(api_key=API_KEY)


# =========================
# content 추출
# =========================
def extract_text_from_content(content: Any) -> Optional[str]:
    if isinstance(content, str):
        t = normalize_ws(content)
        return t if t else None

    if isinstance(content, list):
        texts = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text" and isinstance(item.get("value"), str):
                    t = normalize_ws(item["value"])
                    if t:
                        texts.append(t)
            elif isinstance(item, str):
                t = normalize_ws(item)
                if t:
                    texts.append(t)
        if texts:
            return " ".join(texts)

    return None


def extract_assistant_text_only(content: Any) -> Optional[str]:
    if isinstance(content, str):
        t = normalize_ws(content)
        return t if t else None

    if isinstance(content, list):
        texts = []
        for item in content:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "text" and isinstance(item.get("value"), str):
                t = normalize_ws(item["value"])
                if t:
                    texts.append(t)
        if texts:
            return " ".join(texts)

    return None


def extract_tool_texts(content: Any) -> List[str]:
    results = []

    if isinstance(content, str):
        t = normalize_ws(content)
        if t:
            results.append(t)
        return results

    if isinstance(content, list):
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text" and isinstance(item.get("value"), str):
                    t = normalize_ws(item["value"])
                    if t:
                        results.append(t)
            elif isinstance(item, str):
                t = normalize_ws(item)
                if t:
                    results.append(t)

    return results


# =========================
# passage 추출
# =========================
PASSAGE_RE = re.compile(
    r"<passage\s+id\s*=\s*(\d+)[^>]*>(.*?)</passage>",
    re.IGNORECASE | re.DOTALL
)

def extract_passages_from_tool_text(tool_text: str) -> Dict[int, str]:
    passages = {}
    for m in PASSAGE_RE.finditer(tool_text):
        pid = int(m.group(1))
        ptxt = normalize_ws(m.group(2))
        if ptxt:
            passages[pid] = ptxt
    return passages


# =========================
# 파일 파싱
# =========================
def try_extract_from_simple_schema(obj: Any) -> ParsedFile:
    question = None
    answer = None
    passages = {}
    tool_raw_texts = []

    if isinstance(obj, dict):
        question = obj.get("question")
        answer = obj.get("answer")

        sources = obj.get("sources") or obj.get("passages")
        if isinstance(sources, list):
            for s in sources:
                try:
                    pid = int(s.get("id"))
                    ptxt = normalize_ws(str(s.get("text", "")))
                    if ptxt:
                        passages[pid] = ptxt
                except Exception:
                    pass

    return ParsedFile(
        question=question,
        answer=answer,
        passages=passages,
        tool_raw_texts=tool_raw_texts
    )


def parse_messages_for_qa_and_passages(messages: List[Dict[str, Any]]) -> ParsedFile:
    user_texts: List[str] = []
    assistant_texts: List[str] = []
    tool_raw_texts: List[str] = []
    passages: Dict[int, str] = {}

    for msg in messages:
        role = msg.get("role")
        content = msg.get("content")

        if role == "user":
            t = extract_text_from_content(content)
            if t:
                user_texts.append(t)

        elif role == "assistant":
            t = extract_assistant_text_only(content)
            if t:
                assistant_texts.append(t)

        elif role == "tool":
            texts = extract_tool_texts(content)
            for t in texts:
                tool_raw_texts.append(t)
                extracted = extract_passages_from_tool_text(t)
                passages.update(extracted)

    question = user_texts[-1] if user_texts else None
    answer = assistant_texts[-1] if assistant_texts else None

    return ParsedFile(
        question=question,
        answer=answer,
        passages=passages,
        tool_raw_texts=tool_raw_texts
    )


def parse_file(obj: Any) -> ParsedFile:
    if isinstance(obj, dict) and isinstance(obj.get("messages"), list):
        parsed = parse_messages_for_qa_and_passages(obj["messages"])
        if not parsed.question or not parsed.answer:
            fallback = try_extract_from_simple_schema(obj)
            return ParsedFile(
                question=parsed.question or fallback.question,
                answer=parsed.answer or fallback.answer,
                passages=parsed.passages or fallback.passages,
                tool_raw_texts=parsed.tool_raw_texts or fallback.tool_raw_texts
            )
        return parsed

    return try_extract_from_simple_schema(obj)


# =========================
# 1) citation existence 검사
# =========================
BRACKET_RE = re.compile(r"\[([^\[\]]+)\]")

def parse_single_citation_token(token: str) -> List[int]:
    token = token.strip()

    if re.fullmatch(r"\d+", token):
        return [int(token)]

    m = re.fullmatch(r"(\d+)\s*:\s*(\d+)", token)
    if m:
        s = int(m.group(1))
        e = int(m.group(2))
        if s <= e:
            return list(range(s, e + 1))
        return list(range(e, s + 1))

    return []


def extract_all_citation_ids(answer_text: str) -> Tuple[Set[int], List[str]]:
    found_ids: Set[int] = set()
    bad_chunks: List[str] = []

    for m in BRACKET_RE.finditer(answer_text or ""):
        inside = m.group(1).strip()
        parts = [p.strip() for p in inside.split(",") if p.strip()]
        if not parts:
            continue

        local_ids: List[int] = []
        local_ok = True

        for part in parts:
            ids = parse_single_citation_token(part)
            if not ids:
                local_ok = False
            else:
                local_ids.extend(ids)

        if local_ids:
            found_ids.update(local_ids)

        if not local_ok:
            bad_chunks.append(m.group(0))

    return found_ids, bad_chunks


def evaluate_citation_existence(answer: Optional[str], passages: Dict[int, str]) -> Tuple[str, str, Set[int]]:
    if not answer:
        return "F", "assistant text value가 없습니다.", set()

    citation_ids, bad_chunks = extract_all_citation_ids(answer)

    if not citation_ids and not bad_chunks:
        return "F", "assistant text value에서 인용 표기([i], [i:j], [i:j, k:l] 등)를 찾지 못했습니다.", set()

    existing_ids = set(passages.keys())
    missing_ids = sorted([cid for cid in citation_ids if cid not in existing_ids])

    reasons = []

    if bad_chunks:
        reasons.append("일부 인용 표기를 해석하지 못했습니다: " + ", ".join(bad_chunks))

    if missing_ids:
        reasons.append(
            "assistant text에 등장한 인용 번호 중 tool text의 passage id에 존재하지 않는 번호가 있습니다: "
            + ", ".join(map(str, missing_ids))
        )

    if not passages:
        reasons.append("tool text에서 <passage id=...>...</passage>를 추출하지 못했습니다.")

    if reasons:
        return "F", " / ".join(reasons), citation_ids

    return "T", f"assistant text에 등장한 인용 번호 {sorted(citation_ids)}가 모두 tool text의 passage id에 존재합니다.", citation_ids


# =========================
# evidence 구성
# =========================
def trim_passage(text: str, max_chars: int = PASSAGE_PER_ITEM_MAX_CHARS) -> str:
    text = normalize_ws(text)
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + " ...[truncated]"


def choose_evidence_passages(
    answer: Optional[str],
    passages: Dict[int, str],
    cited_ids: Set[int],
    max_passages: int = MAX_EVIDENCE_PASSAGES,
) -> List[Tuple[int, str]]:
    selected: List[Tuple[int, str]] = []

    for pid in sorted(cited_ids):
        if pid in passages:
            selected.append((pid, trim_passage(passages[pid])))

    if selected:
        return selected[:max_passages]

    for pid in sorted(passages.keys())[:min(FALLBACK_PASSAGES_WHEN_NO_CITATION, max_passages)]:
        selected.append((pid, trim_passage(passages[pid])))

    return selected[:max_passages]


def build_evidence_text(selected_passages: List[Tuple[int, str]]) -> str:
    return "\n".join([f"[{pid}] {text}" for pid, text in selected_passages])


# =========================
# 통합 LLM 판정
# =========================
def parse_combined_judgment(text: str) -> Tuple[str, str, str, str]:
    grounded_tf = "F"
    grounded_reason = "LLM 출력 파싱 실패"
    qa_tf = "F"
    qa_reason = "LLM 출력 파싱 실패"

    m1 = re.search(r"grounded_on_tool\s*[:：]\s*([TF])", text, re.IGNORECASE)
    m2 = re.search(r"grounded_reason\s*[:：]\s*(.*?)(?=\nanswer_to_question\s*[:：]|\Z)", text, re.IGNORECASE | re.DOTALL)
    m3 = re.search(r"answer_to_question\s*[:：]\s*([TF])", text, re.IGNORECASE)
    m4 = re.search(r"answer_reason\s*[:：]\s*(.*)", text, re.IGNORECASE | re.DOTALL)

    if m1:
        grounded_tf = m1.group(1).upper()
    if m2:
        grounded_reason = normalize_ws(m2.group(1))
    if m3:
        qa_tf = m3.group(1).upper()
    if m4:
        qa_reason = normalize_ws(m4.group(1))

    grounded_reason = shorten_text(grounded_reason, REASON_MAX_CHARS)
    qa_reason = shorten_text(qa_reason, REASON_MAX_CHARS)

    return grounded_tf, grounded_reason, qa_tf, qa_reason


def evaluate_combined_llm(
    client: OpenAI,
    question: Optional[str],
    answer: Optional[str],
    passages: Dict[int, str],
    cited_ids: Set[int],
) -> Tuple[str, str, str, str]:
    if not answer and not question:
        return (
            "F", "assistant text와 user text가 모두 없습니다.",
            "F", "assistant text와 user text가 모두 없습니다."
        )
    if not answer:
        return (
            "F", "assistant text value가 없습니다.",
            "F", "assistant text value가 없습니다."
        )
    if not question:
        return (
            "F", "user text value(질문)가 없습니다.",
            "F", "user text value(질문)가 없습니다."
        )
    if not passages:
        return (
            "F", "tool text에서 추출한 passage가 없어 답변의 근거성을 판단할 수 없습니다.",
            "F", "질문은 있으나 tool passages가 없어 근거 검증이 불가능합니다."
        )

    selected_passages = choose_evidence_passages(
        answer=answer,
        passages=passages,
        cited_ids=cited_ids,
        max_passages=MAX_EVIDENCE_PASSAGES,
    )
    evidence_text = build_evidence_text(selected_passages)

    system_prompt = (
        "너는 QA 데이터 품질 검사자다. "
        "주어진 질문, 답변, tool passages를 보고 아래 두 항목을 동시에 평가하라. "
        "1) grounded_on_tool: 답변의 핵심 주장들이 tool passages에 의해 직접 확인되거나 합리적으로 뒷받침되는가 "
        "2) answer_to_question: 답변이 질문의 핵심 요구사항에 적절히 답하는가 "
        "반드시 아래 4줄만 출력하라. 다른 문장은 절대 출력하지 마라.\n"
        "grounded_on_tool: T 또는 F\n"
        "grounded_reason: 한국어 2~3문장, 최대 250자 이내\n"
        "answer_to_question: T 또는 F\n"
        "answer_reason: 한국어 2~3문장, 최대 250자 이내"
    )

    user_prompt = (
        "[user question]\n"
        f"{question}\n\n"
        "[assistant answer]\n"
        f"{answer}\n\n"
        "[tool passages]\n"
        f"{evidence_text}\n\n"
        "판정 기준:\n"
        "- grounded_on_tool은 답변의 핵심 주장 기준으로 판정한다.\n"
        "- answer_to_question은 질문의 핵심 요구사항 충족 여부 기준으로 판정한다.\n"
        "- 일부만 맞아도 중요한 누락이 있으면 F 가능.\n"
        "- passage에 없는 내용을 사실처럼 단정하면 grounded_on_tool은 F.\n"
        "- 출력은 지정한 4줄 형식만 사용."
    )

    resp = client.responses.create(
        model=MODEL,
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        timeout=OPENAI_TIMEOUT,
    )

    out = getattr(resp, "output_text", None) or str(resp)
    out = out.strip()

    return parse_combined_judgment(out)


# =========================
# 파일 단위 처리
# =========================
def process_one_file(path: str) -> Dict[str, Any]:
    base_row = {
        "file": os.path.basename(path),
        "passage_count": 0,

        "citation_exists_tf": "F",
        "citation_exists_reason": "",

        "grounded_on_tool_tf": "F",
        "grounded_on_tool_reason": "",

        "answer_to_question_tf": "F",
        "answer_to_question_reason": "",
    }

    if SAVE_FULL_TEXT:
        base_row["question"] = ""
        base_row["answer"] = ""
    else:
        base_row["question_preview"] = ""
        base_row["answer_preview"] = ""

    # 1) 파일 읽기/파싱
    try:
        obj = read_json(path)
        parsed = parse_file(obj)
    except Exception as e:
        err = f"파일 파싱 오류: {str(e)}"
        base_row["citation_exists_reason"] = err
        base_row["grounded_on_tool_reason"] = err
        base_row["answer_to_question_reason"] = err
        return base_row

    # 실제 추출값 유지
    base_row["passage_count"] = len(parsed.passages)

    if SAVE_FULL_TEXT:
        base_row["question"] = parsed.question or ""
        base_row["answer"] = parsed.answer or ""
    else:
        base_row["question_preview"] = shorten_text(parsed.question or "", QUESTION_PREVIEW_CHARS)
        base_row["answer_preview"] = shorten_text(parsed.answer or "", ANSWER_PREVIEW_CHARS)

    # 2) citation 검사
    try:
        citation_tf, citation_reason, cited_ids = evaluate_citation_existence(
            parsed.answer,
            parsed.passages
        )
        base_row["citation_exists_tf"] = citation_tf
        base_row["citation_exists_reason"] = citation_reason
    except Exception as e:
        cited_ids = set()
        base_row["citation_exists_tf"] = "F"
        base_row["citation_exists_reason"] = f"citation 검사 오류: {str(e)}"

    # 3) LLM 통합 검사
    try:
        client = openai_client()
        grounded_tf, grounded_reason, qa_tf, qa_reason = evaluate_combined_llm(
            client=client,
            question=parsed.question,
            answer=parsed.answer,
            passages=parsed.passages,
            cited_ids=cited_ids,
        )
        base_row["grounded_on_tool_tf"] = grounded_tf
        base_row["grounded_on_tool_reason"] = grounded_reason
        base_row["answer_to_question_tf"] = qa_tf
        base_row["answer_to_question_reason"] = qa_reason
    except Exception as e:
        err = f"LLM 검사 오류: {str(e)}"
        base_row["grounded_on_tool_tf"] = "F"
        base_row["grounded_on_tool_reason"] = err
        base_row["answer_to_question_tf"] = "F"
        base_row["answer_to_question_reason"] = err

    return base_row


# =========================
# 메인
# =========================
def main():
    print("=== 출처 포함 QA 데이터 통합 검사기 (최적화 + 오류 분리 처리 버전) ===")
    print("적용된 내용:")
    print("1) citation 존재 여부 검사 (기계식)")
    print("2) grounded + QA 적절성 검사 (LLM 1회 통합)")
    print("3) cited passage 우선 사용")
    print("4) passage 길이 제한")
    print("5) 병렬 처리")
    print("6) citation/LLM 오류 분리 처리")
    print()

    in_dir = ask_path("검사할 JSON 폴더 경로를 입력하세요: ")
    out_dir = ask_path("검사 결과를 저장할 폴더 경로를 입력하세요: ")

    if not os.path.isdir(in_dir):
        print(f"❌ 폴더를 찾을 수 없습니다: {in_dir}")
        sys.exit(1)

    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, "result.csv")

    json_files = [
        os.path.join(in_dir, f)
        for f in os.listdir(in_dir)
        if f.lower().endswith(".json")
    ]

    if not json_files:
        print("⚠️ JSON 파일이 없습니다.")
        sys.exit(0)

    rows: List[Dict[str, Any]] = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_one_file, path): path for path in json_files}

        for future in tqdm(as_completed(futures), total=len(futures), desc="검사 중"):
            rows.append(future.result())

    rows.sort(key=lambda x: x.get("file", ""))

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False, encoding="utf-8")

    print()
    print(f"✅ 완료: {out_csv}")
    print("모든 JSON 파일에 대해 3가지 검사를 한 번에 수행했습니다.")
    print(f"병렬 worker 수: {MAX_WORKERS}")
    print(f"원문 전체 저장 여부: {SAVE_FULL_TEXT}")


if __name__ == "__main__":
    main()