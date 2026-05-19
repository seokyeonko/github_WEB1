#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
출처 포함 QA 데이터 통합 검사기
- 모든 JSON 파일에 대해 아래 3가지를 한 번에 모두 검사함.

검사 항목
1. citation_exists:
   - assistant text value 안의 참조 표기 [33:33], [30:30, 34:34] 등이
     tool text의 <passage id=...>에 실제 존재하는지 기계적으로 검사

2. grounded_on_tool:
   - assistant text value가 tool text 내용에 기반해 작성되었는지 LLM으로 검사

3. answer_to_question:
   - assistant text value가 user text value(질문)에 적절히 답했는지 LLM으로 검사

출력
- result.csv (UTF-8, BOM 없음)
- 각 항목별 T/F와 상세 이유 저장
"""

import os
import re
import json
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv

# =========================
# 설정
# =========================
load_dotenv()
API_KEY = os.environ.get("OPENAI_API_KEY")
MODEL = "GPT-5-nano"
OPENAI_TIMEOUT = 60
SLEEP_BETWEEN_REQUESTS = 0.2

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


def openai_client() -> OpenAI:
    if not API_KEY or not API_KEY.startswith("sk-"):
        raise RuntimeError("OPENAI_API_KEY 환경변수가 없거나 형식이 올바르지 않습니다.")
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


def evaluate_citation_existence(answer: Optional[str], passages: Dict[int, str]) -> Tuple[str, str]:
    if not answer:
        return "F", "assistant text value가 없습니다."

    citation_ids, bad_chunks = extract_all_citation_ids(answer)

    if not citation_ids and not bad_chunks:
        return "F", "assistant text value에서 인용 표기([i], [i:j], [i:j, k:l] 등)를 찾지 못했습니다."

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
        return "F", " / ".join(reasons)

    return "T", f"assistant text에 등장한 인용 번호 {sorted(citation_ids)}가 모두 tool text의 passage id에 존재합니다."


# =========================
# LLM 판정 공통
# =========================
def llm_judge_tf(client: OpenAI, system_prompt: str, user_prompt: str) -> Tuple[str, str]:
    resp = client.responses.create(
        model=MODEL,
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.0,
        timeout=OPENAI_TIMEOUT,
    )

    out = getattr(resp, "output_text", None)
    if not out:
        out = str(resp)
    out = out.strip()

    decision_match = re.search(r"decision\s*[:：]\s*([TF])", out, re.IGNORECASE)
    reason_match = re.search(r"reason\s*[:：]\s*(.*)", out, re.IGNORECASE | re.DOTALL)

    decision = decision_match.group(1).upper() if decision_match else "F"
    reason = reason_match.group(1).strip() if reason_match else out
    reason = normalize_ws(reason)

    return decision, reason


# =========================
# 2) tool grounding 검사
# =========================
def build_tool_evidence_text(passages: Dict[int, str], max_chars: int = 20000) -> str:
    chunks = []
    total = 0

    for pid in sorted(passages.keys()):
        block = f"[{pid}] {passages[pid]}"
        if total + len(block) > max_chars:
            remain = max_chars - total
            if remain > 100:
                chunks.append(block[:remain])
            break
        chunks.append(block)
        total += len(block) + 1

    return "\n".join(chunks)


def evaluate_grounded_on_tool(client: OpenAI, answer: Optional[str], passages: Dict[int, str]) -> Tuple[str, str]:
    if not answer:
        return "F", "assistant text value가 없습니다."

    if not passages:
        return "F", "tool text에서 추출한 passage가 없어 답변의 근거성을 판단할 수 없습니다."

    sys_prompt = (
        "너는 QA 데이터 품질 검사자다. "
        "assistant 답변이 주어진 tool passage 내용에 근거하고 있는지 평가하라. "
        "답변의 핵심 주장들이 passage에서 직접 확인되거나 합리적으로 뒷받침되면 T, "
        "핵심 주장 중 중요한 부분이 passage에 없거나 passage와 어긋나면 F로 판단하라. "
        "반드시 아래 형식만 출력하라.\n"
        "decision: T 또는 F\n"
        "reason: 상세 이유"
    )

    evidence_text = build_tool_evidence_text(passages)

    user_prompt = (
        "다음 assistant 답변이 tool passage 내용에 근거하는지 평가하라.\n\n"
        f"[assistant answer]\n{answer}\n\n"
        f"[tool passages]\n{evidence_text}\n\n"
        "주의:\n"
        "- 답변의 일부가 맞더라도 핵심 내용이 passage로 뒷받침되지 않으면 F\n"
        "- F일 경우, 어떤 주장 또는 문단이 근거 부족인지 구체적으로 설명\n"
        "- passage에 없는 내용을 사실처럼 단정하면 F"
    )

    return llm_judge_tf(client, sys_prompt, user_prompt)


# =========================
# 3) 질문 적절성 검사
# =========================
def evaluate_answer_to_question(client: OpenAI, question: Optional[str], answer: Optional[str]) -> Tuple[str, str]:
    if not question:
        return "F", "user text value(질문)가 없습니다."
    if not answer:
        return "F", "assistant text value(답변)가 없습니다."

    sys_prompt = (
        "너는 QA 데이터 품질 검사자다. "
        "assistant 답변이 user 질문에 적절히 답하고 있는지 평가하라. "
        "질문의 핵심 요구사항을 충족하면 T, "
        "중요 항목 누락, 불완전한 답변, 엉뚱한 답변이면 F로 판단하라. "
        "반드시 아래 형식만 출력하라.\n"
        "decision: T 또는 F\n"
        "reason: 상세 이유"
    )

    user_prompt = (
        "다음 질문과 답변을 보고 답변 적절성을 평가하라.\n\n"
        f"[user question]\n{question}\n\n"
        f"[assistant answer]\n{answer}\n\n"
        "주의:\n"
        "- 질문이 여러 하위 요구를 포함하면, 중요한 요구사항의 충족 여부를 따져라\n"
        "- 답변이 일부만 답했으면 F 가능성이 높다\n"
        "- F일 경우, 누락된 항목과 부족한 점을 구체적으로 적어라"
    )

    return llm_judge_tf(client, sys_prompt, user_prompt)


# =========================
# 메인
# =========================
def main():
    print("=== 출처 포함 QA 데이터 통합 검사기 ===")
    print("이 코드는 검사 3가지를 모두 자동으로 수행합니다.")
    print("1) citation 존재 여부 검사")
    print("2) tool 근거 기반 답변 검사")
    print("3) 질문 적절 답변 검사")
    print()

    in_dir = ask_path("검사할 JSON 폴더 경로를 입력하세요: ")
    out_dir = ask_path("검사 결과를 저장할 폴더 경로를 입력하세요: ")

    if not os.path.isdir(in_dir):
        print(f"❌ 폴더를 찾을 수 없습니다: {in_dir}")
        sys.exit(1)

    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, "result.csv")

    try:
        client = openai_client()
    except Exception as e:
        print(f"❌ OpenAI 초기화 실패: {e}")
        sys.exit(1)

    json_files = [
        os.path.join(in_dir, f)
        for f in os.listdir(in_dir)
        if f.lower().endswith(".json")
    ]

    if not json_files:
        print("⚠️ JSON 파일이 없습니다.")
        sys.exit(0)

    rows = []

    for path in tqdm(json_files, desc="검사 중"):
        try:
            obj = read_json(path)
            parsed = parse_file(obj)

            # 1) 참조 존재 여부 검사
            citation_tf, citation_reason = evaluate_citation_existence(
                parsed.answer,
                parsed.passages
            )

            # 2) tool 근거 기반 검사
            grounded_tf, grounded_reason = evaluate_grounded_on_tool(
                client,
                parsed.answer,
                parsed.passages
            )

            # 3) 질문 적절 답변 검사
            qa_tf, qa_reason = evaluate_answer_to_question(
                client,
                parsed.question,
                parsed.answer
            )

            row = {
                "file": os.path.basename(path),
                "question": parsed.question or "",
                "answer": parsed.answer or "",
                "passage_count": len(parsed.passages),

                "citation_exists_tf": citation_tf,
                "citation_exists_reason": citation_reason,

                "grounded_on_tool_tf": grounded_tf,
                "grounded_on_tool_reason": grounded_reason,

                "answer_to_question_tf": qa_tf,
                "answer_to_question_reason": qa_reason,
            }

        except Exception as e:
            err = f"오류: {str(e)}"
            row = {
                "file": os.path.basename(path),
                "question": "",
                "answer": "",
                "passage_count": 0,

                "citation_exists_tf": "F",
                "citation_exists_reason": err,

                "grounded_on_tool_tf": "F",
                "grounded_on_tool_reason": err,

                "answer_to_question_tf": "F",
                "answer_to_question_reason": err,
            }

        rows.append(row)
        time.sleep(SLEEP_BETWEEN_REQUESTS)

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False, encoding="utf-8")

    print()
    print(f"✅ 완료: {out_csv}")
    print("모든 JSON 파일에 대해 3가지 검사를 한 번에 수행했습니다.")


if __name__ == "__main__":
    main()