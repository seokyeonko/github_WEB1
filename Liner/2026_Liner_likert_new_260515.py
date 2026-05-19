#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
출처 포함 QA 데이터 품질 검사기

평가 항목 (LLM 1회 통합 평가)
1. citation_matching_score (5점 만점)
   - 답변의 [doc_id:passage_id] 인용 표기가 실제 문서/패시지와 얼마나 정확하게 대응하는가
2. question_clarity_score (2점 만점)
   - 질문의 의도가 명확하게 전달되고, 문장 자체에 결함이 없는지
3. question_answerability_score (3점 만점)
   - 올바른 답변을 만들어낼 수 있는 질문인지
4. answer_completeness_score (2점 만점)
   - 답변이 질문에 대한 핵심 사항/요청을 모두 반영하고 완결된 형태인지
5. source_based_answer_accuracy_score (3점 만점)
   - 질의에 대한 답변이 출처(tool)에 기반하는지

출력
- likert_result.csv (UTF-8-SIG)
"""

import os
import re
import json
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from tqdm import tqdm

# =========================
# 설정
# =========================
from dotenv import load_dotenv
load_dotenv()
API_KEY = os.environ.get("OPENAI_API_KEY")
MODEL = "gpt-5.4-mini"
OPENAI_TIMEOUT = 60

MAX_WORKERS = 4

# CSV 저장 옵션
SAVE_FULL_TEXT = False
QUESTION_PREVIEW_CHARS = 200
ANSWER_PREVIEW_CHARS = 200

# evidence 구성 옵션
MAX_EVIDENCE_PASSAGES = 8
PASSAGE_PER_ITEM_MAX_CHARS = 1000
FALLBACK_PASSAGES_WHEN_NO_VALID_CITATION = 5

# reason 길이 제한
REASON_MAX_CHARS = 300
OVERALL_COMMENT_MAX_CHARS = 400

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
    documents: Dict[int, Dict[str, Any]]  # {doc_id: {"meta": {...}, "passages": {pid: text}}}


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
        raise RuntimeError("API 키가 입력되지 않았거나 형식이 올바르지 않습니다.")
    return OpenAI(api_key=API_KEY)


def safe_int_in_range(value: Any, min_value: int, max_value: int, default: int = 0) -> int:
    try:
        iv = int(value)
        if min_value <= iv <= max_value:
            return iv
    except Exception:
        pass
    return default


# =========================
# content 추출
# =========================
def extract_text_from_content(content: Any) -> Optional[str]:
    if isinstance(content, str):
        t = normalize_ws(content)
        return t or None

    texts: List[str] = []
    items = content if isinstance(content, list) else [content]

    for item in items:
        if isinstance(item, dict):
            # 일반 text 구조
            if isinstance(item.get("text"), str):
                t = normalize_ws(item["text"])
                if t:
                    texts.append(t)

            # value 기반 구조
            if isinstance(item.get("value"), str):
                t = normalize_ws(item["value"])
                if t:
                    texts.append(t)
        elif isinstance(item, str):
            t = normalize_ws(item)
            if t:
                texts.append(t)

    out = " ".join(texts).strip()
    return out or None


def extract_assistant_text_only(content: Any) -> Optional[str]:
    """
    assistant는 최종 자연어 답변만 추출.
    type=function 등은 제외.
    """
    if isinstance(content, str):
        t = normalize_ws(content)
        return t or None

    texts: List[str] = []
    items = content if isinstance(content, list) else [content]

    for item in items:
        if isinstance(item, dict):
            if item.get("type") == "text" and isinstance(item.get("value"), str):
                t = normalize_ws(item["value"])
                if t:
                    texts.append(t)
        elif isinstance(item, str):
            t = normalize_ws(item)
            if t:
                texts.append(t)

    out = " ".join(texts).strip()
    return out or None


# =========================
# documents / passages 파싱
# =========================
DOCUMENT_RE = re.compile(r"<document\s+([^>]*)>(.*?)</document>", re.DOTALL | re.IGNORECASE)
PASSAGE_RE = re.compile(r"<passage\s+([^>]*)>(.*?)</passage>", re.DOTALL | re.IGNORECASE)


def _parse_attrs(attr_str: str) -> Dict[str, str]:
    out = {}
    pattern = re.compile(r'(\w+)\s*=\s*(?:"([^"]*)"|\'([^\']*)\'|([^\s>]+))')

    for match in pattern.finditer(attr_str):
        key = match.group(1)
        if match.group(2) is not None:
            val = match.group(2)
        elif match.group(3) is not None:
            val = match.group(3)
        else:
            val = match.group(4)

        if key not in out:
            out[key] = val

    return out


def parse_messages_for_qa_and_documents(messages: List[Dict[str, Any]]) -> ParsedFile:
    user_texts: List[str] = []
    assistant_texts: List[str] = []
    documents: Dict[int, Dict[str, Any]] = {}

    for msg in messages:
        role = msg.get("role")
        content = msg.get("content")
        val = extract_text_from_content(content)

        # content 전체 문자열에서 <document> 블록 탐색
        if isinstance(val, str) and val:
            for dm in DOCUMENT_RE.finditer(val):
                attrs = _parse_attrs(dm.group(1))
                if "id" not in attrs:
                    continue
                try:
                    did = int(str(attrs["id"]).strip())
                except Exception:
                    continue

                meta = {k: v for k, v in attrs.items() if k != "id"}
                body = dm.group(2)

                passages: Dict[int, str] = {}
                for pm in PASSAGE_RE.finditer(body):
                    pattrs = _parse_attrs(pm.group(1))
                    if "id" not in pattrs:
                        continue
                    try:
                        pid = int(str(pattrs["id"]).strip())
                    except Exception:
                        continue

                    ptxt = normalize_ws(pm.group(2))
                    if ptxt:
                        passages[pid] = ptxt

                if passages:
                    if did not in documents:
                        documents[did] = {"meta": dict(meta), "passages": {}}
                    else:
                        documents[did]["meta"].update(meta)
                    documents[did]["passages"].update(passages)

        if role == "user":
            t = extract_text_from_content(content)
            if t:
                user_texts.append(t)

        if role == "assistant":
            t = extract_assistant_text_only(content)
            if t:
                assistant_texts.append(t)

    question = user_texts[-1] if user_texts else None
    answer = assistant_texts[-1] if assistant_texts else None

    return ParsedFile(question=question, answer=answer, documents=documents)


def try_extract_from_simple_schema(obj: Any) -> ParsedFile:
    question = obj.get("question") if isinstance(obj, dict) else None
    answer = obj.get("answer") if isinstance(obj, dict) else None
    documents: Dict[int, Dict[str, Any]] = {}

    sources = (obj.get("sources") or obj.get("documents")) if isinstance(obj, dict) else None
    if isinstance(sources, list):
        for s in sources:
            try:
                did = int(s.get("id"))
            except Exception:
                continue

            meta = {k: v for k, v in s.items() if k not in {"id", "passages"}}
            passages: Dict[int, str] = {}

            for p in (s.get("passages") or []):
                try:
                    pid = int(p.get("id"))
                except Exception:
                    continue

                ptxt = ""
                if isinstance(p.get("text"), str):
                    ptxt = normalize_ws(p["text"])
                elif isinstance(p.get("value"), str):
                    ptxt = normalize_ws(p["value"])

                if ptxt:
                    passages[pid] = ptxt

            if passages:
                if did not in documents:
                    documents[did] = {"meta": dict(meta), "passages": {}}
                else:
                    documents[did]["meta"].update(meta)
                documents[did]["passages"].update(passages)

    return ParsedFile(question=question, answer=answer, documents=documents)


def parse_file_top(obj: Any) -> ParsedFile:
    if isinstance(obj, dict) and isinstance(obj.get("messages"), list):
        parsed = parse_messages_for_qa_and_documents(obj["messages"])
        if not parsed.question or not parsed.answer or not parsed.documents:
            fb = try_extract_from_simple_schema(obj)
            return ParsedFile(
                question=parsed.question or fb.question,
                answer=parsed.answer or fb.answer,
                documents=parsed.documents or fb.documents
            )
        return parsed

    return try_extract_from_simple_schema(obj)


# =========================
# citation 추출 / 감사
# =========================
BRACKET_GROUP_RE = re.compile(r"\[([^\[\]]+)\]")
PAIR_RE = re.compile(r"(?<!\d)(\d+)\s*:\s*(\d+)(?!\d)")


def extract_docpass_citations(text: str) -> Tuple[List[Tuple[int, int]], List[str]]:
    """
    답변 전체에서 [doc_id:passage_id] 패턴 추출
    반환:
    - pairs: [(doc_id, passage_id), ...]
    - malformed_groups: 콜론은 있으나 유효한 숫자:숫자 쌍을 못 찾은 bracket 원문
    """
    pairs: List[Tuple[int, int]] = []
    malformed_groups: List[str] = []

    for m in BRACKET_GROUP_RE.finditer(text or ""):
        inside = m.group(1).strip()
        found = PAIR_RE.findall(inside)

        if found:
            for d_str, p_str in found:
                pairs.append((int(d_str), int(p_str)))
        else:
            if ":" in inside:
                malformed_groups.append(m.group(0))

    return pairs, malformed_groups


def audit_citations(answer: Optional[str], documents: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
    pairs, malformed_groups = extract_docpass_citations(answer or "")

    invalids: List[Dict[str, Any]] = []
    valid_pairs: List[Tuple[int, int]] = []

    for d, p in pairs:
        doc = documents.get(d)
        if not doc:
            invalids.append({
                "doc_id": d,
                "passage_id": p,
                "reason": "문서 없음"
            })
            continue

        passages = doc.get("passages", {}) or {}
        if p in passages or str(p) in passages:
            valid_pairs.append((d, p))
        else:
            invalids.append({
                "doc_id": d,
                "passage_id": p,
                "reason": "패시지 없음"
            })

    unique_pairs = list(dict.fromkeys(pairs))
    unique_valid_pairs = list(dict.fromkeys(valid_pairs))

    doc_count = len(documents)
    passage_count = sum(len(doc.get("passages", {}) or {}) for doc in documents.values())

    return {
        "doc_count": doc_count,
        "passage_count": passage_count,
        "total_citation_pairs": len(pairs),
        "unique_citation_pairs": len(unique_pairs),
        "valid_citation_count": len(valid_pairs),
        "unique_valid_citation_pairs": len(unique_valid_pairs),
        "invalid_citation_count": len(invalids),
        "malformed_citation_group_count": len(malformed_groups),
        "malformed_citation_group_examples": malformed_groups[:5],
        "invalid_citation_examples": invalids[:10],
        "valid_pairs_for_evidence": unique_valid_pairs,
    }


# =========================
# evidence 구성
# =========================
def trim_passage(text: str, max_chars: int = PASSAGE_PER_ITEM_MAX_CHARS) -> str:
    text = normalize_ws(text)
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + " ...[truncated]"


def choose_evidence_passages(
    documents: Dict[int, Dict[str, Any]],
    valid_pairs: List[Tuple[int, int]],
    max_passages: int = MAX_EVIDENCE_PASSAGES
) -> List[Tuple[int, int, str]]:
    selected: List[Tuple[int, int, str]] = []

    # 1) 유효한 인용 우선
    for d, p in valid_pairs:
        doc = documents.get(d)
        if not doc:
            continue
        passages = doc.get("passages", {}) or {}
        if p in passages:
            selected.append((d, p, trim_passage(passages[p])))

    if selected:
        return selected[:max_passages]

    # 2) fallback: 앞쪽 문서/패시지 일부 사용
    count = 0
    for d in sorted(documents.keys()):
        doc = documents[d]
        passages = doc.get("passages", {}) or {}
        for p in sorted(passages.keys()):
            selected.append((d, p, trim_passage(passages[p])))
            count += 1
            if count >= min(FALLBACK_PASSAGES_WHEN_NO_VALID_CITATION, max_passages):
                return selected

    return selected[:max_passages]


def build_evidence_text(selected_passages: List[Tuple[int, int, str]]) -> str:
    return "\n".join([f"[{d}:{p}] {text}" for d, p, text in selected_passages])


# =========================
# LLM 출력 파싱
# =========================
def clean_json_block(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```json\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^```\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _parsed_likert_dict(data: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "citation_matching_score": safe_int_in_range(data.get("citation_matching_score"), 1, 5, 0),
        "citation_matching_reason": shorten_text(str(data.get("citation_matching_reason", "")).strip(), REASON_MAX_CHARS),
        "question_clarity_score": safe_int_in_range(data.get("question_clarity_score"), 1, 2, 0),
        "question_clarity_reason": shorten_text(str(data.get("question_clarity_reason", "")).strip(), REASON_MAX_CHARS),
        "question_answerability_score": safe_int_in_range(data.get("question_answerability_score"), 1, 3, 0),
        "question_answerability_reason": shorten_text(str(data.get("question_answerability_reason", "")).strip(), REASON_MAX_CHARS),
        "answer_completeness_score": safe_int_in_range(data.get("answer_completeness_score"), 1, 2, 0),
        "answer_completeness_reason": shorten_text(str(data.get("answer_completeness_reason", "")).strip(), REASON_MAX_CHARS),
        "source_based_answer_accuracy_score": safe_int_in_range(data.get("source_based_answer_accuracy_score"), 1, 3, 0),
        "source_based_answer_accuracy_reason": shorten_text(str(data.get("source_based_answer_accuracy_reason", "")).strip(), REASON_MAX_CHARS),
        "overall_comment": shorten_text(str(data.get("overall_comment", "")).strip(), OVERALL_COMMENT_MAX_CHARS),
    }


def parse_likert_output(raw_text: str) -> Dict[str, Any]:
    default = {
        "citation_matching_score": 0,
        "citation_matching_reason": "LLM 출력 파싱 실패",
        "question_clarity_score": 0,
        "question_clarity_reason": "LLM 출력 파싱 실패",
        "question_answerability_score": 0,
        "question_answerability_reason": "LLM 출력 파싱 실패",
        "answer_completeness_score": 0,
        "answer_completeness_reason": "LLM 출력 파싱 실패",
        "source_based_answer_accuracy_score": 0,
        "source_based_answer_accuracy_reason": "LLM 출력 파싱 실패",
        "overall_comment": "LLM 출력 파싱 실패",
    }

    text = clean_json_block(raw_text)

    # 1차: 전체 JSON 파싱
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return _parsed_likert_dict(data)
    except Exception:
        pass

    # 2차: JSON 블록 추출 재시도
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            data = json.loads(m.group(0))
            if isinstance(data, dict):
                return _parsed_likert_dict(data)
        except Exception:
            pass

    return default


# =========================
# LLM 평가
# =========================
def evaluate_likert_with_llm(
    client: OpenAI,
    question: Optional[str],
    answer: Optional[str],
    documents: Dict[int, Dict[str, Any]],
    citation_audit: Dict[str, Any],
) -> Dict[str, Any]:
    if not question and not answer:
        return {
            "citation_matching_score": 1,
            "citation_matching_reason": "질문과 답변이 모두 없어 인용 매칭을 정상적으로 평가할 수 없습니다.",
            "question_clarity_score": 1,
            "question_clarity_reason": "질문이 없어 의도와 문장 완결성을 파악할 수 없습니다.",
            "question_answerability_score": 1,
            "question_answerability_reason": "질문이 없어 올바른 답변 생성 가능성을 판단할 수 없습니다.",
            "answer_completeness_score": 1,
            "answer_completeness_reason": "답변이 없어 질문과의 관련성이나 완결성을 평가할 수 없습니다.",
            "source_based_answer_accuracy_score": 1,
            "source_based_answer_accuracy_reason": "답변이 없어 출처 기반 정확성을 평가할 수 없습니다.",
            "overall_comment": "질문과 답변이 모두 누락되어 전체 품질 평가가 어렵습니다.",
        }

    if not answer:
        return {
            "citation_matching_score": 1,
            "citation_matching_reason": "assistant 답변이 없어 인용 매칭을 평가하기 어렵습니다.",
            "question_clarity_score": 2 if question else 1,
            "question_clarity_reason": "질문은 존재하나 답변이 없어 후속 항목 평가는 제한됩니다." if question else "질문이 없어 의도 파악이 불가능합니다.",
            "question_answerability_score": 2 if question else 1,
            "question_answerability_reason": "질문 자체의 응답 가능성은 일부 판단 가능하나, 답변 부재로 전체 품질 확인은 제한됩니다." if question else "질문이 없어 응답 가능성을 판단할 수 없습니다.",
            "answer_completeness_score": 1,
            "answer_completeness_reason": "assistant 답변이 없어 질문의 핵심 사항을 반영했는지 평가할 수 없습니다.",
            "source_based_answer_accuracy_score": 1,
            "source_based_answer_accuracy_reason": "assistant 답변이 없어 출처 기반성을 평가할 수 없습니다.",
            "overall_comment": "답변이 없어 답변 관련 항목은 최저점으로 처리했습니다.",
        }

    valid_pairs = citation_audit.get("valid_pairs_for_evidence", [])
    selected_passages = choose_evidence_passages(documents, valid_pairs, MAX_EVIDENCE_PASSAGES)
    evidence_text = build_evidence_text(selected_passages)

    citation_summary = {
        "doc_count": citation_audit.get("doc_count", 0),
        "passage_count": citation_audit.get("passage_count", 0),
        "total_citation_pairs": citation_audit.get("total_citation_pairs", 0),
        "unique_citation_pairs": citation_audit.get("unique_citation_pairs", 0),
        "valid_citation_count": citation_audit.get("valid_citation_count", 0),
        "unique_valid_citation_pairs": citation_audit.get("unique_valid_citation_pairs", 0),
        "invalid_citation_count": citation_audit.get("invalid_citation_count", 0),
        "malformed_citation_group_count": citation_audit.get("malformed_citation_group_count", 0),
        "malformed_citation_group_examples": citation_audit.get("malformed_citation_group_examples", []),
        "invalid_citation_examples": citation_audit.get("invalid_citation_examples", []),
    }

    system_prompt = """
너는 출처 포함 QA 데이터의 품질을 평가하는 검사자다.
반드시 아래 5개 항목을 지정된 점수 범위 안에서 평가하라.

[항목 1: citation_matching_score / 5점 만점]
답변의 [doc_id:passage_id] 인용 표기가 실제 문서/패시지와 얼마나 정확하게 대응하는가.
- 5점: 모든 인용이 정확하게 매칭되며 오류가 없다.
- 4점: 대부분 정확하고 일부 경미한 문제만 있다.
- 3점: 맞는 인용과 애매하거나 오류인 인용이 혼재한다.
- 2점: 여러 인용이 부정확하거나 중요한 문제가 있다.
- 1점: 대부분 틀렸거나 인용 체계가 거의 성립하지 않는다.

[항목 2: question_clarity_score / 2점 만점]
질문의 의도가 명확하게 전달되고, 문장 자체에 결함이 없는지 평가한다.
- 2점: 일부 오탈자, 오류가 있더라도 질문의 의도가 분명히 파악 가능하다.
- 1점: 문장이 잘리거나 심각한 비문으로 의도 파악이 불가능하다.

[항목 3: question_answerability_score / 3점 만점]
올바른 답변을 만들어낼 수 있는 질문인지 평가한다.
- 3점: 질문만으로 올바른 답변을 만들어낼 수 있다. 필요한 조건·맥락이 충분하고, 정답이나 핵심 표현이 질문 안에 노출되지 않는다.
- 2점: 답변 생성은 가능하나 일부 조건이 빠져 있거나 모호해서 합리적 추론이 필요하다. 또는 정답 힌트가 약하게 포함되어 있다.
- 1점: 핵심 조건이 누락되어 올바른 답변을 만들기 어렵거나, 원문·도구·이미지 범위를 벗어난 질문이거나, 질문 안에 정답이 직접 노출되어 있다.

[항목 4: answer_completeness_score / 2점 만점]
답변이 질문과 관련이 있고, 질문에 대한 핵심 사항/요청을 모두 반영해 완결된 형태인지 평가한다.
- 2점: 답변이 질문에 대한 핵심 사항/요청을 모두 반영하고 완결된 형태다.
- 1점: 답변이 질문과 관련이 없거나 핵심 요청을 충족하지 못한다.

[항목 5: source_based_answer_accuracy_score / 3점 만점]
질의에 대한 답변이 출처(tool passages)에 기반하는지 평가한다.
- 3점: 대부분의 답변이 출처에 근거하여 작성되었다.
- 2점: 많은 답변이 출처에 근거하여 작성되었으나 일부 근거가 약하거나 확인이 필요하다.
- 1점: 출처에 전혀 근거하지 않거나 참조하지 않은 답변이다.

중요:
- 항목 1은 주로 citation_audit_summary를 근거로 판단하라.
- 항목 2와 3은 user_question 자체를 중심으로 판단하라.
- 항목 4는 user_question과 assistant_answer의 대응성을 판단하라.
- 항목 5는 assistant_answer와 tool_passages_for_grounding의 내용 일치 여부를 판단하라.
- reason은 각 항목별로 한국어 1~3문장, 최대 300자 이내로 작성하라.
- overall_comment는 전체 총평을 한국어 2~4문장으로 작성하라.
- 반드시 JSON 객체만 출력하라. 마크다운, 코드블록, 설명 문장은 금지한다.
출력 형식은 정확히 아래 키를 사용하라:
{
  "citation_matching_score": 1,
  "citation_matching_reason": "...",
  "question_clarity_score": 1,
  "question_clarity_reason": "...",
  "question_answerability_score": 1,
  "question_answerability_reason": "...",
  "answer_completeness_score": 1,
  "answer_completeness_reason": "...",
  "source_based_answer_accuracy_score": 1,
  "source_based_answer_accuracy_reason": "...",
  "overall_comment": "..."
}
""".strip()

    user_prompt = (
        "[user_question]\n"
        f"{question or ''}\n\n"
        "[assistant_answer]\n"
        f"{answer or ''}\n\n"
        "[citation_audit_summary]\n"
        f"{json.dumps(citation_summary, ensure_ascii=False, indent=2)}\n\n"
        "[tool_passages_for_grounding]\n"
        f"{evidence_text}\n"
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
    return parse_likert_output(out)


# =========================
# 파일 단위 처리
# =========================
def process_one_file(path: str) -> Dict[str, Any]:
    base_row: Dict[str, Any] = {
        "file": os.path.basename(path),
        "doc_count": 0,
        "passage_count": 0,
        "total_citation_pairs": 0,
        "invalid_citation_count": 0,
        "malformed_citation_group_count": 0,

        "citation_matching_score": 0,
        "citation_matching_reason": "",
        "question_clarity_score": 0,
        "question_clarity_reason": "",
        "question_answerability_score": 0,
        "question_answerability_reason": "",
        "answer_completeness_score": 0,
        "answer_completeness_reason": "",
        "source_based_answer_accuracy_score": 0,
        "source_based_answer_accuracy_reason": "",
        "overall_comment": "",
        "status": "OK",
        "error_message": "",
    }

    if SAVE_FULL_TEXT:
        base_row["question"] = ""
        base_row["answer"] = ""
    else:
        base_row["question_preview"] = ""
        base_row["answer_preview"] = ""

    # 1) 파일 파싱
    try:
        obj = read_json(path)
        parsed = parse_file_top(obj)
    except Exception as e:
        base_row["status"] = "PARSE_ERROR"
        base_row["error_message"] = f"파일 파싱 실패: {str(e)}"
        return base_row

    if SAVE_FULL_TEXT:
        base_row["question"] = parsed.question or ""
        base_row["answer"] = parsed.answer or ""
    else:
        base_row["question_preview"] = shorten_text(parsed.question or "", QUESTION_PREVIEW_CHARS)
        base_row["answer_preview"] = shorten_text(parsed.answer or "", ANSWER_PREVIEW_CHARS)

    # 2) 인용 감사(기계식)
    try:
        citation_audit = audit_citations(parsed.answer, parsed.documents)
        base_row["doc_count"] = citation_audit["doc_count"]
        base_row["passage_count"] = citation_audit["passage_count"]
        base_row["total_citation_pairs"] = citation_audit["total_citation_pairs"]
        base_row["invalid_citation_count"] = citation_audit["invalid_citation_count"]
        base_row["malformed_citation_group_count"] = citation_audit["malformed_citation_group_count"]
    except Exception as e:
        base_row["status"] = "CITATION_AUDIT_ERROR"
        base_row["error_message"] = f"citation 감사 실패: {str(e)}"
        return base_row

    # 3) LLM 평가
    try:
        client = openai_client()
        result = evaluate_likert_with_llm(
            client=client,
            question=parsed.question,
            answer=parsed.answer,
            documents=parsed.documents,
            citation_audit=citation_audit,
        )
        base_row["citation_matching_score"] = result["citation_matching_score"]
        base_row["citation_matching_reason"] = result["citation_matching_reason"]
        base_row["question_clarity_score"] = result["question_clarity_score"]
        base_row["question_clarity_reason"] = result["question_clarity_reason"]
        base_row["question_answerability_score"] = result["question_answerability_score"]
        base_row["question_answerability_reason"] = result["question_answerability_reason"]
        base_row["answer_completeness_score"] = result["answer_completeness_score"]
        base_row["answer_completeness_reason"] = result["answer_completeness_reason"]
        base_row["source_based_answer_accuracy_score"] = result["source_based_answer_accuracy_score"]
        base_row["source_based_answer_accuracy_reason"] = result["source_based_answer_accuracy_reason"]
        base_row["overall_comment"] = result["overall_comment"]
    except Exception as e:
        base_row["status"] = "LLM_ERROR"
        base_row["error_message"] = f"LLM 평가 실패: {str(e)}"

    return base_row


# =========================
# 메인
# =========================
def main():
    print("=== 출처 포함 QA 데이터 품질 검사기 ===")
    print("평가 항목:")
    print("1) citation_matching_score: 인용 매칭 정확성 / 5점")
    print("2) question_clarity_score: 질문 명확성 / 2점")
    print("3) question_answerability_score: 질문 응답 가능성 / 3점")
    print("4) answer_completeness_score: 답변 완결성 / 2점")
    print("5) source_based_answer_accuracy_score: 출처 기반 답변 정확성 / 3점")
    print()

    in_dir = ask_path("검사할 JSON 폴더 경로를 입력하세요: ")
    out_dir = ask_path("검사 결과를 저장할 폴더 경로를 입력하세요: ")

    if not os.path.isdir(in_dir):
        print(f"❌ 폴더를 찾을 수 없습니다: {in_dir}")
        sys.exit(1)

    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, "likert_result.csv")

    json_files = [
        os.path.join(in_dir, f)
        for f in os.listdir(in_dir)
        if f.lower().endswith(".json")
    ]

    result_columns = [
        "file", "doc_count", "passage_count", "total_citation_pairs",
        "invalid_citation_count", "malformed_citation_group_count",
        "citation_matching_score", "citation_matching_reason",
        "question_clarity_score", "question_clarity_reason",
        "question_answerability_score", "question_answerability_reason",
        "answer_completeness_score", "answer_completeness_reason",
        "source_based_answer_accuracy_score", "source_based_answer_accuracy_reason",
        "overall_comment", "status", "error_message",
    ]

    if SAVE_FULL_TEXT:
        result_columns.extend(["question", "answer"])
    else:
        result_columns.extend(["question_preview", "answer_preview"])

    if not json_files:
        print("⚠️ JSON 파일이 없습니다.")
        pd.DataFrame(columns=result_columns).to_csv(out_csv, index=False, encoding="utf-8-sig")
        sys.exit(0)

    rows: List[Dict[str, Any]] = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_one_file, path): path for path in json_files}
        for future in tqdm(as_completed(futures), total=len(futures), desc="품질 평가 중"):
            rows.append(future.result())

    rows.sort(key=lambda x: x.get("file", ""))

    df = pd.DataFrame(rows)
    df = df.reindex(columns=result_columns)
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    print()
    print(f"✅ 완료: {out_csv}")
    print(f"총 파일 수: {len(json_files)}")
    print(f"병렬 worker 수: {MAX_WORKERS}")
    print(f"원문 전체 저장 여부: {SAVE_FULL_TEXT}")


if __name__ == "__main__":
    main()
