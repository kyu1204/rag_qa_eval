"""질의 -> 검색 -> 응답 불가 게이트 -> 생성 -> 인용 검증.

API, UI, Eval Harness가 모두 run()/answer()를 쓴다. 결과에는 검색 결과·점수·모델·토큰·지연이
그대로 담겨 평가에서 "검색이 틀렸는지, 생성이 틀렸는지"를 나눠 볼 수 있다.
"""

import re
import time
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field

from openai import OpenAI
from pgvector import Vector

from app.config import settings
from app.db import connect
from app.embed import embed

PROMPT_VERSION = "v1"
NO_ANSWER = "[[NO_ANSWER]]"
SYSTEM_PROMPT = f"""당신은 정부 정책 안내 문서를 근거로 답하는 상담원이다.
규칙:
1. 아래 [출처]에 적힌 내용만으로 답한다. 출처에 없는 사실, 수치, 날짜를 추측하거나 보태지 않는다.
2. 사실을 말한 문장 끝마다 근거 출처 번호를 [1], [2]처럼 붙인다.
3. 출처로 답할 수 없으면 다른 말 없이 정확히 {NO_ANSWER} 만 출력한다.
4. 한국어로 간결하게 답한다."""


@dataclass
class Hit:
    ref: int  # 프롬프트 속 출처 번호 (1부터)
    chunk_id: int
    document_id: str
    title: str
    content: str
    page_start: int | None
    page_end: int | None
    score: float  # 코사인 유사도
    meta: dict = field(default_factory=dict)

    def label(self) -> str:
        page = self.meta.get("printed_page") or self.page_start
        heading = self.meta.get("heading", "")
        return f"{self.title} p.{page}" + (f" ({heading})" if heading else "")


@dataclass
class Result:
    status: str  # answered | insufficient_context
    answer: str
    citations: list[dict]
    retrieved: list[Hit]
    warnings: list[str] = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    def to_dict(self, debug: bool = False) -> dict:
        d = asdict(self)
        if not debug:
            d.pop("retrieved")
        return d


def retrieve(question: str, top_k: int) -> list[Hit]:
    vector = Vector(embed([question])[0])
    with connect() as conn:
        # 여러 index_version이 공존하면 HNSW가 필터 전에 후보를 잘라 top_k보다 적게 줄 수 있다
        conn.execute("SET hnsw.iterative_scan = relaxed_order")
        rows = conn.execute(
            """SELECT c.id, c.document_id, d.title, c.content, c.page_start, c.page_end,
                      1 - (c.embedding <=> %s) AS score, c.meta
               FROM chunks c JOIN documents d ON d.id = c.document_id
               WHERE c.index_version = %s
               ORDER BY c.embedding <=> %s
               LIMIT %s""",
            (vector, settings.index_version(), vector, top_k),
        ).fetchall()
    return [Hit(i + 1, *row) for i, row in enumerate(rows)]


def build_messages(question: str, hits: list[Hit]) -> list[dict]:
    sources = "\n\n".join(f"[{h.ref}] {h.label()}\n{h.content}" for h in hits)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"[출처]\n{sources}\n\n[질문]\n{question}"},
    ]


def parse_citations(answer: str, hits: list[Hit]) -> tuple[str, list[dict], list[str]]:
    """[n] 인용을 검증한다. 출처 범위 밖 번호는 본문에서 지우고 경고로 남긴다."""
    valid = {h.ref: h for h in hits}
    refs = [int(n) for n in re.findall(r"\[(\d+)\]", answer)]
    invalid = sorted({n for n in refs if n not in valid})
    for n in invalid:
        answer = answer.replace(f"[{n}]", "")
    warnings = [f"존재하지 않는 출처 번호 제거: {invalid}"] if invalid else []
    cited = [valid[n] for n in dict.fromkeys(refs) if n in valid]
    if not cited:
        warnings.append("인용 없는 답변")
    citations = [
        {
            "ref": h.ref,
            "document_id": h.document_id,
            "title": h.title,
            "page_start": h.page_start,
            "page_end": h.page_end,
            "printed_page": h.meta.get("printed_page"),
            "heading": h.meta.get("heading"),
            "snippet": h.content[:200],
            "score": round(h.score, 4),
        }
        for h in cited
    ]
    return answer.strip(), citations, warnings


def llm() -> OpenAI:
    return OpenAI(base_url=settings.llm_base_url, api_key=settings.elice_api_key, timeout=90, max_retries=3)


def run(question: str, top_k: int | None = None, api: OpenAI | None = None) -> Iterator[tuple[str, object]]:
    """스트리밍 단위 이벤트를 낸다: ("sources", [Hit]) -> ("delta", str)* -> ("done", Result)."""
    started = time.perf_counter()
    hits = retrieve(question, top_k or settings.top_k)
    meta = {
        "index_version": settings.index_version(),
        "embed_model": settings.embed_model,
        "llm_model": settings.llm_model,
        "prompt_version": PROMPT_VERSION,
        "top_score": round(hits[0].score, 4) if hits else None,
    }
    yield "sources", hits

    if not hits or hits[0].score < settings.min_score:  # 게이트1: LLM을 부르지 않고 거절
        meta["latency_ms"] = round((time.perf_counter() - started) * 1000)
        yield "done", Result("insufficient_context", "", [], hits, ["검색 점수 미달"], meta)
        return

    stream = (api or llm()).chat.completions.create(
        model=settings.llm_model,
        messages=build_messages(question, hits),
        reasoning_effort=settings.llm_reasoning_effort,
        temperature=settings.llm_temperature,
        seed=settings.llm_seed,
        stream=True,
        stream_options={"include_usage": True},
    )
    text, usage, held = "", None, True
    for event in stream:
        if event.usage:
            usage = {"prompt_tokens": event.usage.prompt_tokens, "completion_tokens": event.usage.completion_tokens}
        if not (event.choices and event.choices[0].delta.content):
            continue
        text += event.choices[0].delta.content
        # 게이트2(센티널): 앞부분이 [[NO_ANSWER]]일 수 있는 동안은 내보내지 않는다
        if held and NO_ANSWER.startswith(text.strip()):
            continue
        if held:
            held = False
            yield "delta", text
        else:
            yield "delta", event.choices[0].delta.content

    meta |= {"usage": usage, "latency_ms": round((time.perf_counter() - started) * 1000)}
    if NO_ANSWER in text:
        yield "done", Result("insufficient_context", "", [], hits, ["모델이 출처로 답할 수 없다고 판단"], meta)
        return
    answer, citations, warnings = parse_citations(text, hits)
    yield "done", Result("answered", answer, citations, hits, warnings, meta)


def answer(question: str, top_k: int | None = None) -> Result:
    for kind, payload in run(question, top_k):
        if kind == "done":
            return payload
    raise RuntimeError("run()이 결과 없이 끝났다")


if __name__ == "__main__":
    import json
    import sys

    result = answer(" ".join(sys.argv[1:]))
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
