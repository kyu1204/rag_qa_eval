"""질의 API. 서버 실행: uv run uvicorn app.api:app --reload

- POST /query: 일반 JSON 응답, 또는 stream=true면 SSE(sources -> delta* -> done)
- GET /: 간단한 웹 UI (app/static/index.html)
- 스키마 문서: /docs (OpenAPI)
"""

import json
from pathlib import Path
from typing import Literal

from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from app import rag
from app.config import settings
from app.db import connect

app = FastAPI(title="정책 변화 안내 RAG QA", version="0.1.0")
STATIC = Path(__file__).parent / "static"


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000, examples=["노란우산공제 납입한도가 얼마로 늘어나나요?"])
    top_k: int = Field(default=settings.top_k, ge=1, le=50)
    stream: bool = False
    debug: bool = Field(default=False, description="true면 검색된 top-k 전체(retrieved)를 함께 돌려준다")


class Citation(BaseModel):
    ref: int = Field(description="답변 속 [n] 번호")
    document_id: str
    title: str
    page_start: int | None = Field(description="PDF 쪽 인덱스")
    page_end: int | None
    printed_page: int | None = Field(description="책자 인쇄 쪽번호 (분석기 경로에서만)")
    heading: str | None
    snippet: str
    score: float = Field(description="검색 코사인 유사도")


class QueryResponse(BaseModel):
    status: Literal["answered", "insufficient_context"]
    answer: str
    citations: list[Citation]
    warnings: list[str]
    meta: dict = Field(description="index_version, embed_model, llm_model, prompt_version, top_score, usage, latency_ms")
    retrieved: list[dict] | None = None


def _sse(event: str, data) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@app.post("/query", response_model=QueryResponse, response_model_exclude_none=True)
def query(req: QueryRequest):
    if not req.stream:
        return rag.answer(req.question, req.top_k).to_dict(debug=req.debug)

    def events():
        for kind, payload in rag.run(req.question, req.top_k):
            if kind == "sources":
                yield _sse("sources", [{"ref": h.ref, "label": h.label(), "score": round(h.score, 4)} for h in payload])
            elif kind == "delta":
                yield _sse("delta", payload)
            else:
                yield _sse("done", payload.to_dict(debug=req.debug))

    return StreamingResponse(events(), media_type="text/event-stream")


@app.get("/health")
def health():
    with connect() as conn:
        chunks = conn.execute("SELECT count(*) FROM chunks WHERE index_version = %s", (settings.index_version(),)).fetchone()[0]
    return {"ok": True, "index_version": settings.index_version(), "chunks": chunks}


@app.get("/", include_in_schema=False)
def ui():
    return FileResponse(STATIC / "index.html")
