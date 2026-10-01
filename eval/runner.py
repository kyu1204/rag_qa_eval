"""골드셋 문항을 지금 설정의 RAG로 실행하고, 채점에 필요한 추적을 남긴다."""

from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager

from app import ingest, rag
from app.config import Settings, settings
from app.db import connect
from eval.goldset import GoldSet, group


@contextmanager
def overridden(overrides: dict[str, str]) -> Iterator[None]:
    """설정을 덮어쓴다 (.env 위에 얹고 타입 검증). 끝나면 원래 값으로 되돌린다.
    모든 모듈이 같은 settings 객체를 보므로 실험 설정이 검색·생성·채점 전체에 적용된다."""
    unknown = [k for k in overrides if k.lower() not in Settings.model_fields]
    if unknown:
        raise ValueError(f"알 수 없는 설정: {unknown}")
    old = settings.model_dump()
    new = Settings(**{k.lower(): v for k, v in overrides.items()})
    for name in Settings.model_fields:
        setattr(settings, name, getattr(new, name))
    try:
        yield
    finally:
        for name, value in old.items():
            setattr(settings, name, value)


def ensure_index() -> int:
    """지금 설정의 index_version 청크가 없으면 코퍼스를 적재한다. 청크 수를 돌려준다."""
    count = lambda: connect().execute(  # noqa: E731
        "SELECT count(*) FROM chunks WHERE index_version = %s", (settings.index_version(),)
    ).fetchone()[0]
    if (n := count()) == 0:
        if ingest.main([settings.corpus_dir]) != 0:
            raise RuntimeError(f"{settings.corpus_dir} 적재 실패")
        n = count()
    return n


def run_one(item: dict, repeat: int) -> dict:
    record = {"id": item["id"], "group": group(item), "type": item["type"], "repeat": repeat,
              "question": item["question"], "error": None}
    try:
        result = rag.answer(item["question"])
    except Exception as e:  # 한 문항의 API 오류가 전체 실행을 멈추지 않게 기록만 한다
        return record | {"status": "error", "error": f"{type(e).__name__}: {e}"[:300]}
    return record | {
        "status": result.status,
        "answer": result.answer,
        "citations": result.citations,
        "warnings": result.warnings,
        "retrieved": [{"ref": h.ref, "document_id": h.document_id, "page_start": h.page_start,
                       "page_end": h.page_end, "score": round(h.score, 4), "content": h.content}
                      for h in result.retrieved],
        "latency_ms": result.meta.get("latency_ms"),
        "usage": result.meta.get("usage"),
    }


def run_items(gold: GoldSet, repeats: int = 1, workers: int = 6,
              on_done: Callable[[dict], None] | None = None) -> list[dict]:
    jobs = [(item, r) for r in range(repeats) for item in gold.items]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        records = []
        for record in pool.map(lambda job: run_one(*job), jobs):
            records.append(record)
            if on_done:
                on_done(record)
    return records
