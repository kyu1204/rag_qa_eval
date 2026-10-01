"""app.ingest 멱등성·교체 테스트. docker compose의 DB가 필요하고, 임베딩 API는 가짜로 바꾼다."""

import pytest
from rich.progress import Progress

from app import ingest
from app.config import settings
from app.db import connect, ensure_schema

DOC_ID = "pytest-ingest.md"


@pytest.fixture
def conn(monkeypatch):
    try:
        c = connect()
    except Exception:
        pytest.skip("DB 없음: docker compose up -d --wait")
    ensure_schema(c)
    monkeypatch.setattr(ingest, "embed", lambda texts, on_batch=None: [[0.1] * settings.embed_dim for _ in texts])
    c.execute("DELETE FROM documents WHERE id = %s", (DOC_ID,))
    yield c
    c.execute("DELETE FROM documents WHERE id = %s", (DOC_ID,))
    c.close()


def _run(conn, path, version="test-v1", force=False):
    with Progress(disable=True) as progress:
        return ingest.ingest_file(conn, path, version, force, False, progress)


def _count(conn, version=None):
    sql, args = "SELECT count(*) FROM chunks WHERE document_id = %s", [DOC_ID]
    if version:
        sql, args = sql + " AND index_version = %s", args + [version]
    return conn.execute(sql, args).fetchone()[0]


def test_ingest_is_idempotent_and_replaces_changed_content(conn, tmp_path):
    path = tmp_path / DOC_ID
    path.write_text("정책 안내 문장입니다. " * 400)

    first = _run(conn, path)
    assert first.status == "ingested" and first.chunks > 1 and _count(conn) == first.chunks
    assert _run(conn, path).status == "skipped"  # 같은 해시·같은 버전

    other = _run(conn, path, version="test-v2")  # 설정이 바뀐 버전은 공존
    assert other.status == "ingested" and _count(conn) == first.chunks * 2

    path.write_text("바뀐 내용입니다. " * 10)  # 내용이 바뀌면 모든 버전 청크를 갈아치운다
    changed = _run(conn, path)
    assert changed.status == "replaced" and _count(conn) == changed.chunks == _count(conn, "test-v1")


def test_unsupported_file_is_reported_not_ingested(conn, tmp_path):
    path = tmp_path / "image.png"
    path.write_bytes(b"\x89PNG\r\n")
    assert _run(conn, path).status == "unsupported"
