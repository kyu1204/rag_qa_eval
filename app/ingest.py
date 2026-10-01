"""적재 CLI: 파일 -> 추출 -> 청킹 -> 임베딩 -> pgvector.

    uv run python -m app.ingest data/corpus [--dry-run] [--force]

문서 1개 = 트랜잭션 1개. 같은 파일(이름)·같은 해시·같은 index_version 청크가 있으면 건너뛰고,
해시가 바뀌면 그 문서의 청크를 전부 지우고 다시 넣는다. 임베딩(네트워크)은 트랜잭션 밖에서 끝낸다.
"""

import argparse
import hashlib
import statistics
import sys
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import pymupdf
from pgvector import Vector
from psycopg.types.json import Jsonb
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn
from rich.table import Table

from app.chunking import Chunk, chunk
from app.config import settings
from app.db import connect, ensure_schema
from app.embed import embed
from app.extract import extract, sniff

console = Console(stderr=True)


@dataclass
class Result:
    file: str
    status: str  # ingested | replaced | skipped | unsupported | failed | dry-run
    pages: int = 0
    chunks: int = 0
    tokens: int = 0
    seconds: float = 0.0
    note: str = ""


def iter_files(paths: list[str]) -> list[Path]:
    files = []
    for p in map(Path, paths):
        files += sorted(f for f in p.rglob("*") if f.is_file() and not f.name.startswith(".")) if p.is_dir() else [p]
    return files


def page_count(data: bytes, kind: str) -> int | None:
    return pymupdf.open(stream=data, filetype="pdf").page_count if kind == "pdf" else None


def mime_type(kind: str, name: str) -> str:
    return "application/pdf" if kind == "pdf" else ("text/markdown" if name.endswith(".md") else "text/plain")


def ingest_file(conn, path: Path, version: str, force: bool, dry_run: bool, progress: Progress) -> Result:
    started = time.perf_counter()
    name = unicodedata.normalize("NFC", path.name)  # macOS 파일명은 NFD일 수 있다
    data = path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    kind = sniff(data, name)
    if kind is None:
        return Result(name, "unsupported", note="PDF/MD/TXT 아님")

    row = None if dry_run else conn.execute("SELECT content_hash FROM documents WHERE id = %s", (name,)).fetchone()
    same_hash = row is not None and row[0] == digest
    if same_hash and not force:
        exists = conn.execute(
            "SELECT count(*) FROM chunks WHERE document_id = %s AND index_version = %s", (name, version)
        ).fetchone()[0]
        if exists:
            return Result(name, "skipped", chunks=exists, note="해시·index_version 동일")

    pages = extract(data, name, settings.extractor)
    chunks = chunk(pages, Path(name).stem)
    tokens = sum(c.meta["token_count"] for c in chunks)
    if dry_run:
        dist = sorted(c.meta["token_count"] for c in chunks)
        note = f"p50 {statistics.median(dist):.0f} / max {dist[-1]} 토큰" if dist else "청크 없음"
        return Result(name, "dry-run", len(pages), len(chunks), tokens, time.perf_counter() - started, note)

    task = progress.add_task(f"임베딩 {name[:30]}", total=len(chunks))
    vectors = embed([c.embed_text for c in chunks], on_batch=lambda n: progress.advance(task, n))
    progress.remove_task(task)

    with conn.transaction():
        conn.execute(
            """INSERT INTO documents (id, title, source, mime_type, content_hash, page_count)
               VALUES (%s, %s, %s, %s, %s, %s)
               ON CONFLICT (id) DO UPDATE SET title = EXCLUDED.title, source = EXCLUDED.source,
                 mime_type = EXCLUDED.mime_type, content_hash = EXCLUDED.content_hash,
                 page_count = EXCLUDED.page_count, created_at = now()""",
            (name, Path(name).stem, str(path), mime_type(kind, name), digest, page_count(data, kind)),
        )
        if same_hash:  # 같은 내용 재적재(--force): 현재 버전만 교체
            conn.execute("DELETE FROM chunks WHERE document_id = %s AND index_version = %s", (name, version))
        else:  # 새 문서이거나 내용이 바뀜: 모든 버전의 청크가 낡았다
            conn.execute("DELETE FROM chunks WHERE document_id = %s", (name,))
        _insert_chunks(conn, name, version, chunks, vectors)

    status = "replaced" if row is not None and not same_hash else "ingested"
    return Result(name, status, len(pages), len(chunks), tokens, time.perf_counter() - started)


def _insert_chunks(conn, document_id: str, version: str, chunks: list[Chunk], vectors: list[list[float]]) -> None:
    with conn.cursor() as cur:
        cur.executemany(
            """INSERT INTO chunks (document_id, index_version, ord, content, page_start, page_end, meta, embedding)
               VALUES (%s, %s, %s, %s, %s, %s, %s, %s)""",
            [
                (document_id, version, c.ord, c.content, c.page_start, c.page_end, Jsonb(c.meta), Vector(v))
                for c, v in zip(chunks, vectors, strict=True)
            ],
        )


def print_summary(results: list[Result], version: str) -> None:
    table = Table(title=f"적재 결과  extractor={settings.extractor}  chunker={settings.chunk_strategy}  index_version={version}")
    for col, justify in [("파일", "left"), ("상태", "left"), ("쪽", "right"), ("청크", "right"), ("토큰", "right"), ("초", "right"), ("비고", "left")]:
        table.add_column(col, justify=justify)
    colors = {"ingested": "green", "replaced": "yellow", "skipped": "dim", "dry-run": "cyan"}
    for r in results:
        color = colors.get(r.status, "red")
        table.add_row(r.file, f"[{color}]{r.status}[/]", str(r.pages or ""), str(r.chunks or ""),
                      f"{r.tokens:,}" if r.tokens else "", f"{r.seconds:.1f}" if r.seconds else "", r.note)
    console.print(table)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="문서를 추출·청킹·임베딩해 pgvector에 적재한다")
    parser.add_argument("paths", nargs="+", help="파일 또는 디렉터리")
    parser.add_argument("--dry-run", action="store_true", help="추출·청킹 통계만 출력 (DB·API 호출 없음)")
    parser.add_argument("--force", action="store_true", help="해시가 같아도 다시 적재")
    args = parser.parse_args(argv)

    version = settings.index_version()
    files = iter_files(args.paths)
    conn = None if args.dry_run else connect()
    if conn:
        ensure_schema(conn)

    results = []
    columns = [TextColumn("{task.description}"), BarColumn(), MofNCompleteColumn(), TimeElapsedColumn()]
    with Progress(*columns, console=console, transient=True) as progress:
        overall = progress.add_task("파일", total=len(files))
        for path in files:
            progress.update(overall, description=f"파일 {path.name[:30]}")
            try:
                results.append(ingest_file(conn, path, version, args.force, args.dry_run, progress))
            except Exception as e:  # 한 파일 실패가 나머지를 막지 않게, 결과표에 남긴다
                results.append(Result(unicodedata.normalize("NFC", path.name), "failed", note=f"{type(e).__name__}: {e}"[:120]))
            progress.advance(overall)

    print_summary(results, version)
    if conn:
        conn.close()
    return 1 if any(r.status == "failed" for r in results) else 0


if __name__ == "__main__":
    sys.exit(main())
