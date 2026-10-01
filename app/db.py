import psycopg
from pgvector.psycopg import register_vector

from app.config import settings

# 문서 1행 = 원본 파일 1개, 청크 N행이 document_id로 연결된다.
# 스키마 변경은 마이그레이션 대신 `docker compose down -v` 후 재적재한다 (재적재 수 분).
SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
  id           TEXT PRIMARY KEY,                  -- NFC 파일명 (골드셋이 참조하는 안정 ID)
  title        TEXT NOT NULL,
  source       TEXT NOT NULL,                     -- 리포 내 상대 경로
  mime_type    TEXT NOT NULL,
  content_hash TEXT NOT NULL,                     -- sha256(원본 바이트), 변경 감지
  page_count   INT,
  created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS chunks (
  id            BIGSERIAL PRIMARY KEY,
  document_id   TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
  index_version TEXT NOT NULL,                    -- 청킹·임베딩 설정 지문 (검색 필터)
  ord           INT  NOT NULL,                    -- 문서 내 순번
  content       TEXT NOT NULL,                    -- 인용·LLM 입력용 원문
  page_start    INT,                              -- 인쇄 쪽번호, 쪽 개념 없는 문서는 NULL
  page_end      INT,
  meta          JSONB NOT NULL DEFAULT '{{}}',     -- heading, chapter, ministry, title, effective, tags, token_count
  embedding     vector({dim}) NOT NULL,
  UNIQUE (document_id, index_version, ord)
);

CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw ON chunks USING hnsw (embedding vector_cosine_ops);
"""


def connect() -> psycopg.Connection:
    """vector 확장을 보장한 연결을 돌려준다.

    register_vector는 vector 타입 OID를 조회하므로 CREATE EXTENSION 뒤에 불러야 한다
    (새 DB 첫 연결에서 순서가 바뀌면 실패한다).
    """
    conn = psycopg.connect(settings.database_url, autocommit=True)
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    register_vector(conn)
    return conn


def ensure_schema(conn: psycopg.Connection) -> None:
    conn.execute(SCHEMA.format(dim=settings.embed_dim))


if __name__ == "__main__":
    with connect() as conn:
        ensure_schema(conn)
        version = conn.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector'").fetchone()[0]
        print(f"ok: pgvector {version}, schema applied (embed_dim={settings.embed_dim})")
