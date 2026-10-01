import psycopg
from pgvector.psycopg import register_vector

from app.config import settings


def connect() -> psycopg.Connection:
    """vector 확장을 보장한 연결을 돌려준다.

    register_vector는 vector 타입 OID를 조회하므로 CREATE EXTENSION 뒤에 불러야 한다
    (새 DB 첫 연결에서 순서가 바뀌면 실패한다).
    """
    conn = psycopg.connect(settings.database_url, autocommit=True)
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    register_vector(conn)
    return conn


if __name__ == "__main__":
    with connect() as conn:
        version = conn.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector'").fetchone()[0]
        print(f"ok: pgvector {version}")
