import hashlib
import json
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """모든 조절값의 단일 소스. 값은 환경변수 또는 .env에서 읽는다."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql://rag:rag@localhost:5432/rag"

    elice_api_key: str = ""
    embed_base_url: str = ""
    embed_model: str = ""
    embed_dim: int = 1536

    chunk_strategy: Literal["policy", "split"] = "policy"
    chunk_tokens: int = 500  # split 전략에서만 사용
    chunk_overlap: int = 75  # split 전략에서만 사용

    def index_version(self) -> str:
        """청킹·임베딩 설정의 지문. 설정이 바뀌면 다른 버전으로 적재·검색된다."""
        params = {"chunker": self.chunk_strategy, "embed_model": self.embed_model, "embed_dim": self.embed_dim}
        if self.chunk_strategy == "split":
            params |= {"chunk_tokens": self.chunk_tokens, "chunk_overlap": self.chunk_overlap}
        return hashlib.sha1(json.dumps(params, sort_keys=True).encode()).hexdigest()[:8]


settings = Settings()
