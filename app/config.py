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

    llm_base_url: str = ""
    llm_model: str = ""
    llm_reasoning_effort: Literal["none", "low", "medium", "high"] = "none"
    llm_temperature: float = 0.0
    llm_seed: int = 42

    top_k: int = 5
    min_score: float = 0.0  # 게이트1: top1 코사인 유사도가 이보다 낮으면 LLM 호출 없이 응답 불가. Part B에서 보정

    extractor: Literal["generic", "policy_book"] = "generic"  # policy_book = 책자 전용 분석기
    chunk_strategy: Literal["split", "policy"] = "split"  # policy는 policy_book 추출 결과 전용
    chunk_tokens: int = 500  # split 전략에서만 사용
    chunk_overlap: int = 75  # split 전략에서만 사용

    def index_version(self) -> str:
        """청킹·임베딩 설정의 지문. 설정이 바뀌면 다른 버전으로 적재·검색된다."""
        params = {"extractor": self.extractor, "chunker": self.chunk_strategy, "embed_model": self.embed_model, "embed_dim": self.embed_dim}
        if self.chunk_strategy == "split":
            params |= {"chunk_tokens": self.chunk_tokens, "chunk_overlap": self.chunk_overlap}
        return hashlib.sha1(json.dumps(params, sort_keys=True).encode()).hexdigest()[:8]


settings = Settings()
