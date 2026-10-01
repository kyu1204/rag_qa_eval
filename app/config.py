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
    rag_prompt: Literal["v1", "v2"] = "v1"  # 생성 시스템 프롬프트 버전 (app/rag.py PROMPTS)

    top_k: int = 5
    min_score: float = 0.0  # 게이트1: top1 코사인 유사도가 이보다 낮으면 LLM 호출 없이 응답 불가. Part B에서 보정

    corpus_dir: str = "data/corpus"  # 평가 시 인덱스가 없으면 여기서 자동 적재

    # 평가 judge: TypeSafe Jev (판정 전용 모델, 버전 고정)
    typesafe_api_key: str = ""
    judge_endpoint: str = "https://api.typesafe.ai/v1/systemone"
    judge_model: str = "jev-1.13.0"
    judge_template: Literal["judge-ko-v1", "judge-ko-v2"] = "judge-ko-v1"  # 판정 문구 버전 (eval/judge.py TEMPLATE_SETS)
    judge_repeats: int = 2  # 같은 판정을 반복해 확률 평균 (PoC 흔들림 최대 0.09)
    tau_value: float = 0.5  # 정량 값 판정 문턱
    tau_fact: float = 0.8  # 정성 사실 포함 문턱. 사람 라벨 85건: 0.3이면 81%(judge가 일부만 있는 사실도 예), 0.75~0.85면 94%
    tau_must_not: float = 0.7  # 사람 라벨 14건 모두 위반 없음, 0.5면 오탐 2건. 실제 위반 표본이 없어 놓침은 미측정
    tau_support: float = 0.5  # 답변 문장이 출처로 뒷받침되는가
    coverage_full: float = 0.75  # 정성 2점에 필요한 사실 커버율

    # 비용 추정 (Elice 단가, 원/100만 토큰)
    llm_price_in: float = 304
    llm_price_out: float = 1827

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
