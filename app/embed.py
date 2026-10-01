from collections.abc import Callable

from openai import OpenAI

from app.config import settings

BATCH_SIZE = 64  # 실측: 100개 배치도 정상. 요청 하나가 너무 커지지 않게 여유를 둔다


def client() -> OpenAI:
    # Elice ML API는 모델마다 엔드포인트가 따로라 base_url이 곧 모델 선택이다
    return OpenAI(base_url=settings.embed_base_url, api_key=settings.elice_api_key, timeout=60, max_retries=5)


def embed(
    texts: list[str],
    api: OpenAI | None = None,
    on_batch: Callable[[int], None] | None = None,
) -> list[list[float]]:
    """texts를 배치로 임베딩한다. on_batch(처리한 개수)는 진행 표시용."""
    api = api or client()
    vectors: list[list[float]] = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        resp = api.embeddings.create(model=settings.embed_model, input=batch)
        got = [d.embedding for d in sorted(resp.data, key=lambda d: d.index)]
        if len(got) != len(batch) or any(len(v) != settings.embed_dim for v in got):
            raise ValueError(
                f"임베딩 응답 불일치: {len(got)}/{len(batch)}개, 차원 {len(got[0]) if got else 0} != {settings.embed_dim}"
            )
        vectors += got
        if on_batch:
            on_batch(len(batch))
    return vectors
