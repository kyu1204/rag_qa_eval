"""app.embed.embed() 테스트: 배치 분할, 응답 순서 복원, 진행 콜백, 차원 검증."""

from types import SimpleNamespace

import pytest

from app import embed as embed_mod
from app.config import settings


class FakeAPI:
    """embeddings.create만 흉내 내는 가짜 클라이언트. 응답 순서를 일부러 뒤집는다."""

    def __init__(self, dim):
        self.dim, self.calls = dim, []
        self.embeddings = SimpleNamespace(create=self.create)

    def create(self, model, input):
        self.calls.append(len(input))
        data = [SimpleNamespace(index=i, embedding=[float(i)] * self.dim) for i in reversed(range(len(input)))]
        return SimpleNamespace(data=data)


def test_embed_batches_keeps_order_and_reports_progress():
    api, seen = FakeAPI(settings.embed_dim), []
    vectors = embed_mod.embed([f"t{i}" for i in range(150)], api, on_batch=seen.append)
    assert api.calls == [64, 64, 22] and seen == [64, 64, 22]
    assert len(vectors) == 150 and vectors[1][0] == 1.0  # 응답 순서가 뒤섞여도 index로 정렬


def test_embed_rejects_wrong_dimension():
    with pytest.raises(ValueError):
        embed_mod.embed(["a"], FakeAPI(settings.embed_dim + 1))
