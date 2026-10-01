"""eval.runner 테스트: 설정 덮어쓰기의 타입 검증·원복, 문항 오류 격리."""

import pytest

from app import rag
from app.config import settings
from eval import runner


def test_overridden_coerces_types_and_restores():
    before_k, before_strategy, before_version = settings.top_k, settings.chunk_strategy, settings.index_version()
    with runner.overridden({"TOP_K": "8", "CHUNK_STRATEGY": "policy", "EXTRACTOR": "policy_book"}):
        assert settings.top_k == 8 and isinstance(settings.top_k, int)
        assert settings.index_version() != before_version  # 다른 인덱스를 가리킨다
    assert (settings.top_k, settings.chunk_strategy, settings.index_version()) == (before_k, before_strategy, before_version)


def test_overridden_rejects_unknown_and_invalid_values():
    with pytest.raises(ValueError, match="알 수 없는 설정"):
        with runner.overridden({"TOPK": "8"}):
            pass
    with pytest.raises(ValueError):  # Literal 밖의 값은 pydantic이 거부
        with runner.overridden({"CHUNK_STRATEGY": "magic"}):
            pass


def test_run_one_records_errors_instead_of_raising(monkeypatch):
    def boom(question):
        raise TimeoutError("LLM timeout")
    monkeypatch.setattr(rag, "answer", boom)
    record = runner.run_one({"id": "q1", "type": "quant", "question": "얼마?"}, repeat=0)
    assert record["status"] == "error" and "TimeoutError" in record["error"]
