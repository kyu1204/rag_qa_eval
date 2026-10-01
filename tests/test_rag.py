"""app.rag 테스트: 인용 검증, 게이트1(검색 점수), 게이트2(센티널) 스트리밍 처리. DB·API는 가짜로 바꾼다."""

from types import SimpleNamespace

from app import rag
from app.config import settings


def _hits(n=2, score=0.6):
    return [rag.Hit(i + 1, i, "doc.pdf", "문서", f"내용 {i}", 10 + i, 10 + i, score) for i in range(n)]


def _fake_llm(*pieces):
    def chunk(text):
        return SimpleNamespace(usage=None, choices=[SimpleNamespace(delta=SimpleNamespace(content=text))])

    def create(**kwargs):
        usage = SimpleNamespace(usage=SimpleNamespace(prompt_tokens=10, completion_tokens=3), choices=[])
        return iter([chunk(p) for p in pieces] + [usage])

    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))


def _events(monkeypatch, pieces, hits):
    monkeypatch.setattr(rag, "retrieve", lambda q, k: hits)
    return list(rag.run("질문", api=_fake_llm(*pieces)))


def test_parse_citations_drops_unknown_refs():
    answer, citations, warnings = rag.parse_citations("연 1,800만원입니다 [1][7]. 근거 [2][1]", _hits())
    assert answer == "연 1,800만원입니다 [1]. 근거 [2][1]"
    assert [c["ref"] for c in citations] == [1, 2]
    assert warnings == ["존재하지 않는 출처 번호 제거: [7]"]


def test_parse_citations_flags_uncited_answer():
    assert rag.parse_citations("근거 없이 답함", _hits())[2] == ["인용 없는 답변"]


def test_answer_streams_deltas_and_cites(monkeypatch):
    events = _events(monkeypatch, ["연 1,800", "만원 [1]"], _hits())
    kinds = [k for k, _ in events]
    assert kinds[0] == "sources" and kinds[-1] == "done" and "delta" in kinds
    result = events[-1][1]
    assert result.status == "answered" and result.answer == "연 1,800만원 [1]"
    assert result.meta["usage"] == {"prompt_tokens": 10, "completion_tokens": 3}


def test_sentinel_is_never_streamed(monkeypatch):
    events = _events(monkeypatch, ["[[NO_", "ANSWER]]"], _hits())
    assert not [p for k, p in events if k == "delta"]
    assert events[-1][1].status == "insufficient_context"


def test_low_retrieval_score_skips_llm(monkeypatch):
    monkeypatch.setattr(settings, "min_score", 0.5)
    events = _events(monkeypatch, ["호출되면 안 됨"], _hits(score=0.3))
    assert [k for k, _ in events] == ["sources", "done"]
    assert events[-1][1].warnings == ["검색 점수 미달"]


def test_temperature_only_sent_without_reasoning(monkeypatch):
    monkeypatch.setattr(settings, "llm_reasoning_effort", "none")
    assert rag.sampling_params()["temperature"] == settings.llm_temperature
    assert "temperature" not in rag.sampling_params("low")  # Luna는 low 이상에서 temperature=0을 거부한다


def test_build_messages_uses_the_selected_prompt(monkeypatch):
    assert "출처로 답할 수 없으면" in rag.build_messages("질문", _hits())[0]["content"]  # 기본 v1
    monkeypatch.setattr(settings, "rag_prompt", "v2")
    system = rag.build_messages("질문", _hits())[0]["content"]
    assert "일부만 답할 수 있으면" in system and rag.NO_ANSWER in system


def test_prompt_v3_adds_the_effective_date_check(monkeypatch):
    monkeypatch.setattr(settings, "rag_prompt", "v3")
    system = rag.build_messages("질문", _hits())[0]["content"]
    assert "시행일이 질문 속 시점보다 늦으면" in system and "일부만 답할 수 있으면" in system
