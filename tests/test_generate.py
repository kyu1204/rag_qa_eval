"""eval.generate 테스트: 틀린 값 만들기, 고른 구간 뽑기, 가짜 LLM으로 전체 생성 흐름과 기계 검증."""

import json
import re
from types import SimpleNamespace

import pytest

from eval import generate
from eval.goldset import load, validate

DOC = """청년 월세 지원은 만 19~34세 무주택 청년에게 월 20만원을 최대 12개월 지원한다.

신청은 복지로 누리집이나 주민센터에서 한다. 소득 기준은 기준 중위소득 60% 이하이다.

목차 1. 주거 2. 일자리 3. 교육 4. 복지 5. 문화 6. 환경 7. 안전 8. 행정"""


def test_perturb_changes_last_number_and_skips_values_in_source():
    assert generate.perturb("월 20만원", "월 20만원 지원") == ("월 40만원", "40만")
    assert generate.perturb("만 19~34세", "") == ("만 19~39세", "39세")  # 범위는 뒤쪽을 바꿔야 뒤집히지 않는다
    assert generate.perturb("1,800만원", "") == ("3,600만원", "3,600만")
    assert generate.perturb("매년 1회", "매년 2회 점검") == ("매년 3회", "3회")  # 원문에 있는 값은 피한다
    assert generate.perturb("2026년 10월 22일", "")[0] == "2026년 10월 15일"  # 29일 대신 어느 달에나 있는 날
    assert generate.perturb("2026.7.1.", "")[0] == "2026.7.8."
    assert generate.perturb("무주택 청년", "") is None


def test_stratified_spreads_first_picks():
    order = generate.stratified(list(range(10)), 5, seed=1)
    assert sorted(x // 2 for x in order[:5]) == [0, 1, 2, 3, 4] and sorted(order) == list(range(10))


def fake_llm(prompt: str) -> dict:
    if "루브릭 포인트" in prompt:  # eval/atomize
        return {"facts": [re.search(r"\[포인트\] (.+)", prompt).group(1)]}
    if "[틀린 값]" in prompt:
        wrong = re.search(r"\[틀린 값\] (.+)", prompt).group(1)
        return {"question": f"청년 월세 지원이 {wrong}이라던데 맞나요?"}
    if "[구간 앞부분]" in prompt:
        return {"items": [{"question": "2027년 최저임금은 얼마인가요?", "keywords": ["최저임금"]},
                          {"question": "월세 지원은 얼마인가요?", "keywords": ["월세"]}]}
    section = prompt.split("[구간]\n")[-1]  # 지시문 말고 구간 본문만 본다
    if "월 20만원" not in section:
        return {"skip": "목차"} if section.startswith("목차") else {
            "question_literal": "q", "question_user": "q", "answer": "50%", "quote": "지어낸 인용구 50%"}
    if "must_include" in prompt:
        return {"subtype": "요약", "question": "청년 월세 지원 내용을 정리해줘",
                "must_include": [{"point": "만 19~34세 무주택 청년이 대상이다", "quote": "만 19~34세 무주택 청년에게"},
                                 {"point": "월 20만원을 최대 12개월 받는다", "quote": "월 20만원을 최대 12개월 지원한다"},
                                 {"point": "지어낸 포인트", "quote": "문서에 없는 문장"}],
                "must_not": ["월 30만원을 지원한다 (실제: 월 20만원)"]}
    return {"question_literal": "청년 월세 지원 금액은 월 얼마인가요?", "question_user": "월세 도움은 한 달에 얼마예요?",
            "answer": "월 20만원", "answer_kind": "금액", "quote": "월 20만원을 최대 12개월 지원한다"}


def test_generate_writes_verified_drafts(tmp_path, monkeypatch):
    monkeypatch.setattr(generate, "WINDOW_TOKENS", 40)  # 문단마다 구간 하나
    doc = tmp_path / "doc.md"
    doc.write_text(DOC, encoding="utf-8")
    reply = lambda **kw: SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(  # noqa: E731
        content=json.dumps(fake_llm(kw["messages"][0]["content"]), ensure_ascii=False)))])
    api = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=reply)))
    out = tmp_path / "gold"
    build = generate.generate([str(doc)], out, n_quant=3, n_qual=1, n_traps=1, n_out=2, api=api)

    rows = {kind: [json.loads(line) for line in (out / f"{kind}.jsonl").open(encoding="utf-8")] for kind in ("quant", "qual", "neg")}
    items = [r for rs in rows.values() for r in rs]
    assert all(validate(r) == [] and r["status"] == "draft" for r in items)
    assert [r["id"] for r in rows["quant"]] == ["quant-01-literal", "quant-01-user"]  # 지어낸 인용구는 탈락
    assert rows["quant"][0]["evidence"] == [{"document_id": "doc.md", "page": None, "quote": "월 20만원을 최대 12개월 지원한다"}]
    trap, out_item = rows["neg"]
    assert "월 40만원" in trap["question"] and trap["premise"]["from"] == "quant-01" and set(trap["scoring"]) == {"2", "1", "0"}
    assert out_item["keywords"] == ["최저임금"]  # "월세"는 문서에 있어서 탈락
    qual = rows["qual"][0]
    assert len(qual["must_include"]) == 2 and qual["dropped_points"][0]["point"] == "지어낸 포인트"
    reasons = {r["reason"] for r in build["rejects"]}
    assert {"인용구가 원문에 없음", "키워드가 문서에 있음 ['월세']"} <= reasons

    gold = load(out)  # 초안뿐이라 평가 대상 0개
    assert gold.items == [] and gold.skipped == len(items) and gold.documents[0]["id"] == "doc.md"
    review = (out / "review.md").read_text(encoding="utf-8")
    assert "<details>" not in review and not any(line.startswith(">") for line in review.splitlines())
    with pytest.raises(SystemExit):  # 검토한 골드셋을 덮어쓰지 않는다
        generate.generate([str(doc)], out, api=api)
