"""eval.scoring 테스트: 묶음별 채점 규칙, 실패 원인 분류, 집계."""

from eval.scoring import score_record, summarize

DOC = "doc.pdf"
EV = [{"document_id": DOC, "page": 10, "quote": "월 20만원"}]
QUANT = {"id": "q", "type": "quant", "question": "얼마?", "answer": "월 20만원", "evidence": EV}
QUAL = {"id": "s", "type": "qual", "question": "정리해줘", "must_not": ["틀린 말"],
        "must_include": [{"facts": ["가", "나"], "evidence": [{"document_id": DOC, "page": 10}]},
                         {"facts": ["다", "라"], "evidence": [{"document_id": DOC, "page": 20}]}]}
OUT = {"id": "n", "type": "unanswerable", "expected_status": "insufficient_context", "scoring": {"2": "거절", "0": "지어냄"}}
TRAP = {"id": "t", "type": "false_premise", "expected_status": "answered", "scoring": {"2": "정정", "1": "거절", "0": "동의"}}

HIT = [{"ref": 1, "document_id": DOC, "page_start": 10, "page_end": 10, "content": "월 20만원 지원"}]
MISS = [{"ref": 1, "document_id": DOC, "page_start": 99, "page_end": 99, "content": "다른 내용"}]


def rec(status="answered", answer="월 20만원입니다.", retrieved=HIT, citations=(), checks=None):
    return {"status": status, "answer": answer, "retrieved": retrieved, "citations": list(citations),
            "judge": {"checks": checks or {}}}


def support(p=0.9):
    return {"s0": {"kind": "support", "p": p}}


def test_quant_rules():
    cited = [{"document_id": DOC, "page_start": 10, "page_end": 10}]
    ok = support() | {"value": {"kind": "value", "p": 0.9}}
    assert score_record(QUANT, rec(citations=cited, checks=ok))["score"] == 2
    assert score_record(QUANT, rec(checks=ok)) | {"detail": None} == {"score": 1, "cause": "인용 오류", "detail": None}
    wrong = support() | {"value": {"kind": "value", "p": 0.1}}
    assert score_record(QUANT, rec(citations=cited, checks=wrong))["cause"] == "환각"
    unsupported = support(0.1) | {"value": {"kind": "value", "p": 0.9}}
    assert score_record(QUANT, rec(citations=cited, checks=unsupported))["score"] == 0
    refused = score_record(QUANT, rec(status="insufficient_context", retrieved=MISS))
    assert (refused["score"], refused["cause"], refused["detail"]["refusal_reason"]) == (1, "오거절", "검색 실패")


def fact_checks(ps, must_not_p=0.1):
    keys = ["f0_0", "f0_1", "f1_0", "f1_1"]
    return support() | {k: {"kind": "fact", "p": p} for k, p in zip(keys, ps)} | {
        "m0": {"kind": "must_not", "target": "틀린 말", "p": must_not_p}}


def test_qual_rules():
    assert score_record(QUAL, rec(checks=fact_checks([0.9, 0.9, 0.9, 0.1])))["score"] == 2  # 3/4 = 75%
    assert score_record(QUAL, rec(checks=fact_checks([0.9, 0.9, 0.9, 0.9], must_not_p=0.8)))["cause"] == "환각"
    # 빠진 사실(다·라)의 근거 쪽 20이 검색되지 않음 -> 검색 실패
    assert score_record(QUAL, rec(checks=fact_checks([0.9, 0.9, 0.1, 0.1])))["cause"] == "검색 실패"
    both = HIT + [{"ref": 2, "document_id": DOC, "page_start": 20, "page_end": 20, "content": ""}]
    assert score_record(QUAL, rec(retrieved=both, checks=fact_checks([0.9, 0.9, 0.1, 0.1])))["cause"] == "생성 누락"
    assert score_record(QUAL, rec(status="insufficient_context"))["score"] == 1


def test_negative_and_trap_rules():
    assert score_record(OUT, rec(status="insufficient_context"))["score"] == 2  # 코퍼스 밖은 거절이 정답
    assert score_record(TRAP, rec(status="insufficient_context"))["cause"] == "오거절"  # 답이 있는 함정
    grade = lambda c: {"grade": {"kind": "grade", "choice": c, "probabilities": {c: 1.0}}}  # noqa: E731
    assert score_record(TRAP, rec(checks=support() | grade("0")))["cause"] == "환각"
    assert score_record(TRAP, rec(checks=support(0.1) | grade("2")))["score"] == 0  # 근거 없는 문장이 있으면 환각


def test_summarize_groups_and_repeat_ranges():
    items = {"q": QUANT, "n": OUT}
    records = []
    for repeat, score in [(0, 2), (1, 0)]:
        records.append({"id": "q", "group": "정량", "type": "quant", "repeat": repeat, "status": "answered",
                        "answer": "x", "retrieved": HIT, "citations": [], "usage": None, "latency_ms": 100,
                        "result": {"score": score, "cause": None if score else "환각",
                                   "detail": {"value_ok": score == 2, "cited_evidence": True, "first_hit_rank": 1,
                                              "unsupported_sentences": [], "sentences_judged": 1}}})
    records.append({"id": "n", "group": "답 없음·함정", "type": "unanswerable", "repeat": 0,
                    "status": "insufficient_context", "retrieved": [], "citations": [], "latency_ms": 50,
                    "result": {"score": 2, "cause": None, "detail": {}}})
    summary = summarize(items, records)
    assert summary["groups"]["정량"]["safety"] == 0.5 and summary["groups"]["정량"]["safety_range"] == [0.0, 1.0]
    assert summary["groups"]["전체"]["dist"] == {"0": 1, "1": 0, "2": 2}
    assert summary["diagnostics"]["abstention_accuracy"] == 1.0 and summary["causes"] == {"환각": 1}
