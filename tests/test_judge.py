"""eval.judge 테스트: 문장 분리, 판정 구성, 반복 평균. Jev는 가짜 클라이언트로 바꾼다."""

from eval import judge

QUAL = {"id": "s1", "type": "qual", "question": "정리해줘",
        "must_include": [{"facts": ["가", "나"]}, {"facts": ["다"]}],
        "must_not": ["청년 전체가 받는다고 답함 (실제 대상은 19~20세; PDF p.143)"]}
NEG = {"id": "n1", "type": "false_premise", "question": "4주 맞나요?", "scoring": {"2": "정정", "1": "거절", "0": "동의"}}
ANSWERED = {"status": "answered", "answer": "- **한도**가 늘어납니다 [1].\n- 2026년 7월부터 적용됩니다. [2]",
            "retrieved": [{"ref": 1, "content": "출처1"}, {"ref": 2, "content": "출처2"}]}


def test_split_sentences_strips_markup_and_citations():
    assert judge.split_sentences(ANSWERED["answer"]) == ["한도가 늘어납니다.", "2026년 7월부터 적용됩니다."]


def test_strip_explanation_removes_trailing_parenthetical():
    assert judge.strip_explanation(QUAL["must_not"][0]) == "청년 전체가 받는다고 답함"
    assert judge.strip_explanation("소득공제 한도(연 600만원)를 말함") == "소득공제 한도(연 600만원)를 말함"


def test_build_checks_per_type():
    state, checks = judge.build_checks(QUAL, ANSWERED)
    assert {k for k, (kind, _) in checks.items() if kind == "fact"} == {"f0_0", "f0_1", "f1_0"}
    assert checks["m0"] == ("must_not", "청년 전체가 받는다고 답함")
    assert [s["id"] for s in state["sentences"]] == ["s0", "s1"] and checks["s0"] == ("support", "s0")
    assert judge.build_checks(QUAL, {"status": "insufficient_context"}) == ({}, {})
    assert judge.build_checks(NEG, ANSWERED)[1]["grade"] == ("grade", NEG["scoring"])


class FakeJev:
    def __init__(self):
        self.calls = 0

    def ask(self, state, questions):
        self.calls += 1
        p = 0.2 if self.calls % 2 else 0.6  # 반복마다 흔들리는 확률
        answers = {}
        for cid, q in questions.items():
            if q["type"] == "choice":
                answers[cid] = {"probabilities": {"2": 1 - p, "1": p / 2, "0": p / 2}}
            else:
                answers[cid] = {"noul": p}
        return answers, 100


def test_judge_record_averages_repeats():
    out = judge.judge_record(NEG, ANSWERED, FakeJev(), repeats=2)
    assert out["judge_tokens"] == 200
    assert out["checks"]["s0"]["p"] == 0.4 and out["checks"]["s0"]["ps"] == [0.2, 0.6]  # 평균과 회차별 원값
    grade = out["checks"]["grade"]
    assert grade["choice"] == "2" and grade["probabilities"]["2"] == 0.6 and grade["choices"] == ["2", "2"]


def test_judge_record_splits_large_requests(monkeypatch):
    monkeypatch.setattr(judge, "MAX_QUESTIONS", 2)
    fake = FakeJev()
    out = judge.judge_record(QUAL, ANSWERED, fake, repeats=1)
    assert fake.calls == 3 and len(out["checks"]) == 6  # 판정 6개를 2개씩
