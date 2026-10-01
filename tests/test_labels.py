"""eval.labels 테스트: 라벨링 시트 생성·읽기와 일치도·문턱 계산."""

import json
import re

import pytest

from eval import labels

EV = [{"document_id": "doc.pdf", "page": 1}]
QUAL = {"id": "s1", "type": "qual", "question": "정리해줘", "status": "approved", "provenance": "t",
        "must_not": ["틀린 말 (설명)"],
        "must_include": [{"facts": ["가 | 나", "다", "라", "마"], "evidence": EV}]}
QUANT = {"id": "q1", "type": "quant", "question": "얼마?", "answer": "10원", "status": "approved", "provenance": "t",
         "evidence": EV}


def _setup(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    gold = tmp_path / "gold"
    gold.mkdir()
    (gold / "manifest.json").write_text(json.dumps({"name": "g", "documents": [], "files": ["i.jsonl"]}))
    (gold / "i.jsonl").write_text("".join(json.dumps(i, ensure_ascii=False) + "\n" for i in (QUAL, QUANT)))
    run = tmp_path / "eval/runs/r1_base"
    run.mkdir(parents=True)
    (run / "run.json").write_text(json.dumps({"name": "base", "gold": {"path": str(gold)}}))
    fact_p = {"f0_0": 0.9, "f0_1": 0.4, "f0_2": 0.2, "f0_3": 0.05}
    checks = {k: {"kind": "fact", "p": p} for k, p in fact_p.items()} | {
        "m0": {"kind": "must_not", "p": 0.6}, "s0": {"kind": "support", "p": 0.95}}
    records = [
        {"id": "s1", "repeat": 0, "status": "answered", "type": "qual", "group": "정성", "question": "정리해줘",
         "answer": "가, 다를 받습니다.", "retrieved": [{"ref": 1, "page_start": 1, "page_end": 1, "content": "<출처> | 표\n\n둘째 줄"}],
         "judge": {"checks": checks}},
        {"id": "q1", "repeat": 0, "status": "answered", "type": "quant", "group": "정량", "question": "얼마?",
         "answer": "10원입니다.", "retrieved": [], "judge": {"checks": {"value": {"kind": "value", "p": 0.97}}}},
    ]
    (run / "items.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records))
    return run


def _fill(sheet, verdicts):
    text = sheet.read_text()
    for key, verdict in verdicts.items():
        text = re.sub(rf"^(\| {re.escape(key)} \|.*\|)  \|  \|$", rf"\1 {verdict} |  |", text, flags=re.M)
    sheet.write_text(text)


def test_sheet_roundtrip_and_calibration(tmp_path, monkeypatch):
    run = _setup(tmp_path, monkeypatch)
    sheet = labels.make_sheet(run, support_n=5)
    assert "가 \\| 나" in sheet.read_text()  # 표 안의 | 는 이스케이프
    assert "0.9" not in sheet.read_text()  # judge 확률은 시트에 없다
    fold = sheet.read_text().split("<details>")[1].split("</details>")[0]
    assert "\n\n" not in fold and "&lt;출처&gt; &#124; 표" in fold  # 접힘이 한 HTML 블록, 내용은 이스케이프
    # 사람: 가·다는 있음(O), 라·마는 없음, 틀린 말은 안 함(X), 정량 값 맞음
    _fill(sheet, {"s1#f0_0": "O", "s1#f0_1": "O", "s1#f0_2": "X", "s1#f0_3": "x",
                  "s1#m0": "X", "s1#s0": "?", "q1#value": "O"})
    run_path, parsed = labels.read_sheet(sheet)
    assert run_path == str(run) and parsed["s1#f0_3"] == ("X", "") and len(parsed) == 7

    from app.config import settings
    monkeypatch.setattr(settings, "tau_fact", 0.5)
    monkeypatch.setattr(settings, "tau_must_not", 0.5)
    result = labels.calibrate(sheet)
    fact = result["kinds"]["fact"]
    # 문턱 0.5: judge는 가만 예 -> 다(0.4)를 놓쳐 4개 중 3개 일치
    assert fact["accuracy"] == 0.75 and fact["confusion"]["judge_no_human_yes"] == 1
    assert 0.2 < fact["best_tau"] < 0.4 and fact["best_tau_accuracy"] == 1.0
    assert result["kinds"]["must_not"]["confusion"]["judge_yes_human_no"] == 1  # 0.6 오탐
    assert result["skipped_uncertain"] == 1 and result["kinds"]["value"]["accuracy"] == 1.0


@pytest.mark.parametrize("pairs, expected", [([(True, True), (False, False)], 1.0), ([(True, False), (False, True)], -1.0)])
def test_kappa(pairs, expected):
    assert labels._kappa(pairs) == expected


def test_sheet_spells_out_verdicts_and_rejects_other_values(tmp_path, monkeypatch):
    sheet = labels.make_sheet(_setup(tmp_path, monkeypatch), support_n=5)
    assert "| s1#m0 | 잘못된 내용 (O=위반) |" in sheet.read_text()  # 정상 답변을 O로 적는 혼동 방지
    _fill(sheet, {"q1#value": "2"})  # 함정 기준 숫자를 다른 칸에
    with pytest.raises(ValueError, match="q1#value=2"):
        labels.read_sheet(sheet)
