"""eval.goldset 테스트: 문항 스키마 검증, approved 필터, 해시, 중복 id."""

import json

import pytest

from eval.goldset import load, validate

EV = [{"document_id": "doc.pdf", "page": 3, "quote": "월 20만원"}]
QUANT = {"id": "q1", "type": "quant", "question": "얼마?", "answer": "월 20만원", "evidence": EV,
         "status": "approved", "provenance": "test"}
QUAL = {"id": "s1", "type": "qual", "question": "정리해줘", "status": "approved", "provenance": "test",
        "must_include": [{"point": "p", "facts": ["사실"], "evidence": [{"document_id": "doc.pdf", "page": 3}]}]}
NEG = {"id": "n1", "type": "unanswerable", "question": "없는 것?", "expected_status": "insufficient_context",
       "scoring": {"2": "거절", "0": "지어냄"}, "status": "approved", "provenance": "test"}


def test_valid_items_pass():
    assert validate(QUANT) == validate(QUAL) == validate(NEG) == []


@pytest.mark.parametrize("broken, message", [
    ({**QUANT, "evidence": []}, "evidence 없음"),
    ({**QUANT, "evidence": [{"document_id": "doc.pdf"}]}, "page 또는 quote"),
    ({**QUAL, "must_include": [{"point": "p", "facts": [], "evidence": EV}]}, "facts 없음"),
    ({**NEG, "scoring": {"1": "거절"}}, "2와 0"),
    ({**QUANT, "status": "maybe"}, "status는"),
    ({**QUANT, "type": "essay"}, "알 수 없는 type"),
])
def test_invalid_items_are_reported(broken, message):
    assert any(message in e for e in validate(broken))


def _write(tmp_path, items):
    (tmp_path / "manifest.json").write_text(json.dumps(
        {"name": "t", "documents": [{"id": "doc.pdf", "sha256": "abc"}], "files": ["items.jsonl"]}))
    (tmp_path / "items.jsonl").write_text("".join(json.dumps(i, ensure_ascii=False) + "\n" for i in items))


def test_load_keeps_only_approved_and_hashes_content(tmp_path):
    _write(tmp_path, [QUANT, QUAL, {**NEG, "status": "draft"}])
    gold = load(tmp_path)
    assert [i["id"] for i in gold.items] == ["q1", "s1"] and gold.skipped == 1
    first = gold.hash
    _write(tmp_path, [QUANT, {**QUAL, "question": "바뀐 질문"}, {**NEG, "status": "draft"}])
    assert load(tmp_path).hash != first  # 문항이 바뀌면 골드셋 해시도 바뀐다


def test_load_rejects_duplicate_ids(tmp_path):
    _write(tmp_path, [QUANT, QUANT])
    with pytest.raises(ValueError, match="id 중복"):
        load(tmp_path)
