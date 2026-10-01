"""eval.retrieval 테스트: 근거 단위 순위와 top-k별 hit·재현율 계산."""

from eval import retrieval

EV = lambda page: [{"document_id": "d", "page": page}]  # noqa: E731
QUANT = {"id": "q", "type": "quant", "evidence": EV(10)}
QUAL = {"id": "s", "type": "qual", "must_include": [{"evidence": EV(3)}, {"evidence": EV(7) + EV(8)}]}
OUT = {"id": "n", "type": "unanswerable"}


def test_units_and_first_rank():
    assert retrieval.units(QUAL) == [EV(3), EV(7) + EV(8)] and retrieval.units(OUT) == []
    hits = [{"document_id": "d", "page_start": p, "page_end": p, "content": ""} for p in (9, 8, 10)]
    assert retrieval.first_rank(hits, EV(10)) == 3
    assert retrieval.first_rank(hits, EV(7) + EV(8)) == 2  # 단위 안의 근거 중 하나만 잡혀도 된다
    assert retrieval.first_rank(hits, EV(3)) is None


def test_curve_counts_item_hits_and_unit_recall():
    out = retrieval.curve([QUANT, QUAL, OUT], {"q": [3], "s": [1, None]}, [1, 5])
    assert out["전체"]["items"] == 2 and out["전체"]["units"] == 3  # 근거 없는 문항은 빠진다
    assert out["전체"]["hit"] == {1: 0.5, 5: 1.0}
    assert out["정성"]["recall"] == {1: 0.5, 5: 0.5} and out["정량"]["recall"] == {1: 0.0, 5: 1.0}
