"""eval CLI 테스트: 설정 덮어쓰기 파싱, 두 실행 비교, 리포트 렌더링, 프로덕션 설정과 실험 설정 일치."""

import json

import pytest

from app.config import Settings
from eval import __main__ as cli
from eval import report

GROUP = lambda safety, use: {"n": 2, "errors": 0, "safety": safety, "usefulness": use,  # noqa: E731
                             "safety_range": None, "usefulness_range": None, "dist": {"0": 0, "1": 1, "2": 1}}
DIAG = {key: None for key in report.DIAG_LABELS} | {"judge_cost_usd": 0.001}


def test_parse_overrides_merges_experiment_and_set(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "eval/experiments").mkdir(parents=True)
    (tmp_path / "eval/experiments/policy.toml").write_text('[settings]\nchunk_strategy = "policy"\ntop_k = 5\n')
    assert cli.parse_overrides(["TOP_K=8", "llm_model = x"], "policy") == {
        "CHUNK_STRATEGY": "policy", "TOP_K": "8", "LLM_MODEL": "x"}  # --set이 실험 파일보다 우선
    with pytest.raises(SystemExit):
        cli.parse_overrides(["TOP_K"], None)


def test_prod_env_serves_the_best_experiment():
    """.env.prod(서비스 최고 설정)는 실험 5 종합 (top_k 30)과 생성·검색 설정이 같고, 커밋되므로 키가 없어야 한다."""
    prod = Settings(_env_file=".env.prod")
    best = cli.parse_overrides([], "combined-k30")
    for key in ("LLM_REASONING_EFFORT", "RAG_PROMPT", "TOP_K"):
        assert str(getattr(prod, key.lower())) == best[key]
    assert not (prod.elice_api_key or prod.typesafe_api_key)


def _run_dir(tmp_path, name, scores, safety):
    d = tmp_path / name
    d.mkdir()
    info = {"name": name, "gold": {"hash": "g1"}, "overrides": {}, "index_version": "v"}
    records = [{"id": qid, "result": {"score": s}} for qid, s in scores.items()]
    summary = {"groups": {g: GROUP(safety, 0.5) for g in ["정량", "정성", "답 없음·함정", "전체"]}, "diagnostics": DIAG}
    (d / "run.json").write_text(json.dumps(info))
    (d / "items.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    (d / "summary.json").write_text(json.dumps(summary))
    return d


def test_compare_reports_deltas_and_item_changes(tmp_path, capsys):
    a = _run_dir(tmp_path, "a", {"q1": 1, "q2": 2, "q3": 2}, 0.5)
    b = _run_dir(tmp_path, "b", {"q1": 2, "q2": 0, "q3": 2}, 0.75)
    cli.main(["compare", str(a), str(b)])
    out = capsys.readouterr().out
    assert "50.0% -> 75.0% (+25.0%p)" in out
    assert "좋아진 문항 1: q1(1->2)" in out and "나빠진 문항 1: q2(2->0)" in out


def test_markdown_renders_without_failures():
    info = {"name": "x", "started_at": "t0", "finished_at": "t1", "repeats": 1, "judge_repeats": 2,
            "settings": {"extractor": "generic", "chunk_strategy": "policy", "top_k": 5, "min_score": 0.0},
            "index_version": "v", "index_chunks": 1, "overrides": {},
            "models": {k: "m" for k in ("embed", "llm", "llm_reasoning_effort", "llm_seed", "rag_prompt", "judge", "judge_template")},
            "gold": {"name": "g", "hash": "h", "items": 0}, "git": {"sha": "s", "dirty": False}}
    summary = {"groups": {g: GROUP(1.0, 1.0) for g in ["정량", "정성", "답 없음·함정", "전체"]}, "diagnostics": DIAG, "causes": {}}
    text = report.markdown(info, [], summary)
    assert "## 대표 지표" in text and "## 실패 문항 (0건)" in text and "- 없음" in text
