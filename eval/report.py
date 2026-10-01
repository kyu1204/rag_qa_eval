"""실행 결과 저장(run.json·items.jsonl·summary.json·report.md)과 콘솔 요약."""

import hashlib
import json
import platform
import subprocess
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.table import Table

from app import rag
from app.config import settings
from eval.judge import TEMPLATE_VERSION
from eval.scoring import GROUPS

RUNS = Path("eval/runs")
SECRET_FIELDS = {"elice_api_key", "typesafe_api_key", "database_url"}
console = Console()


def _git() -> dict:
    run = lambda *args: subprocess.run(["git", *args], capture_output=True, text=True).stdout.strip()  # noqa: E731
    return {"sha": run("rev-parse", "--short", "HEAD"), "dirty": bool(run("status", "--porcelain"))}


def public_settings() -> dict:
    """기록용 설정 덤프 (키·접속 문자열 제외)."""
    return {k: v for k, v in settings.model_dump().items() if k not in SECRET_FIELDS}


def run_info(name: str, gold, overrides: dict, repeats: int, chunks: int, started: datetime) -> dict:
    """재현에 필요한 모든 것: 설정·인덱스·모델 버전·골드셋/문서 해시·git·반복 횟수."""
    lock = Path("uv.lock")
    return {
        "name": name,
        "started_at": started.isoformat(timespec="seconds"),
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "gold": {"name": gold.name, "path": str(gold.path), "hash": gold.hash, "items": len(gold.items)},
        "documents": gold.documents,
        "overrides": overrides,
        "settings": public_settings(),
        "index_version": settings.index_version(),
        "index_chunks": chunks,
        "models": {"embed": settings.embed_model, "llm": settings.llm_model,
                   "llm_reasoning_effort": settings.llm_reasoning_effort, "llm_seed": settings.llm_seed,
                   "rag_prompt": rag.PROMPT_VERSION, "judge": settings.judge_model, "judge_template": TEMPLATE_VERSION},
        "repeats": repeats,
        "judge_repeats": settings.judge_repeats,
        "git": _git(),
        "python": platform.python_version(),
        "uv_lock_sha256": hashlib.sha256(lock.read_bytes()).hexdigest()[:12] if lock.exists() else None,
    }


def save(name: str, info: dict, records: list[dict], summary: dict) -> Path:
    out = RUNS / f"{datetime.now():%Y%m%d-%H%M%S}_{name}"
    out.mkdir(parents=True)
    (out / "run.json").write_text(json.dumps(info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out / "items.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out / "report.md").write_text(markdown(info, records, summary), encoding="utf-8")
    return out


def _pct(v) -> str:
    return "-" if v is None else f"{v * 100:.1f}%"


def _range(r) -> str:
    return "" if not r or r[0] == r[1] else f" ({r[0] * 100:.0f}~{r[1] * 100:.0f}%)"


DIAG_LABELS = {
    "retrieval_hit_at_k": ("검색 hit@k (근거 쪽이 top-k에 있음)", _pct),
    "mrr": ("MRR", lambda v: "-" if v is None else f"{v:.3f}"),
    "quant_accuracy": ("정량 정답률", _pct),
    "quant_citation_accuracy": ("정량 근거 쪽 인용률 (정답 중)", _pct),
    "qual_fact_coverage": ("정성 사실 커버율", _pct),
    "qual_fact_coverage_soft": ("정성 사실 커버율 (확률 평균)", _pct),
    "faithfulness": ("충실도 (출처로 뒷받침되는 문장 비율)", _pct),
    "abstention_accuracy": ("응답 불가 정확도 (코퍼스 밖·세부 없음)", _pct),
    "false_refusal_rate": ("오거절률 (답이 있는데 거절)", _pct),
    "latency_p50_ms": ("지연 p50", lambda v: "-" if v is None else f"{v / 1000:.1f}s"),
    "latency_p95_ms": ("지연 p95", lambda v: "-" if v is None else f"{v / 1000:.1f}s"),
    "llm_cost_krw_per_query": ("생성 비용/질의", lambda v: "-" if v is None else f"₩{v:.1f}"),
    "judge_cost_usd": ("judge 비용 (전체)", lambda v: f"${v:.4f}"),
}


def _failure_line(r: dict) -> str:
    res, d = r["result"], r["result"]["detail"]
    bits = []
    if "value_p" in d:
        bits.append(f"값 판정 {d['value_p']:.2f}, 근거 쪽 인용 {'O' if d['cited_evidence'] else 'X'}")
    if "coverage" in d:
        missing = [f["fact"] for f in d["facts"] if not f["covered"]][:3]
        bits.append(f"사실 커버 {d['coverage'] * 100:.0f}%" + (f", 빠진 사실 예: {' / '.join(missing)}" if missing else ""))
    if d.get("must_not_violations"):
        bits.append(f"must_not 위반: {' / '.join(d['must_not_violations'])}")
    if d.get("unsupported_sentences"):
        bits.append(f"근거 없는 문장: {' / '.join(s[:60] for s in d['unsupported_sentences'][:2])}")
    if "grade" in d:
        bits.append(f"기준 선택 {d['grade']}")
    if d.get("refusal_reason"):
        bits.append(f"거절 원인 추정: {d['refusal_reason']}")
    if d.get("evidence_retrieved") is not None:
        bits.append(f"근거 쪽 검색 {'O' if d['evidence_retrieved'] else 'X'}")
    answer = (r.get("answer") or "(거절)").replace("\n", " ")[:120]
    rep = f" r{r['repeat']}" if r["repeat"] else ""
    return (f"- **{r['id']}**{rep} ({r['group']}, {res['score']}점, {res['cause']})  \n"
            f"  질문: {r['question']}  \n  답변: {answer}  \n  " + " · ".join(bits))


def markdown(info: dict, records: list[dict], summary: dict) -> str:
    s, m = info["settings"], info["models"]
    lines = [
        f"# 평가 리포트: {info['name']}", "",
        f"- 실행: {info['started_at']} ~ {info['finished_at']}, 반복 {info['repeats']}회 (judge 판정 {info['judge_repeats']}회 평균)",
        f"- 설정: extractor={s['extractor']}, chunker={s['chunk_strategy']}"
        + (f" {s['chunk_tokens']}/{s['chunk_overlap']}" if s["chunk_strategy"] == "split" else "")
        + f", top_k={s['top_k']}, min_score={s['min_score']}, index={info['index_version']} ({info['index_chunks']}청크)",
        f"- 모델: 임베딩 {m['embed']}, 생성 {m['llm']} (effort {m['llm_reasoning_effort']}, seed {m['llm_seed']}, prompt {m['rag_prompt']}),"
        f" judge {m['judge']} ({m['judge_template']})",
        f"- 덮어쓴 설정: {info['overrides'] or '없음'}",
        f"- 골드셋 {info['gold']['name']} {info['gold']['hash']} ({info['gold']['items']}문항) · git {info['git']['sha']}"
        + (" (미커밋 변경 있음)" if info["git"]["dirty"] else ""), "",
        "## 대표 지표", "",
        "안전성 = 1점 이상 비율(환각 없음), 유용성 = 평균 점수 / 2. 괄호는 반복 실행 간 범위.", "",
        "| 묶음 | 문항 | 안전성 | 유용성 | 0점 | 1점 | 2점 | 오류 |", "|---|---|---|---|---|---|---|---|",
    ]
    for name in GROUPS + ["전체"]:
        g = summary["groups"][name]
        lines.append(f"| {name} | {g['n']} | {_pct(g['safety'])}{_range(g['safety_range'])} | "
                     f"{_pct(g['usefulness'])}{_range(g['usefulness_range'])} | {g['dist']['0']} | {g['dist']['1']} | "
                     f"{g['dist']['2']} | {g['errors']} |")
    lines += ["", "## 진단 지표", "", "| 지표 | 값 |", "|---|---|"]
    lines += [f"| {label} | {fmt(summary['diagnostics'][key])} |" for key, (label, fmt) in DIAG_LABELS.items()]
    lines += ["", "## judge 일관성", "",
              f"같은 판정을 {info['judge_repeats']}회 반복한 흔들림. 뒤집힘 = 회차에 따라 문턱 기준 예/아니오가 달라진 비율.", ""]
    consistency = summary.get("judge_consistency") or {}
    if consistency:
        lines += ["| 판정 | 수 | 판정 뒤집힘 | 확률 차이 평균 | 최대 |", "|---|---|---|---|---|"]
        for kind, c in consistency.items():
            mean_diff = "-" if c["mean_diff"] is None else f"{c['mean_diff']:.3f}"
            max_diff = "-" if c["max_diff"] is None else f"{c['max_diff']:.3f}"
            lines.append(f"| {kind} | {c['n']} | {_pct(c['flip_rate'])} | {mean_diff} | {max_diff} |")
    else:
        lines.append("- 기록 없음 (회차별 확률을 저장하기 전의 실행이거나 judge 반복 1회)")
    lines += ["", "## 실패 원인", ""]
    lines += [f"- {cause}: {n}건" for cause, n in summary["causes"].items()] or ["- 없음"]
    failures = [r for r in records if r["result"]["score"] is None or r["result"]["score"] < 2]
    lines += ["", f"## 실패 문항 ({len(failures)}건)", ""]
    lines += [_failure_line(r) for r in sorted(failures, key=lambda r: (r["result"]["score"] or -1, r["id"]))]
    return "\n".join(lines) + "\n"


def print_summary(summary: dict, path: Path | None = None) -> None:
    table = Table(title="평가 결과")
    for col in ("묶음", "문항", "안전성", "유용성", "0/1/2점", "오류"):
        table.add_column(col, justify="left" if col == "묶음" else "right")
    for name in GROUPS + ["전체"]:
        g = summary["groups"][name]
        table.add_row(name, str(g["n"]), _pct(g["safety"]) + _range(g["safety_range"]),
                      _pct(g["usefulness"]) + _range(g["usefulness_range"]),
                      f"{g['dist']['0']}/{g['dist']['1']}/{g['dist']['2']}", str(g["errors"]))
    console.print(table)
    d = summary["diagnostics"]
    console.print(f"검색 hit@k {_pct(d['retrieval_hit_at_k'])} · 정량 정답률 {_pct(d['quant_accuracy'])} · "
                  f"정성 커버 {_pct(d['qual_fact_coverage'])} · 충실도 {_pct(d['faithfulness'])} · "
                  f"오거절 {_pct(d['false_refusal_rate'])}")
    console.print("실패 원인: " + (", ".join(f"{k} {v}" for k, v in summary["causes"].items()) or "없음"))
    if path:
        console.print(f"저장: {path}/report.md")
