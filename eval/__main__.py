"""평가 하네스 CLI.

    uv run python -m eval run [--name NAME] [--set KEY=VALUE ...] [--exp NAME] [--repeats N] [--limit N]
    uv run python -m eval compare <run A 디렉터리> <run B 디렉터리>
    uv run python -m eval rescore <run 디렉터리> [--set TAU_FACT=0.25 ...]
    uv run python -m eval rejudge <run 디렉터리> [--set JUDGE_REPEATS=3 ...]
    uv run python -m eval label-sheet <run 디렉터리>
    uv run python -m eval calibrate <채운 라벨링 시트>
    uv run python -m eval generate <문서 ...> --out <새 골드셋 디렉터리> [--quant 10 --qual 5 --traps 5 --out-of-corpus 5]
"""

import argparse
import json
import sys
import tomllib
from datetime import datetime
from pathlib import Path

from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn

from app.db import connect
from eval import generate, judge, labels, report, runner, scoring
from eval.goldset import document_problems, load


def parse_overrides(pairs: list[str], exp: str | None) -> dict[str, str]:
    overrides: dict[str, str] = {}
    if exp:
        path = Path("eval/experiments") / f"{exp}.toml"
        overrides |= {k.upper(): str(v) for k, v in tomllib.loads(path.read_text()).get("settings", {}).items()}
    for pair in pairs:
        key, sep, value = pair.partition("=")
        if not sep:
            raise SystemExit(f"--set은 KEY=VALUE 형식: {pair}")
        overrides[key.strip().upper()] = value.strip()
    return overrides


def cmd_run(args) -> int:
    overrides = parse_overrides(args.set, args.exp)
    name = args.name or args.exp or "run"
    started = datetime.now()
    with runner.overridden(overrides):
        gold = load(args.gold)
        if args.limit:
            gold.items = gold.items[: args.limit]
        chunks = runner.ensure_index()
        if problems := document_problems(gold, connect()):
            raise SystemExit("골드셋과 DB 문서가 다르다:\n" + "\n".join(problems))
        items_by_id = {item["id"]: item for item in gold.items}
        columns = [TextColumn("{task.description}"), BarColumn(), MofNCompleteColumn(), TimeElapsedColumn()]
        with Progress(*columns, console=report.console, transient=True) as progress:
            task = progress.add_task("RAG 실행", total=len(gold.items) * args.repeats)
            records = runner.run_items(gold, args.repeats, on_done=lambda _: progress.advance(task))
            progress.update(task, description="judge 판정 중")
            judge.judge_all(items_by_id, [r for r in records if r["status"] != "error"])
        summary = scoring.score_all(items_by_id, records)
        info = report.run_info(name, gold, overrides, args.repeats, chunks, started)
    path = report.save(name, info, records, summary)
    report.print_summary(summary, path)
    return 1 if summary["groups"]["전체"]["errors"] else 0


def _load_run(path: Path) -> tuple[dict, list[dict], dict]:
    info = json.loads((path / "run.json").read_text())
    records = [json.loads(line) for line in (path / "items.jsonl").open(encoding="utf-8")]
    return info, records, json.loads((path / "summary.json").read_text())


def _gold_items(info: dict) -> tuple[dict[str, dict], dict]:
    """(문항, 기록용 gold 항목). 실행 당시 골드셋을 다시 읽어, 그 사이 바뀌었으면 경고한다.
    판정·채점이 새 골드셋 기준이 되므로 기록도 지금 읽은 해시로 바꾼다."""
    gold = load(info["gold"]["path"])
    if gold.hash != info["gold"]["hash"]:
        report.console.print(f"[yellow]경고: 골드셋이 실행 당시({info['gold']['hash']})와 다르다({gold.hash})[/]")
    return {item["id"]: item for item in gold.items}, info["gold"] | {"hash": gold.hash, "items": len(gold.items)}


def cmd_rescore(args) -> int:
    """저장된 답변·판정 확률로 채점만 다시 한다 (문턱 보정 등). 결과는 새 실행 디렉터리로."""
    src = Path(args.run)
    info, records, _ = _load_run(src)
    overrides = parse_overrides(args.set, None)
    with runner.overridden(info["overrides"] | overrides):
        items_by_id, gold_info = _gold_items(info)
        summary = scoring.score_all(items_by_id, records)
        info = info | {"name": f"{info['name']}_rescored", "rescored_from": str(src), "rescore_overrides": overrides,
                       "settings": report.public_settings(), "gold": gold_info}
    path = report.save(info["name"], info, records, summary)
    report.print_summary(summary, path)
    return 0


def cmd_rejudge(args) -> int:
    """저장된 답변은 그대로 두고 judge만 다시 돌린다 (judge 설정·템플릿·반복 횟수를 바꿨을 때)."""
    src = Path(args.run)
    info, records, _ = _load_run(src)
    overrides = parse_overrides(args.set, None)
    with runner.overridden(info["overrides"] | overrides):
        items_by_id, gold_info = _gold_items(info)
        for record in records:
            record.pop("judge", None)
            record.pop("result", None)
        judge.judge_all(items_by_id, [r for r in records if r["status"] != "error"])
        summary = scoring.score_all(items_by_id, records)
        info = info | {"name": f"{info['name']}_rejudged", "rejudged_from": str(src), "rejudge_overrides": overrides,
                       "judge_repeats": runner.settings.judge_repeats, "settings": report.public_settings(), "gold": gold_info,
                       "models": info["models"] | {"judge": runner.settings.judge_model, "judge_template": judge.TEMPLATE_VERSION}}
    path = report.save(info["name"], info, records, summary)
    report.print_summary(summary, path)
    return 0


def cmd_generate(args) -> int:
    build = generate.generate(args.docs, Path(args.out), name=args.name, n_quant=args.quant, n_qual=args.qual,
                              n_traps=args.traps, n_out=args.out_of_corpus, seed=args.seed)
    counts = build["counts"]
    report.console.print(f"초안 정량 {counts['quant']} · 정성 {counts['qual']} · 답 없음·함정 {counts['neg']} (탈락 {len(build['rejects'])}건)"
                         f" -> {args.out}/review.md 를 검토하고 쓸 문항을 approved로 바꾼다")
    return 0


def _item_scores(records: list[dict]) -> dict[str, float]:
    """문항별 평균 점수 (반복 실행 평균, 오류 제외)."""
    acc: dict[str, list[int]] = {}
    for r in records:
        if r["result"]["score"] is not None:
            acc.setdefault(r["id"], []).append(r["result"]["score"])
    return {k: sum(v) / len(v) for k, v in acc.items()}


def cmd_compare(args) -> int:
    (info_a, rec_a, sum_a), (info_b, rec_b, sum_b) = _load_run(Path(args.a)), _load_run(Path(args.b))
    lines = [f"# 비교: {info_a['name']} -> {info_b['name']}", ""]
    if info_a["gold"]["hash"] != info_b["gold"]["hash"]:
        lines += ["**경고: 두 실행의 골드셋이 다르다. 비교가 성립하지 않을 수 있다.**", ""]
    label = lambda info: (info["overrides"] | info.get("rescore_overrides", {}) | info.get("rejudge_overrides", {})) or "없음"  # noqa: E731
    lines += [f"- 덮어쓴 설정: {label(info_a)} -> {label(info_b)}",
              f"- index: {info_a['index_version']} -> {info_b['index_version']}", "",
              "| 묶음 | 안전성 | 유용성 |", "|---|---|---|"]
    for name in scoring.GROUPS + ["전체"]:
        ga, gb = sum_a["groups"][name], sum_b["groups"][name]
        cells = [f"{report._pct(ga[m])} -> {report._pct(gb[m])} ({(gb[m] - ga[m]) * 100:+.1f}%p)"
                 if ga[m] is not None and gb[m] is not None else "-" for m in ("safety", "usefulness")]
        lines.append(f"| {name} | {cells[0]} | {cells[1]} |")
    lines += ["", "| 진단 지표 | A | B |", "|---|---|---|"]
    for key, (label, fmt) in report.DIAG_LABELS.items():
        lines.append(f"| {label} | {fmt(sum_a['diagnostics'][key])} | {fmt(sum_b['diagnostics'][key])} |")
    sa, sb = _item_scores(rec_a), _item_scores(rec_b)
    better = sorted(k for k in sa.keys() & sb.keys() if sb[k] > sa[k])
    worse = sorted(k for k in sa.keys() & sb.keys() if sb[k] < sa[k])
    lines += ["", f"좋아진 문항 {len(better)}: " + (", ".join(f"{k}({sa[k]:g}->{sb[k]:g})" for k in better) or "없음"),
              f"나빠진 문항 {len(worse)}: " + (", ".join(f"{k}({sa[k]:g}->{sb[k]:g})" for k in worse) or "없음")]
    text = "\n".join(lines) + "\n"
    print(text)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    return 0


def cmd_label_sheet(args) -> int:
    path = labels.make_sheet(Path(args.run), args.repeat, args.support)
    report.console.print(f"라벨링 시트: {path}")
    return 0


def cmd_calibrate(args) -> int:
    result = labels.calibrate(Path(args.sheet))
    for kind, m in result["kinds"].items():
        report.console.print(f"{kind}: {m}")
    report.console.print(f"애매(?)로 제외 {result['skipped_uncertain']}건 · 저장: {Path(args.sheet).with_suffix('.calibration.json')}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m eval", description="RAG 평가 하네스")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="골드셋 전체 평가 (실행 -> judge -> 채점 -> 리포트)")
    run.add_argument("--gold", default="eval/gold")
    run.add_argument("--name")
    run.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="설정 덮어쓰기 (여러 번 가능)")
    run.add_argument("--exp", help="eval/experiments/<이름>.toml의 [settings]")
    run.add_argument("--repeats", type=int, default=1, help="같은 골드셋을 몇 번 돌릴지 (생성 비결정성 측정)")
    run.add_argument("--limit", type=int, help="앞에서 N문항만 (빠른 확인용)")
    rescore = sub.add_parser("rescore", help="저장된 실행을 다시 채점 (문턱 보정 등)")
    rescore.add_argument("run")
    rescore.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    compare = sub.add_parser("compare", help="두 실행 비교")
    compare.add_argument("a")
    compare.add_argument("b")
    compare.add_argument("--out", help="비교 결과를 Markdown 파일로도 저장")
    sheet = sub.add_parser("label-sheet", help="사람 라벨링 시트 만들기 (judge 일치도·문턱 보정용)")
    sheet.add_argument("run")
    sheet.add_argument("--repeat", type=int, default=0)
    sheet.add_argument("--support", type=int, default=30, help="문장 근거 표본 수")
    rejudge = sub.add_parser("rejudge", help="저장된 답변으로 judge만 다시 (답변 고정)")
    rejudge.add_argument("run")
    rejudge.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    gen = sub.add_parser("generate", help="문서에서 골드셋 초안 생성 (모든 문항 draft, 사람 검토 후 사용)")
    gen.add_argument("docs", nargs="+", help="문서 파일 (PDF/MD/TXT)")
    gen.add_argument("--out", required=True, help="새 골드셋 디렉터리 (이미 있으면 거부)")
    gen.add_argument("--name", help="골드셋 이름 (기본: 디렉터리 이름)")
    gen.add_argument("--quant", type=int, default=10, help="정량 사실 수 (문항은 literal·user 2배)")
    gen.add_argument("--qual", type=int, default=5)
    gen.add_argument("--traps", type=int, default=5, help="잘못된 전제 문항 수")
    gen.add_argument("--out-of-corpus", type=int, default=5)
    gen.add_argument("--seed", type=int, default=7)
    calib = sub.add_parser("calibrate", help="채운 라벨링 시트로 judge 일치도와 추천 문턱 계산")
    calib.add_argument("sheet")
    args = parser.parse_args(argv)
    commands = {"run": cmd_run, "rescore": cmd_rescore, "compare": cmd_compare, "label-sheet": cmd_label_sheet,
                "calibrate": cmd_calibrate, "rejudge": cmd_rejudge, "generate": cmd_generate}
    return commands[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
