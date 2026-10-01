"""사람 라벨링 시트: judge가 내린 판정을 사람이 같은 기준으로 다시 판정해 일치도와 문턱을 잰다.

    uv run python -m eval label-sheet <run 디렉터리> [--repeat 0] [--support 30]

시트에는 judge 확률을 싣지 않는다 (사람 판정이 judge 쪽으로 끌려가지 않게).
각 표의 마지막 두 칸이 "판정 | 메모"이고, 판정은 O(예)·X(아니오)·?(애매), 함정 기준은 2/1/0.
키(qual-02#f0_0)로 run의 items.jsonl과 짝을 지어 calibrate가 읽는다.
"""

import json
import random
import re
from pathlib import Path

from eval.judge import split_sentences, strip_explanation
from eval.goldset import load

LABELS = Path("eval/labels")
GUIDE = """- 대상 실행: `{run}` (반복 번호 {repeat}의 답변)
- 각 표의 **판정** 칸에 `O`(예) 또는 `X`(아니오)를 적는다. 판단이 정말 어려우면 `?`와 메모. 함정 기준은 `2`/`1`/`0`.
- judge 확률은 일부러 싣지 않았다. 다른 칸은 고치지 않는다 (키로 결과와 짝을 짓는다).
- 판정 기준 (judge 지시문과 같은 기준):
  - **사실**: 답변에 이 사실이 표현은 달라도 들어 있으면 O. 사실의 일부만 있으면 X.
  - **잘못된 내용**: 답변이 이 잘못된 내용을 사실처럼 말하면 O. (맞는 내용을 말했으면 X)
  - **정답 값**: 답변이 이 값을 질문의 답으로 말하면 O. 표기만 다른 같은 값은 O.
  - **함정 기준**: 답변에 가장 잘 맞는 기준 번호.
  - **문장 근거**: 이 문장의 사실 주장이 출처로 완전히 뒷받침되면 O. 숫자·날짜·조건이 하나라도 다르면 X. 사실 주장이 없는 안내 문장은 O.
"""


def _cell(text: str) -> str:
    return " ".join(str(text).split()).replace("|", "\\|")


def _quote(text: str) -> list[str]:
    return [f"> {line}" if line.strip() else ">" for line in text.splitlines()]


def make_sheet(run: Path, repeat: int = 0, support_n: int = 30, seed: int = 7) -> Path:
    info = json.loads((run / "run.json").read_text())
    gold = {item["id"]: item for item in load(info["gold"]["path"]).items}
    records = [r for r in map(json.loads, (run / "items.jsonl").open(encoding="utf-8"))
               if r["repeat"] == repeat and r["status"] == "answered"]
    lines = [f"# 사람 라벨링 시트: {info['name']}", "", GUIDE.format(run=run, repeat=repeat), ""]
    counts = {}

    qual = [r for r in records if r["type"] == "qual"]
    lines += ["## 1. 정성 답변: 사실 포함 여부와 잘못된 내용", ""]
    for r in qual:
        item = gold[r["id"]]
        lines += [f"### {r['id']}", "", f"**질문** {r['question']}", "", "**답변**", "", *_quote(r["answer"]), "",
                  "| 키 | 종류 | 확인할 내용 | 판정 | 메모 |", "|---|---|---|---|---|"]
        for pi, point in enumerate(item["must_include"]):
            for fi, fact in enumerate(point["facts"]):
                lines.append(f"| {r['id']}#f{pi}_{fi} | 사실 | {_cell(fact)} |  |  |")
                counts["사실"] = counts.get("사실", 0) + 1
        for mi, wrong in enumerate(item.get("must_not", [])):
            lines.append(f"| {r['id']}#m{mi} | 잘못된 내용 | {_cell(strip_explanation(wrong))} |  |  |")
            counts["잘못된 내용"] = counts.get("잘못된 내용", 0) + 1
        lines.append("")

    quant = [r for r in records if r["type"] == "quant"]
    lines += ["## 2. 정량 답변: 정답 값", "", "| 키 | 질문 | 답변 | 정답 값 | 판정 | 메모 |", "|---|---|---|---|---|---|"]
    for r in quant:
        lines.append(f"| {r['id']}#value | {_cell(r['question'])} | {_cell(r['answer'])} | {_cell(gold[r['id']]['answer'])} |  |  |")
    counts["정답 값"] = len(quant)
    lines.append("")

    traps = [r for r in records if r["group"] == "답 없음·함정"]
    lines += ["## 3. 답 없음·함정: 기준 번호", ""]
    for r in traps:
        criteria = gold[r["id"]]["scoring"]
        lines += [f"### {r['id']}", "", f"**질문** {r['question']}", "", "**답변**", "", *_quote(r["answer"]), "",
                  *[f"- **{k}**: {criteria[k]}" for k in ("2", "1", "0") if k in criteria], "",
                  "| 키 | 판정 | 메모 |", "|---|---|---|", f"| {r['id']}#grade |  |  |", ""]
    counts["함정 기준"] = len(traps)

    # 문장 근거: 확률이 낮은 문장은 모두, 나머지는 무작위로 채운다 (judge의 관대함도 보려고)
    sentences = []
    for r in records:
        texts = split_sentences(r["answer"])
        for cid, check in r["judge"]["checks"].items():
            if check["kind"] == "support":
                sentences.append((check["p"], r, cid, texts[int(cid[1:])]))
    low = [s for s in sentences if s[0] < 0.9]
    rest = [s for s in sentences if s[0] >= 0.9]
    picked = low + random.Random(seed).sample(rest, max(0, min(len(rest), support_n - len(low))))
    picked.sort(key=lambda s: (s[1]["id"], int(s[2][1:])))
    lines += ["## 4. 문장 근거 (표본)", "", "답변 맥락에서 문장을 읽고, 출처를 펼쳐 확인한다.", ""]
    for rid in dict.fromkeys(s[1]["id"] for s in picked):
        group = [s for s in picked if s[1]["id"] == rid]
        r = group[0][1]
        lines += [f"### {rid}", "", f"**질문** {r['question']}", "", "**답변**", "", *_quote(r["answer"]), "",
                  "<details><summary>출처 펼치기</summary>", ""]
        for h in r["retrieved"]:
            lines += [f"**[{h['ref']}] PDF p.{h['page_start']}-{h['page_end']}**", "", *_quote(h["content"]), ""]
        lines += ["</details>", "", "| 키 | 문장 | 판정 | 메모 |", "|---|---|---|---|"]
        lines += [f"| {rid}#{cid} | {_cell(text)} |  |  |" for _, _, cid, text in group]
        lines.append("")
    counts["문장 근거"] = len(picked)

    total = sum(counts.values())
    summary = " · ".join(f"{k} {v}" for k, v in counts.items())
    lines.insert(2, f"총 {total}건: {summary}\n")
    LABELS.mkdir(parents=True, exist_ok=True)
    out = LABELS / f"{run.name}_r{repeat}.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def read_sheet(path: Path) -> tuple[str, dict[str, tuple[str, str]]]:
    """(run 디렉터리, {키: (판정, 메모)}). 판정이 빈 행은 건너뛴다."""
    text = path.read_text(encoding="utf-8")
    run = re.search(r"대상 실행: `([^`]+)`", text).group(1)
    labels = {}
    for line in text.splitlines():
        if not re.match(r"^\|\s*[\w-]+#\w+\s*\|", line):
            continue
        cells = [c.strip() for c in re.split(r"(?<!\\)\|", line)[1:-1]]
        verdict, memo = cells[-2].upper(), cells[-1]
        if verdict:
            labels[cells[0]] = (verdict, memo)
    return run, labels


KIND_BY_PREFIX = {"f": "fact", "m": "must_not", "s": "support"}
TAU = {"fact": "tau_fact", "must_not": "tau_must_not", "value": "tau_value", "support": "tau_support"}


def _kind(check_id: str) -> str:
    return check_id if check_id in ("value", "grade") else KIND_BY_PREFIX[check_id[0]]


def _kappa(pairs: list[tuple[bool, bool]]) -> float | None:
    n = len(pairs)
    if not n:
        return None
    po = sum(a == b for a, b in pairs) / n
    pa, pb = sum(a for a, _ in pairs) / n, sum(b for _, b in pairs) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return round((po - pe) / (1 - pe), 3) if pe < 1 else None


def _best_tau(rows: list[tuple[float, bool]]) -> float:
    """사람 판정과 가장 많이 맞는 문턱 (후보 = 확률 사이 중간값). 동률이면 0.5에 가까운 값."""
    ps = sorted({p for p, _ in rows} | {0.0, 1.0})
    candidates = [(a + b) / 2 for a, b in zip(ps, ps[1:])] or [0.5]
    return max(candidates, key=lambda t: (sum((p >= t) == h for p, h in rows), -abs(t - 0.5)))


def calibrate(sheet: Path) -> dict:
    """사람 라벨과 judge 판정의 일치도, 지금 문턱의 혼동행렬, 추천 문턱(문항 단위 교차검증 포함)."""
    from app.config import settings

    run, labels = read_sheet(sheet)
    records = {(r["id"], r["repeat"]): r for r in map(json.loads, (Path(run) / "items.jsonl").open(encoding="utf-8"))}
    repeat = int(re.search(r"_r(\d+)", sheet.stem).group(1))
    rows: dict[str, list[tuple[str, float | str, bool | str]]] = {}
    skipped = 0
    for key, (verdict, _) in labels.items():
        item_id, check_id = key.split("#")
        check = records[(item_id, repeat)]["judge"]["checks"][check_id]
        kind = _kind(check_id)
        if kind == "grade":
            rows.setdefault(kind, []).append((item_id, check["choice"], verdict))
        elif verdict in ("O", "X"):
            rows.setdefault(kind, []).append((item_id, check["p"], verdict == "O"))
        else:
            skipped += 1
    out = {"sheet": str(sheet), "run": run, "skipped_uncertain": skipped, "kinds": {}}
    for kind, data in rows.items():
        if kind == "grade":
            out["kinds"][kind] = {"n": len(data), "agreement": round(sum(j == h for _, j, h in data) / len(data), 3)}
            continue
        tau = getattr(settings, TAU[kind])
        pairs = [(p >= tau, h) for _, p, h in data]
        tp = sum(j and h for j, h in pairs)
        fp = sum(j and not h for j, h in pairs)
        fn = sum(h and not j for j, h in pairs)
        # 문항 단위 교차검증: 한 문항을 빼고 정한 문턱으로 그 문항을 맞히는 비율 (같은 데이터로 정하고 재는 낙관 편향 보정)
        items = sorted({i for i, _, _ in data})
        cv_hits = 0
        for held in items:
            train = [(p, h) for i, p, h in data if i != held]
            t = _best_tau(train) if train else tau
            cv_hits += sum((p >= t) == h for i, p, h in data if i == held)
        best = _best_tau([(p, h) for _, p, h in data])
        out["kinds"][kind] = {
            "n": len(data), "human_yes": sum(h for _, _, h in data), "tau": tau,
            "accuracy": round(sum(j == h for j, h in pairs) / len(pairs), 3), "kappa": _kappa(pairs),
            "confusion": {"judge_yes_human_yes": tp, "judge_yes_human_no": fp, "judge_no_human_yes": fn,
                          "judge_no_human_no": len(pairs) - tp - fp - fn},
            "best_tau": round(best, 3),
            "best_tau_accuracy": round(sum((p >= best) == h for _, p, h in data) / len(data), 3),
            "cv_accuracy": round(cv_hits / len(data), 3),
        }
    path = sheet.with_suffix(".calibration.json")
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return out
