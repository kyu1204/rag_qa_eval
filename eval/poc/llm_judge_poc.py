"""PoC: Jev 대신 일반 LLM(Gemini 3.5 Flash-Lite, Elice ML API)을 judge로 쓸 수 있는가.

    uv run python -m eval.poc.llm_judge_poc

사람이 라벨링한 기준선 답변(eval/labels/20261001-173916_baseline_r0.md)의 판정 149건을
Jev와 같은 판정 문구(eval/judge.py의 TEMPLATES)로 LLM에게 묻고, 사람 라벨과의 일치도를 Jev와 나란히 잰다.
- 이 엔드포인트는 토큰 확률(logprobs)과 seed를 400으로 거부한다 (2026-10-01 실측).
  그래서 같은 판정을 VOTES번 묻고 '예' 비율을 확률로 쓴다 (문턱 0.5 = 다수결).
- 일치도 계산은 하네스의 calibrate를 그대로 쓴다 (판정 확률만 LLM 것으로 바꾼 임시 실행 기록을 만들어서).
- 결과: eval/poc/llm_judge_result.json
"""

import json
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from openai import OpenAI

from app.config import settings
from eval import judge, labels
from eval.goldset import load

SHEET = Path("eval/labels/20261001-173916_baseline_r0.md")
BASE_URL = "https://mlapi.run/3fc54e02-bf9b-483e-b0d8-571ff86f04af/v1"  # Elice ML API, 키는 ELICE_API_KEY
MODEL = "gemini-3.5-flash-lite"
VOTES = 3
OUT = Path("eval/poc/llm_judge_result.json")
PROMPT = """너는 RAG 답변을 채점하는 judge다. 아래 [자료]의 question, answer, sources, sentences를 읽고 [판정]마다 답하라.
- type이 noul인 판정은 instructions의 물음에 "예" 또는 "아니오"로 답한다.
- type이 choice인 판정은 criteria의 키("2", "1", "0") 중 answer에 가장 잘 맞는 하나로 답한다.
JSON 객체 하나만 출력한다: {{"<판정 id>": "<답>", ...}}

[자료]
{state}

[판정]
{questions}"""


def ask(api, state: dict, asked: dict) -> tuple[dict, int]:
    questions = [{"id": cid, **judge._question(kind, target)} for cid, (kind, target) in asked.items()]
    resp = api.chat.completions.create(
        model=MODEL, response_format={"type": "json_object"},
        messages=[{"role": "user", "content": PROMPT.format(state=json.dumps(state, ensure_ascii=False),
                                                            questions=json.dumps(questions, ensure_ascii=False))}])
    return json.loads(resp.choices[0].message.content), resp.usage.prompt_tokens + resp.usage.completion_tokens


def vote(answer) -> float | None:
    text = str(answer).strip().lower()
    return 1.0 if text in ("예", "yes", "true") else 0.0 if text in ("아니오", "아니요", "no", "false") else None


def main() -> None:
    run_dir, sheet_labels = labels.read_sheet(SHEET)
    gold = {item["id"]: item for item in load("eval/gold").items}
    records = [json.loads(line) for line in (Path(run_dir) / "items.jsonl").open(encoding="utf-8")]
    by_id = {r["id"]: r for r in records if r["repeat"] == 0}
    wanted: dict[str, list[str]] = {}
    for key in sheet_labels:
        item_id, cid = key.split("#")
        wanted.setdefault(item_id, []).append(cid)

    jobs = []  # (문항 id, state, 물을 판정) x VOTES
    for item_id, cids in wanted.items():
        state, checks = judge.build_checks(gold[item_id], by_id[item_id])
        jobs += [(item_id, state, {cid: checks[cid] for cid in cids})] * VOTES
    api, started = OpenAI(base_url=BASE_URL, api_key=settings.elice_api_key), time.time()
    with ThreadPoolExecutor(8) as pool:
        answers = list(pool.map(lambda job: ask(api, job[1], job[2]), jobs))
    elapsed = time.time() - started

    # 판정마다 표 모으기 -> 확률(예 비율) / 기준은 다수결
    raw: dict[tuple[str, str], list] = {}
    kinds: dict[tuple[str, str], str] = {}
    for (item_id, _, asked), (reply, _) in zip(jobs, answers, strict=True):
        for cid, (kind, _) in asked.items():
            raw.setdefault((item_id, cid), []).append(reply.get(cid))
            kinds[(item_id, cid)] = kind
    jev_checks = {(item_id, cid): by_id[item_id]["judge"]["checks"][cid] for item_id, cid in raw}
    invalid = 0
    for (item_id, cid), got in raw.items():
        kind = kinds[(item_id, cid)]
        if kind == "grade":
            choices = [str(g) for g in got if str(g) in ("2", "1", "0")]
            invalid += len(got) - len(choices)
            check = {"kind": kind, "choice": max(set(choices), key=choices.count) if choices else None, "choices": choices}
        else:
            ps = [v for v in map(vote, got) if v is not None]
            invalid += len(got) - len(ps)
            check = {"kind": kind, "p": sum(ps) / len(ps) if ps else 0.0, "ps": ps}
        by_id[item_id]["judge"]["checks"][cid] = check  # 이 판정만 LLM 것으로 바꾼다

    # 하네스 calibrate를 그대로 쓰도록 임시 실행 기록과 시트 사본을 만든다 (문턱 0.5 = 다수결)
    tmp = Path(tempfile.mkdtemp())
    (tmp / "items.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")
    sheet = tmp / SHEET.name
    sheet.write_text(SHEET.read_text(encoding="utf-8").replace(f"`{run_dir}`", f"`{tmp}`"), encoding="utf-8")
    for tau in ("tau_value", "tau_fact", "tau_must_not", "tau_support"):
        setattr(settings, tau, 0.5)
    llm = labels.calibrate(sheet)["kinds"]
    jev = json.loads(SHEET.with_suffix(".calibration.json").read_text(encoding="utf-8"))["kinds"]

    split = [len(set(v for v in map(vote, got) if v is not None)) > 1 for (_, cid), got in raw.items() if cid != "grade"]
    result = {
        "model": MODEL, "votes": VOTES, "checks": len(raw), "calls": len(jobs), "invalid_answers": invalid,
        "seconds": round(elapsed, 1), "tokens": sum(t for _, t in answers),
        "vote_split_rate": round(sum(split) / len(split), 3),
        "agreement": {kind: {"llm": llm.get(kind), "jev": jev.get(kind)} for kind in jev},
        "rows": [{"key": f"{item_id}#{cid}", "kind": kinds[(item_id, cid)], "human": sheet_labels[f"{item_id}#{cid}"][0],
                  "jev": jev_checks[(item_id, cid)].get("p", jev_checks[(item_id, cid)].get("choice")), "llm_votes": got}
                 for (item_id, cid), got in raw.items()],
    }
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"판정 {result['checks']}건 x {VOTES}표, 호출 {len(jobs)}회, {elapsed:.0f}초, 토큰 {result['tokens']:,}, "
          f"형식 오류 {invalid}, 표가 갈린 판정 {result['vote_split_rate'] * 100:.1f}%")
    for kind, pair in result["agreement"].items():
        line = [kind.ljust(9)]
        for name, k in pair.items():
            if not k:
                continue
            if kind == "grade":
                line.append(f"{name} 일치 {k['agreement']:.0%}")
            else:
                line.append(f"{name} 일치 {k['accuracy']:.1%} (문턱 {k['tau']}) 최적 {k['best_tau']} {k['best_tau_accuracy']:.1%}"
                            f" 교차검증 {k['cv_accuracy']:.1%} κ {k['kappa']}")
        print(" | ".join(line))


if __name__ == "__main__":
    main()
