"""PoC: Jev(System One)를 한국어 RAG 답변 judge로 쓸 수 있는가.

    uv run python -m eval.poc.jev_poc collect   # 기준선 RAG 답변 수집 -> eval/poc/answers.json
    uv run python -m eval.poc.jev_poc judge     # 판정 실행 -> eval/poc/result.json

판정 세 종류를 정답(truth)이 분명한 사례로 잰다.
- value: 정량 답변이 정답 값을 말하는가 (정답 값 = 참, 숫자를 바꾼 값 = 거짓)
- support: 답변 문장이 출처로 뒷받침되는가 (원문장 = 참, 숫자를 바꾼 문장 = 거짓)
- point: 정성 답변이 루브릭 포인트를 담았는가 (진실값은 labels.json, 다른 문항의 포인트 = 거짓)
지시문은 영어(EN)와 한국어(KO) 두 가지로 같은 사례를 돌려 비교하고, 같은 요청을 두 번 보내 흔들림을 본다.
"""

import json
import os
import re
import sys
import time
import urllib.request
from pathlib import Path

from app import rag

POC = Path("eval/poc")
GOLD = Path("eval/gold")
JEV_URL = "https://api.typesafe.ai/v1/systemone"
JEV_MODEL = "jev-1.13.0"
QUAL_IDS = ["qual-02", "qual-05", "qual-11"]
CONTROL_IDS = ["qual-13", "qual-16"]  # 다른 문항 포인트를 섞어 넣은 대조군 (진실값 거짓)

TEMPLATES = {
    "EN": {
        "value": 'Does the "answer" explicitly state the following value as the answer to the "question"? '
                 'Value: "{target}". The same value written differently (e.g. 5천만원 = 5,000만 원) counts; a different number does not.',
        "support": 'Look only at the sentence with id "{sid}", read as an answer to the "question". '
                   'Is it fully supported by the "sources"? Answer no if any number, date or condition in it differs from the sources.',
        "point": 'Does the "answer" convey the following point, possibly in different words? Point: "{target}". '
                 "Answer yes only if every key fact of the point (amounts, dates, eligibility) appears in the answer.",
    },
    "KO": {
        "value": '"answer"가 "question"에 대한 답으로 다음 값을 분명히 말하는가? 값: "{target}". '
                 "표기만 다른 같은 값(예: 5천만원 = 5,000만 원)은 같은 값으로 본다. 다른 숫자는 아니다.",
        "support": 'id가 "{sid}"인 문장만 "question"에 대한 답으로 읽어라. 그 문장이 "sources"로 완전히 뒷받침되는가? '
                   "숫자, 날짜, 조건이 하나라도 출처와 다르면 아니다.",
        "point": '"answer"가 다음 포인트를 (표현은 달라도) 담고 있는가? 포인트: "{target}". '
                 "포인트의 핵심 사실(금액, 날짜, 대상 요건)이 모두 답변에 있을 때만 그렇다.",
    },
}


def jev(state: dict, questions: dict[str, str]) -> tuple[dict[str, float], int, float]:
    payload = {"model": JEV_MODEL, "state": state,
               "questions": {k: {"type": "noul", "instructions": v} for k, v in questions.items()}}
    req = urllib.request.Request(JEV_URL, json.dumps(payload, ensure_ascii=False).encode(),
                                 {"Authorization": f"Bearer {os.environ['TYPESAFE_API_KEY']}",
                                  "Content-Type": "application/json"})
    started = time.perf_counter()
    with urllib.request.urlopen(req, timeout=60) as resp:
        body = json.loads(resp.read())
    probs = {k: v["noul"] for k, v in body["answers"].items()}
    return probs, body["usage"]["input_tokens"], time.perf_counter() - started


def load(name: str) -> list[dict]:
    return [json.loads(line) for line in (GOLD / name).open(encoding="utf-8")]


def collect() -> None:
    quant = [q for q in load("quant.jsonl") if q["variant"] == "literal"]
    qual = {q["id"]: q for q in load("qual.jsonl")}
    out = []
    for item in quant + [qual[i] for i in QUAL_IDS]:
        result = rag.answer(item["question"])
        out.append({"id": item["id"], "question": item["question"], "status": result.status,
                    "answer": result.answer, "sources": [h.content for h in result.retrieved],
                    "cited_refs": [c["ref"] for c in result.citations]})
        print(item["id"], result.status, "|", result.answer[:80].replace("\n", " "))
    (POC / "answers.json").write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def perturb(value: str, fallback: str = "") -> str:
    """값의 첫 숫자를 바꾼다 (진실값이 확실한 거짓 사례 만들기). 숫자가 없으면 다른 문항의 정답으로 바꾼다."""
    m = re.search(r"\d+", value)
    if not m:
        return fallback
    n = int(m.group())
    return value[: m.start()] + str(n * 3 if n < 10 else n + n // 2) + value[m.end():]


def strip_ref(point: str) -> str:
    return re.sub(r"\s*\(PDF p\..*?\)\s*$", "", point).strip()


def build_cases() -> list[dict]:
    """(state, 질문 id -> (종류, 대상, 진실값)) 묶음 목록. 요청 하나 = state 하나."""
    answers = {a["id"]: a for a in json.loads((POC / "answers.json").read_text())}
    labels = json.loads((POC / "labels.json").read_text())
    quant = {q["id"]: q for q in load("quant.jsonl")}
    qual = {q["id"]: q for q in load("qual.jsonl")}
    cases = []
    quant_ids = [k for k in answers if k.startswith("quant")]
    for qid, a in answers.items():
        if qid.startswith("quant"):
            gold = quant[qid]["answer"]
            other = quant[quant_ids[(quant_ids.index(qid) + 1) % len(quant_ids)]]["answer"]
            lab = labels["value"][qid]
            checks = {"v_neg": ("value", perturb(gold, other), False)}
            if lab["states_gold"] is not None:  # None = 사람이 봐도 애매해 제외
                checks["v_pos"] = ("value", gold, lab["states_gold"])
            cases.append({"id": qid, "state": {"question": a["question"], "answer": a["answer"]}, "checks": checks})
            if lab.get("sentence"):  # 정답 값을 담은 문장과, 그 숫자를 바꾼 문장
                sentence = lab["sentence"]
                fake = sentence.replace(lab["value_in_sentence"], perturb(lab["value_in_sentence"]))
                state = {"question": a["question"], "sources": a["sources"][:3],
                         "sentences": [{"id": "a", "text": sentence}, {"id": "b", "text": fake}]}
                checks = {"s_a": ("support", "a", lab["supported"]), "s_b": ("support", "b", False)}
                cases.append({"id": qid + ":support", "state": state, "checks": checks})
        else:
            points = [strip_ref(p["point"]) for p in qual[qid]["must_include"]]
            controls = [strip_ref(qual[c]["must_include"][0]["point"]) for c in CONTROL_IDS]
            checks = {f"p{i}": ("point", p, labels["point"][qid][i]) for i, p in enumerate(points)}
            checks |= {f"c{i}": ("point", p, False) for i, p in enumerate(controls)}
            cases.append({"id": qid, "state": {"question": a["question"], "answer": a["answer"]}, "checks": checks})
    return cases


def judge(repeats: int = 2) -> None:
    cases = build_cases()
    rows, tokens, latency = [], 0, []
    for lang, tpl in TEMPLATES.items():
        for run in range(repeats):
            for case in cases:
                questions = {
                    cid: tpl[kind].format(target=target, sid=target)
                    for cid, (kind, target, _) in case["checks"].items()
                }
                probs, used, secs = jev(case["state"], questions)
                tokens += used
                latency.append(secs)
                for cid, (kind, target, truth) in case["checks"].items():
                    rows.append({"lang": lang, "run": run, "case": case["id"], "check": cid, "kind": kind,
                                 "target": target, "truth": truth, "p": probs[cid]})

    summary = {}
    for lang in TEMPLATES:
        for kind in ("value", "support", "point", "all"):
            sel = [r for r in rows if r["lang"] == lang and r["run"] == 0 and (kind == "all" or r["kind"] == kind)]
            if not sel:
                continue
            pos = [r["p"] for r in sel if r["truth"]]
            neg = [r["p"] for r in sel if not r["truth"]]
            pairs = [(p, n) for p in pos for n in neg]
            summary[f"{lang}/{kind}"] = {
                "n": len(sel), "pos": len(pos), "neg": len(neg),
                "accuracy@0.5": round(sum((r["p"] >= 0.5) == r["truth"] for r in sel) / len(sel), 3),
                "mean_p_pos": round(sum(pos) / len(pos), 3) if pos else None,
                "mean_p_neg": round(sum(neg) / len(neg), 3) if neg else None,
                "auc": round(sum((p > n) + 0.5 * (p == n) for p, n in pairs) / len(pairs), 3) if pairs else None,
            }
        r0 = {(r["case"], r["check"]): r["p"] for r in rows if r["lang"] == lang and r["run"] == 0}
        r1 = {(r["case"], r["check"]): r["p"] for r in rows if r["lang"] == lang and r["run"] == 1}
        summary[f"{lang}/repeat"] = {
            "max_abs_diff": round(max(abs(r0[k] - r1[k]) for k in r0), 3),
            "decision_flips@0.5": sum((r0[k] >= 0.5) != (r1[k] >= 0.5) for k in r0),
        }
    errors = [r for r in rows if r["run"] == 0 and (r["p"] >= 0.5) != r["truth"]]
    result = {"model": JEV_MODEL, "summary": summary, "input_tokens": tokens,
              "cost_usd": round(tokens * 0.042 / 1e6, 5), "mean_latency_s": round(sum(latency) / len(latency), 3),
              "errors": errors, "rows": rows}
    (POC / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for k, v in summary.items():
        print(f"{k:12} {v}")
    print(f"input_tokens={tokens} cost=${result['cost_usd']} mean_latency={result['mean_latency_s']}s errors(run0)={len(errors)}")
    for e in errors:
        print(f"  오판 {e['lang']} {e['case']} {e['check']} truth={e['truth']} p={e['p']} | {str(e['target'])[:70]}")


if __name__ == "__main__":
    {"collect": collect, "judge": judge}[sys.argv[1]]()
