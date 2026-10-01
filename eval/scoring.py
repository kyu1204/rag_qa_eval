"""판정 확률 -> 문항 점수(0/1/2)·실패 원인 -> 묶음별 지표. 전부 결정적 규칙이라 문턱만 바꿔 다시 채점할 수 있다.

점수 원칙: 0 = 틀림·환각(사용자에게 해롭다), 1 = 안전하지만 불완전(거절·일부·인용 오류), 2 = 정확하고 유용.
근거 없는 답변 문장은 묶음과 관계없이 환각으로 본다.

실패 원인 (튜닝 방향): 환각 / 오거절 / 검색 실패(근거 쪽이 top-k에 없음) / 생성 누락(근거는 검색됐는데 사실이 빠짐) / 인용 오류.
"""

import statistics

from app.config import settings
from eval.judge import split_sentences

GROUPS = ["정량", "정성", "답 없음·함정"]


def _covers(chunk: dict, doc: str, page: int | None, quote: str | None) -> bool:
    if chunk["document_id"] != doc:
        return False
    if page is not None and chunk.get("page_start") is not None:
        return chunk["page_start"] <= page <= (chunk.get("page_end") or chunk["page_start"])
    return bool(quote) and "".join(quote.split()) in "".join(chunk.get("content", "").split())


def evidence_retrieved(record: dict, evidence: list[dict]) -> bool:
    return any(_covers(c, e["document_id"], e.get("page"), e.get("quote")) for c in record.get("retrieved", []) for e in evidence)


def first_hit_rank(record: dict, evidence: list[dict]) -> int | None:
    for rank, chunk in enumerate(record.get("retrieved", []), 1):
        if any(_covers(chunk, e["document_id"], e.get("page"), e.get("quote")) for e in evidence):
            return rank
    return None


def item_evidence(item: dict) -> list[dict]:
    if item["type"] == "qual":
        return [e for p in item["must_include"] for e in p["evidence"]]
    return item.get("evidence", [])


def score_record(item: dict, record: dict) -> dict:
    """{"score", "cause", "detail"}. score None = 실행 오류."""
    if record["status"] == "error":
        return {"score": None, "cause": "실행 오류", "detail": {}}
    checks = record.get("judge", {}).get("checks", {})
    sentences = split_sentences(record.get("answer", ""))
    unsupported = [sentences[int(cid[1:])] for cid, c in checks.items()
                   if c["kind"] == "support" and c["p"] < settings.tau_support]
    evidence = item_evidence(item)
    hit = evidence_retrieved(record, evidence) if evidence else None
    detail = {"evidence_retrieved": hit, "first_hit_rank": first_hit_rank(record, evidence) if evidence else None,
              "unsupported_sentences": unsupported,
              "sentences_judged": sum(c["kind"] == "support" for c in checks.values())}
    refused = record["status"] == "insufficient_context"

    def refusal(score: int) -> dict:
        why = "검색 실패" if hit is False else "생성 판단"
        return {"score": score, "cause": "오거절", "detail": detail | {"refusal_reason": why}}

    if item["type"] == "quant":
        if refused:
            return refusal(1)
        value_p = checks["value"]["p"]
        cited = any(_covers(c, e["document_id"], e.get("page"), e.get("quote"))
                    for c in record["citations"] for e in evidence)
        detail |= {"value_p": value_p, "value_ok": value_p >= settings.tau_value, "cited_evidence": cited}
        if not detail["value_ok"] or unsupported:
            return {"score": 0, "cause": "환각", "detail": detail}
        return {"score": 2, "cause": None, "detail": detail} if cited else {"score": 1, "cause": "인용 오류", "detail": detail}

    if item["type"] == "qual":
        if refused:
            return refusal(1)
        facts = []
        for pi, point in enumerate(item["must_include"]):
            point_hit = evidence_retrieved(record, point["evidence"])
            for fi, fact in enumerate(point["facts"]):
                p = checks[f"f{pi}_{fi}"]["p"]
                facts.append({"fact": fact, "p": p, "covered": p >= settings.tau_fact, "evidence_retrieved": point_hit})
        violations = [c["target"] for c in checks.values() if c["kind"] == "must_not" and c["p"] >= settings.tau_must_not]
        coverage = sum(f["covered"] for f in facts) / len(facts)
        detail |= {"coverage": round(coverage, 4), "coverage_soft": round(statistics.mean(f["p"] for f in facts), 4),
                   "facts": facts, "must_not_violations": violations}
        if violations or unsupported:
            return {"score": 0, "cause": "환각", "detail": detail}
        if coverage >= settings.coverage_full:
            return {"score": 2, "cause": None, "detail": detail}
        missing = [f for f in facts if not f["covered"]]
        not_retrieved = sum(not f["evidence_retrieved"] for f in missing)
        cause = "검색 실패" if not_retrieved * 2 > len(missing) else "생성 누락"
        return {"score": 1, "cause": cause, "detail": detail}

    # 답 없음·함정: 거절은 상태로 정한다 (1이 정의된 문항 = 답이 있는 함정이라 거절은 1점, 아니면 거절이 정답)
    scoring = item["scoring"]
    if refused:
        return refusal(1) if "1" in scoring else {"score": 2, "cause": None, "detail": detail}
    grade = checks["grade"]
    detail |= {"grade": grade["choice"], "grade_probabilities": grade["probabilities"]}
    score = int(grade["choice"])
    if score > 0 and unsupported:
        score = 0
    cause = {0: "환각", 1: "오거절", 2: None}[score]
    return {"score": score, "cause": cause, "detail": detail}


def _share(values: list[bool]) -> float | None:
    return round(sum(values) / len(values), 4) if values else None


def _mean(values: list[float]) -> float | None:
    return round(statistics.mean(values), 4) if values else None


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round(q * (len(ordered) - 1)))]


def headline(records: list[dict]) -> dict:
    """안전성(1점 이상 비율)·유용성(평균 점수 / 2)·점수 분포. 실행 오류 문항은 제외."""
    scores = [r["result"]["score"] for r in records if r["result"]["score"] is not None]
    return {"n": len(scores), "errors": len(records) - len(scores),
            "safety": _share([s >= 1 for s in scores]),
            "usefulness": round(statistics.mean(scores) / 2, 4) if scores else None,
            "dist": {str(k): scores.count(k) for k in (0, 1, 2)}}


def summarize(items_by_id: dict[str, dict], records: list[dict]) -> dict:
    repeats = sorted({r["repeat"] for r in records})
    by_group = {g: [r for r in records if r["group"] == g] for g in GROUPS} | {"전체": records}
    out: dict = {"repeats": len(repeats), "groups": {}}
    for name, recs in by_group.items():
        per_repeat = [headline([r for r in recs if r["repeat"] == k]) for k in repeats]
        entry = headline(recs)
        for metric in ("safety", "usefulness"):
            values = [h[metric] for h in per_repeat if h[metric] is not None]
            entry[f"{metric}_range"] = [min(values), max(values)] if values else None
        out["groups"][name] = entry

    ok = [r for r in records if r["status"] != "error"]
    answerable = [r for r in ok if items_by_id[r["id"]]["type"] in ("quant", "qual")
                  or items_by_id[r["id"]].get("expected_status") == "answered"]
    with_evidence = [r for r in ok if item_evidence(items_by_id[r["id"]])]
    quant = [r for r in ok if r["type"] == "quant"]
    qual_answered = [r for r in ok if r["type"] == "qual" and r["status"] == "answered"]
    unanswerable = [r for r in ok if items_by_id[r["id"]].get("expected_status") == "insufficient_context"]
    judged = [r["result"]["detail"] for r in ok if r["status"] == "answered"]
    k_hits = [r["result"]["detail"]["first_hit_rank"] for r in with_evidence]
    usage = [r["usage"] for r in ok if r.get("usage")]
    cost = [(u["prompt_tokens"] * settings.llm_price_in + u["completion_tokens"] * settings.llm_price_out) / 1e6 for u in usage]
    judge_tokens = sum(r.get("judge", {}).get("judge_tokens", 0) for r in records)
    sentences = sum(d["sentences_judged"] for d in judged)
    out["diagnostics"] = {
        "retrieval_hit_at_k": _share([rank is not None for rank in k_hits]),
        "mrr": _mean([1 / rank if rank else 0.0 for rank in k_hits]),
        "quant_accuracy": _share([r["result"]["detail"].get("value_ok", False) for r in quant]),
        "quant_citation_accuracy": _share([r["result"]["detail"]["cited_evidence"] for r in quant
                                           if r["result"]["detail"].get("value_ok")]),
        "qual_fact_coverage": _mean([r["result"]["detail"]["coverage"] for r in qual_answered]),
        "qual_fact_coverage_soft": _mean([r["result"]["detail"]["coverage_soft"] for r in qual_answered]),
        "faithfulness": round(1 - sum(len(d["unsupported_sentences"]) for d in judged) / sentences, 4) if sentences else None,
        "abstention_accuracy": _share([r["result"]["score"] == 2 for r in unanswerable]),
        "false_refusal_rate": _share([r["status"] == "insufficient_context" for r in answerable]),
        "latency_p50_ms": _percentile([r["latency_ms"] for r in ok if r.get("latency_ms")], 0.5),
        "latency_p95_ms": _percentile([r["latency_ms"] for r in ok if r.get("latency_ms")], 0.95),
        "llm_cost_krw_per_query": round(statistics.mean(cost), 2) if cost else None,
        "judge_tokens": judge_tokens,
        "judge_cost_usd": round(judge_tokens * 0.042 / 1e6, 4),
    }
    causes: dict[str, int] = {}
    for r in records:
        if cause := r["result"]["cause"]:
            causes[cause] = causes.get(cause, 0) + 1
    out["causes"] = dict(sorted(causes.items(), key=lambda kv: -kv[1]))
    return out


def score_all(items_by_id: dict[str, dict], records: list[dict]) -> dict:
    for record in records:
        record["result"] = score_record(items_by_id[record["id"]], record)
    return summarize(items_by_id, records)

