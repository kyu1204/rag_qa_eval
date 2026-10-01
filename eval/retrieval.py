"""검색만 따로 잰다: top-k를 늘려 가며 골드셋 근거가 검색 결과에 들어오는지 (생성·judge 없이 임베딩 호출만).

    uv run python -m eval.retrieval [--k 1,3,5,8,10,15,20,30,50]

- 근거 단위: 정성은 포인트(사실 묶음)마다, 정량·함정은 문항마다 하나. 단위 안의 근거 쪽 중 하나만 검색돼도 그 단위는 잡힌 것이다.
  근거가 없는 코퍼스 밖 문항은 뺀다.
- hit@k: 문항의 근거 단위 중 하나라도 top-k에 있는 문항 비율 (하네스 진단 지표 hit@k와 같은 정의)
- 근거 재현율@k: 근거 단위 중 top-k에 들어온 비율. 정성처럼 근거가 여러 곳인 문항의 검색 실패는 이쪽에 드러난다.
- 결과: eval/runs/<시각>_retrieval/summary.json, report.md
"""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from datetime import datetime

from app import rag
from app.config import settings
from eval.goldset import group, load
from eval.report import RUNS
from eval.scoring import GROUPS, _covers


def units(item: dict) -> list[list[dict]]:
    """근거 단위 목록. 단위 하나 = 그중 하나만 검색돼도 되는 근거들."""
    if item["type"] == "qual":
        return [point["evidence"] for point in item["must_include"]]
    return [item["evidence"]] if item.get("evidence") else []


def first_rank(hits: list[dict], unit: list[dict]) -> int | None:
    """근거 단위가 처음 검색된 순위 (1부터). 없으면 None."""
    return next((rank for rank, h in enumerate(hits, 1)
                 if any(_covers(h, e["document_id"], e.get("page"), e.get("quote")) for e in unit)), None)


def _share(flags) -> float | None:
    flags = list(flags)
    return round(sum(flags) / len(flags), 4) if flags else None


def curve(items: list[dict], ranks: dict[str, list[int | None]], ks: list[int]) -> dict:
    """묶음별 hit@k와 근거 재현율@k. ranks: 문항 id -> 근거 단위마다 처음 검색된 순위."""
    out = {}
    for name in [*GROUPS, "전체"]:
        chosen = [i for i in items if ranks.get(i["id"]) and name in ("전체", group(i))]
        if chosen:
            out[name] = {
                "items": len(chosen), "units": sum(len(ranks[i["id"]]) for i in chosen),
                "hit": {k: _share(any(r is not None and r <= k for r in ranks[i["id"]]) for i in chosen) for k in ks},
                "recall": {k: _share(r is not None and r <= k for i in chosen for r in ranks[i["id"]]) for k in ks},
            }
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", default="1,3,5,8,10,15,20,30,50", help="쉼표로 구분한 top-k 목록")
    parser.add_argument("--gold", default="eval/gold")
    args = parser.parse_args(argv)
    ks = sorted({int(k) for k in args.k.split(",")})
    gold = load(args.gold)
    items = [i for i in gold.items if units(i)]

    def ranks_of(item):
        hits = [asdict(h) for h in rag.retrieve(item["question"], max(ks))]
        return [first_rank(hits, unit) for unit in units(item)]

    with ThreadPoolExecutor(6) as pool:
        ranks = dict(zip((i["id"] for i in items), pool.map(ranks_of, items), strict=True))
    result = {"index_version": settings.index_version(), "embed_model": settings.embed_model,
              "gold": {"name": gold.name, "hash": gold.hash}, "ks": ks, "curve": curve(items, ranks, ks), "ranks": ranks}

    out = RUNS / f"{datetime.now():%Y%m%d-%H%M%S}_retrieval"
    out.mkdir(parents=True)
    (out / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    c = result["curve"]
    lines = ["# 검색 곡선: top-k별 근거 검색", "",
             (f"- 인덱스 {result['index_version']} ({result['embed_model']}), 골드셋 {gold.name} {gold.hash}, "
              f"문항 {c['전체']['items']}개, 근거 단위 {c['전체']['units']}개"), "",
             "| top-k | hit@k 전체 | 근거 재현율 전체 | " + " | ".join(f"근거 재현율 {g}" for g in GROUPS if g in c) + " |",
             "|---|---|---|" + "---|" * sum(g in c for g in GROUPS)]
    for k in ks:
        lines.append(f"| {k} | {c['전체']['hit'][k] * 100:.1f}% | {c['전체']['recall'][k] * 100:.1f}% | "
                     + " | ".join(f"{c[g]['recall'][k] * 100:.1f}%" for g in GROUPS if g in c) + " |")
    missing = [f"{item_id} 단위 {u}" for item_id, rs in ranks.items() for u, r in enumerate(rs) if r is None]
    lines += ["", f"## top-{max(ks)}에도 없는 근거 단위 ({len(missing)}개)", "", *[f"- {m}" for m in missing]]
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"저장: {out}/report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
