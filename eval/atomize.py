"""정성 루브릭 포인트 -> 사실 하나짜리 체크 문장들.

    uv run python -m eval.atomize [--force]

포인트 하나에 금액·날짜·대상이 묶여 있으면 사람도 judge도 "포함됐나"가 애매하다 (Jev PoC에서 확인).
그래서 포인트를 채점자가 하나씩 예/아니오로 볼 수 있는 사실로 쪼갠다. 골드셋 생성기도 이 함수를 쓴다.

검증(기계적): 분해 전 포인트와 분해 후 사실들의 숫자 집합이 같아야 한다.
숫자가 사라지면 사실을 빠뜨린 것이고, 새 숫자가 생기면 지어낸 것이다. 걸린 포인트는 경고로 남겨 사람이 본다.
"""

import argparse
import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from app.config import settings
from app.rag import llm, sampling_params

GOLD = Path("eval/gold/qual.jsonl")
REVIEW = Path("eval/gold/qual_atomized_review.md")
PROMPT_VERSION = "atomize-v2"  # v1 대비: 시행일 분리, 조건 유지
PROMPT = """아래는 RAG 답변 채점용 루브릭 포인트다. 채점자가 답변을 보고 하나씩 예/아니오로 확인할 수 있도록
이 포인트를 "사실 하나짜리 문장"들로 나눠라.

규칙:
- 문장 하나에는 확인할 사실 하나만 넣는다 (금액, 비율, 기간, 날짜, 대상 요건, 소관 기관, 적용 조건 중 하나).
- 같은 값을 다른 단위로 함께 쓴 것(예: "분기별 300만원(연 1,200만원)")은 사실 하나로 둔다.
- 시행일·적용 시점은 내용과 떼어 별도 사실로 둔다 (예: "한도가 연 1,800만원으로 바뀐다" / "한도 변경은 2026.7.1. 이후 납입분부터 적용된다").
- 바뀌기 전과 후를 함께 말하는 포인트는 "바뀐 뒤 값"과 "바뀌기 전 값"을 각각의 사실로 둔다.
- 조건이 붙은 내용은 조건을 유지한다 ("A하면 B를 받는다"를 "B를 받는다"나 "A한다"로 바꾸지 않는다).
- 문장마다 정책 이름이나 대상을 주어로 넣어, 그 문장만 읽어도 무엇에 대한 사실인지 알 수 있게 한다.
- 포인트에 없는 내용을 더하지 말고, 포인트의 내용을 빠뜨리지 않는다. 숫자와 날짜는 원문 그대로 쓴다.

JSON 객체 하나만 출력한다: {{"facts": ["...", "..."]}}

[질문] {question}
[포인트] {point}"""


def strip_ref(point: str) -> str:
    """포인트 끝의 근거 표기 "(PDF p.N, 정책명)"을 뗀다. 근거 쪽은 포인트 단위로 따로 보관한다."""
    return re.sub(r"\s*\(PDF p\.[^()]*(?:\([^()]*\)[^()]*)*\)\s*$", "", point).strip()


def numbers(text: str) -> set[str]:
    """비교용 숫자 집합. "2026.7.1." 같은 점 날짜는 연·월·일로 풀고, 천 단위 쉼표는 지운다.
    소수(0.5%p 등)는 그대로 둔다."""
    text = re.sub(r"(\d{4})\.(\d{1,2})\.(?:(\d{1,2})\.?)?", lambda m: " ".join(g for g in m.groups() if g) + " ", text)
    return set(re.findall(r"\d+(?:\.\d+)?", re.sub(r"(?<=\d),(?=\d{3})", "", text)))


def check(point: str, facts: list[str]) -> list[str]:
    warnings = []
    before, after = numbers(point), set().union(*(numbers(f) for f in facts)) if facts else set()
    if lost := sorted(before - after):
        warnings.append(f"사라진 숫자 {lost}")
    if added := sorted(after - before):
        warnings.append(f"새로 생긴 숫자 {added}")
    if len(facts) < 1:
        warnings.append("사실 0개")
    # 원자성 경고: 날짜는 값 1개로 세고, 값이 3개 이상이면 사실이 여러 개 섞였을 수 있다 (사람 검토 표시용)
    if crowded := [i for i, f in enumerate(facts) if value_count(f) >= 3]:
        warnings.append(f"사실 여러 개가 한 문장에 있을 수 있음 {crowded}")
    return warnings


def value_count(text: str) -> int:
    """값의 개수. 날짜(2026.7.1. / 2026년 7월 1일)와 범위(19~34세)는 각각 값 1개로 센다."""
    units = r"\d{4}\.\d{1,2}\.(?:\d{1,2}\.?)?|\d{4}년\s*\d{1,2}월(?:\s*\d{1,2}일)?|\d+(?:\.\d+)?\s*~\s*\d+(?:\.\d+)?"
    whole = re.findall(units, text)
    return len(set(whole)) + len(numbers(re.sub(units, " ", text)))


def atomize(question: str, point: str, api) -> list[str]:
    """분해 후 숫자가 사라졌으면 빠진 숫자를 알려주고 한 번 다시 시킨다."""
    messages = [{"role": "user", "content": PROMPT.format(question=question, point=point)}]
    facts: list[str] = []
    for _ in range(2):
        resp = api.chat.completions.create(
            model=settings.llm_model, messages=messages,
            response_format={"type": "json_object"}, **sampling_params("none"),
        )
        content = resp.choices[0].message.content
        facts = [f.strip() for f in json.loads(content)["facts"] if f.strip()]
        lost = sorted(numbers(point) - set().union(*(numbers(f) for f in facts)))
        if not lost:
            break
        messages += [{"role": "assistant", "content": content},
                     {"role": "user", "content": f"포인트의 숫자 {lost}가 담긴 사실이 빠졌다. 빠뜨리지 말고 다시 나눠라."}]
    return facts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="이미 분해된 포인트도 다시 분해")
    parser.add_argument("--redo-lost", action="store_true", help="숫자가 사라진 경고가 있는 포인트만 다시 분해")
    args = parser.parse_args()

    items = [json.loads(line) for line in GOLD.open(encoding="utf-8")]
    lost = lambda p: any(w.startswith("사라진") for w in p.get("atomize_warnings", []))  # noqa: E731
    jobs = [(item, p) for item in items for p in item["must_include"]
            if args.force or "facts" not in p or (args.redo_lost and lost(p))]
    if jobs:
        api = llm()
        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(lambda job: atomize(job[0]["question"], strip_ref(job[1]["point"]), api), jobs))
        for (_, p), facts in zip(jobs, results, strict=True):
            p["facts"] = facts
            p.pop("review", None)  # 다시 분해했으면 이전 검토 수정은 무효
            p["atomize"] = {"model": settings.llm_model, "prompt_version": PROMPT_VERSION}
    for item in items:  # 검증은 매번 전체를 다시 한다 (검증 규칙을 고쳐도 LLM을 다시 부르지 않게)
        for p in item["must_include"]:
            p["atomize_warnings"] = check(strip_ref(p["point"]), p["facts"])
    GOLD.write_text("".join(json.dumps(i, ensure_ascii=False) + "\n" for i in items), encoding="utf-8")

    lines = ["# 정성 루브릭 사실 단위 분해 검토", "",
             f"- 분해: {settings.llm_model}, {PROMPT_VERSION}. 검증: 분해 전후 숫자 집합 비교, 한 문장 값 3개 이상 경고.",
             "- 검토할 것: 사실이 원래 포인트 안에 있는지, 빠진 사실이 없는지, 문장 하나에 사실이 하나인지, 조건이 바뀌지 않았는지.", ""]
    for item in items:
        lines += [f"## {item['id']} ({item['subtype']})", "", f"질문: {item['question']}", ""]
        for i, p in enumerate(item["must_include"]):
            notes = [f"경고: {'; '.join(p['atomize_warnings'])}"] if p["atomize_warnings"] else []
            notes += [p["review"]] if p.get("review") else []
            lines.append(f"- 포인트 {i}: {strip_ref(p['point'])}" + "".join(f"  **{n}**" for n in notes))
            lines += [f"  - [{i}-{chr(97 + j)}] {f}" for j, f in enumerate(p["facts"])]
        lines.append("")
    REVIEW.write_text("\n".join(lines), encoding="utf-8")

    n_points = sum(len(i["must_include"]) for i in items)
    n_facts = sum(len(p["facts"]) for i in items for p in i["must_include"])
    flagged = [(i["id"], k, p["atomize_warnings"]) for i in items for k, p in enumerate(i["must_include"]) if p["atomize_warnings"]]
    print(f"포인트 {n_points}개 -> 사실 {n_facts}개, 경고 {len(flagged)}건 -> {REVIEW}")
    for f in flagged:
        print("  경고", *f)


if __name__ == "__main__":
    main()
