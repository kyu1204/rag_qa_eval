"""정량 골드셋 생성: LLM이 정책 1건에서 사실 문항을 만들고, 스크립트가 원문 대조로 검증한다.

    uv run python -m eval.gen_quant [--n 10] [--seed 7]

- 장(분야)별로 고르게 정책을 뽑아 정책마다 문항 2개를 만든다:
  literal(원문 표현을 쓴 질문)과 user(사용자 말투로 바꾼 질문). 같은 사실을 두 표현으로 물어
  "질문이 원문 단어를 베끼면 검색 점수가 부풀려지는가"를 Part B에서 직접 비교한다.
- 검증(기계적): 인용구가 그 쪽 원문에 글자 그대로 있는가, 정답이 인용구 안에 있는가.
  통과한 것만 남긴다. LLM이 지어낸 정답이 골드셋에 들어가는 것을 막는 장치다.
- 결과: eval/gold/quant.jsonl, 생성 기록(모델·시드·프롬프트·탈락 사유): eval/gold/quant_build.json
"""

import argparse
import json
import random
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

from app.config import settings
from app.extract import Page, extract_policy_book
from app.rag import llm, sampling_params

CORPUS_PDF = Path("data/corpus/2026년 하반기부터 이렇게 달라집니다.pdf")
OUT_DIR = Path("eval/gold")
PROMPT_VERSION = "quant-v1"
PROMPT = """아래는 정부 정책 안내 문서의 정책 1건이다. 이 정책에서 정답이 하나로 정해지는 사실 질문 1개를 만들어라.

조건:
- 정답은 금액, 비율, 인원, 기간, 날짜, 나이 같은 수치이거나, 대상 요건처럼 문서에 명시된 짧은 사실이다.
- quote는 정답이 들어 있는 원문 구절을 글자 하나 바꾸지 말고 그대로 복사한다 (120자 이내).
- answer는 30자 이내이고 quote 안에 그대로 들어 있어야 한다.
- question_literal은 문서의 정책명과 표현을 그대로 써서 묻는다.
- question_user는 같은 사실을 일반 시민이 일상 말투로 묻는 질문이다. 정책명이나 문서 표현을 그대로 쓰지 말고 바꿔 말한다.
- 질문만 보고도 어떤 정책인지 알 수 있어야 한다 (다른 정책과 헷갈리지 않게).

JSON 객체 하나만 출력한다:
{{"question_literal": "...", "question_user": "...", "answer": "...", "answer_kind": "금액|비율|인원|기간|날짜|나이|요건", "quote": "..."}}

[정책] {title} ({ministry})
{text}"""


def norm(s: str) -> str:
    """비교용 정규화: NFC, 따옴표 통일, 공백 제거 (PDF 줄 결합 차이에 둔감하게)."""
    s = unicodedata.normalize("NFC", s)
    s = re.sub(r"[‘’“”]", "'", s)
    return re.sub(r"\s+", "", s)


def pick_policies(pages: list[Page], n: int, seed: int) -> list[Page]:
    """장별로 섞은 뒤 장을 돌아가며 하나씩 뽑는다 (특정 분야 쏠림 방지)."""
    by_chapter = defaultdict(list)
    for p in pages:
        by_chapter[p.meta["chapter"]].append(p)
    rng = random.Random(seed)
    queues = [rng.sample(v, len(v)) for _, v in sorted(by_chapter.items())]
    picked = []
    while len(picked) < n and any(queues):
        for q in queues:
            if q and len(picked) < n:
                picked.append(q.pop())
    return picked


def generate(page: Page, api) -> dict:
    resp = api.chat.completions.create(
        model=settings.llm_model,
        messages=[{"role": "user", "content": PROMPT.format(text=page.text, **page.meta)}],
        response_format={"type": "json_object"},
        **sampling_params("none"),
    )
    return json.loads(resp.choices[0].message.content)


def verify(item: dict, page: Page) -> str | None:
    """탈락 사유를 돌려준다. 통과면 None."""
    keys = ("question_literal", "question_user", "answer", "quote")
    if any(not str(item.get(k, "")).strip() for k in keys):
        return "필드 누락"
    if norm(item["quote"]) not in norm(page.text):
        return "인용구가 원문에 없음"
    if norm(item["answer"]) not in norm(item["quote"]):
        return "정답이 인용구에 없음"
    if norm(page.meta["title"]) in norm(item["question_user"]):
        return "user 질문이 정책명을 그대로 씀"
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=10, help="통과시킬 정책 수 (문항은 2배)")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    pages = extract_policy_book(CORPUS_PDF.read_bytes())
    candidates = pick_policies(pages, len(pages), args.seed)  # 전체 순서만 정하고 통과 n개에서 멈춘다
    api, items, rejects = llm(), [], []
    for page in candidates:
        if len(items) // 2 >= args.n:
            break
        try:
            gen = generate(page, api)  # API 오류는 잡지 않는다: 설정 문제면 첫 호출에서 바로 멈춰야 한다
        except json.JSONDecodeError as e:
            gen = {"raw_error": str(e)}
        reason = verify(gen, page)
        label = f"PDF p.{page.page} {page.meta['title']}"
        if reason:
            rejects.append({"page": page.page, "title": page.meta["title"], "reason": reason, "generated": gen})
            print(f"  탈락  {label} - {reason}")
            continue
        n = len(items) // 2 + 1
        evidence = [{"document_id": CORPUS_PDF.name, "page": page.page, "quote": gen["quote"]}]
        source = {k: page.meta[k] for k in ("title", "ministry", "chapter", "printed_page")}
        for variant in ("literal", "user"):
            items.append({
                "id": f"quant-{n:02d}-{variant}",
                "type": "quant",
                "variant": variant,
                "question": gen[f"question_{variant}"],
                "answer": gen["answer"],
                "answer_kind": gen.get("answer_kind"),
                "evidence": evidence,
                "source_policy": source,
            })
        print(f"  통과  {label} | {gen['question_user']} -> {gen['answer']}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUT_DIR / "quant.jsonl").open("w", encoding="utf-8") as f:
        f.writelines(json.dumps(it, ensure_ascii=False) + "\n" for it in items)
    build = {
        "generator_model": settings.llm_model,
        "sampling": sampling_params("none"),
        "selection_seed": args.seed,
        "prompt_version": PROMPT_VERSION,
        "prompt": PROMPT,
        "corpus": CORPUS_PDF.name,
        "passed_policies": len(items) // 2,
        "tried_policies": len(items) // 2 + len(rejects),
        "rejects": rejects,
    }
    (OUT_DIR / "quant_build.json").write_text(json.dumps(build, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"통과 {build['passed_policies']} / 시도 {build['tried_policies']} -> {OUT_DIR / 'quant.jsonl'} ({len(items)}문항)")


if __name__ == "__main__":
    main()
