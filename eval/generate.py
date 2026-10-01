"""범용 골드셋 생성기: 어떤 문서든 같은 방법으로 평가 문항 초안을 만든다.

    uv run python -m eval generate <문서 ...> --out eval/gold_<이름> [--quant 10] [--qual 5] [--traps 5] [--out-of-corpus 5]

mofe-2026h2 골드셋을 만든 절차를 문서 구조에 기대지 않게 옮긴 것이다.
- 범용 추출 -> 약 1,000토큰 구간 -> 문서를 앞에서부터 등분해 구간을 돌아가며 뽑는다 (한쪽 쏠림 방지).
- 정량: LLM이 구간에서 사실 질문 1개를 literal(원문 표현)·user(사용자 말투) 두 표현으로 만든다.
  인용구가 구간에 글자 그대로 있고(공백 무시) 정답이 인용구 안에 있어야 통과. LLM이 지어낸 정답을 막는다.
- 잘못된 전제: 통과한 정량 정답의 첫 숫자를 규칙으로 틀린 값으로 바꾸고, LLM이 그 값을 믿는 사용자의 확인 질문으로 쓴다.
  채점 기준 2/1/0(바로잡음 / 거절 / 동의)은 자동.
- 코퍼스 밖: LLM이 문서 주변 주제의 질문과 핵심 키워드를 제안하고, 키워드가 문서 전체에 한 번도 없을 때만 남긴다.
- 정성: LLM이 구간에서 여러 사실을 모아야 하는 질문·포인트(근거 인용구)·must_not을 만들고,
  인용구가 확인된 포인트만 eval/atomize로 사실 하나짜리 문장으로 나눈다.
- 모든 문항은 status draft. review.md를 보고 사람이 approved로 바꾼 것만 평가에 쓰인다.
"""

import hashlib
import json
import random
import re
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

from app.chunking import split_chunks
from app.config import settings
from app.extract import Page, extract
from app.rag import llm, sampling_params
from eval import atomize

VERSION = "generate-v1"
WINDOW_TOKENS = 1000
WORKERS = 8

QUANT_PROMPT = """아래는 문서의 한 구간이다. 이 구간에서 정답이 하나로 정해지는 사실 질문 1개를 만들어라.

조건:
- 정답은 금액, 비율, 인원, 횟수, 기간, 나이, 날짜 같은 수치이거나, 대상·요건·기관처럼 구간에 명시된 짧은 사실이다.
  구간에 여러 사실이 있으면 날짜보다 내용(금액·대상·요건 등)을 우선한다.
- 질문하는 시점에 따라 정답이 달라지는 질문(지금 시행 중인지, 며칠 남았는지 등)은 만들지 않는다.
- answer는 질문에 대한 완결된 답이어야 한다. 질문이 넓으면 answer가 조건 하나만 담지 않도록 질문을 좁혀라.
- quote는 정답이 들어 있는 구간의 구절을 글자 하나 바꾸지 말고 그대로 복사한다 (100자 이내).
- answer는 30자 이내이고 quote 안에 그대로 들어 있어야 한다.
- question_literal은 구간의 제목과 표현을 그대로 써서 묻는다.
- question_user는 같은 사실을 일반 사용자가 일상 말투로 묻는 질문이다. 구간의 제목이나 표현을 그대로 쓰지 말고 바꿔 말한다.
- 질문만 보고도 무엇에 대한 질문인지 알 수 있어야 한다 (문서의 다른 부분과 헷갈리지 않게).
- 목차·표지·색인처럼 사실 질문을 만들 수 없는 구간이면 {{"skip": "이유"}}만 출력한다.

JSON 객체 하나만 출력한다:
{{"question_literal": "...", "question_user": "...", "answer": "...", "answer_kind": "금액|비율|인원|횟수|기간|나이|날짜|요건|기타", "quote": "..."}}

[문서] {doc}
[구간]
{text}"""

TRAP_PROMPT = """아래 사실의 값을 잘못 알고 있는 사용자가 맞는지 확인하는 질문 1개를 만들어라.

조건:
- 사용자는 [틀린 값]을 사실로 믿고 있다. 질문에 [틀린 값]을 글자 그대로 넣는다.
- [정답]은 질문에 쓰지 않는다.
- "~라던데 맞나요?", "~인 거죠?"처럼 확인을 구하는 일상 말투로 쓴다.
- 질문만 보고도 무엇에 대한 질문인지 알 수 있어야 한다.

JSON 객체 하나만 출력한다: {{"question": "..."}}

[원래 질문] {question}
[정답] {answer}
[틀린 값] {wrong}
[근거 구절] {quote}"""

OUT_PROMPT = """아래는 한 문서에서 고르게 뽑은 구간들의 앞부분이다. 이 문서의 독자가 물을 법하지만 이 문서가 다루지 않는 대상에 대한 질문 {k}개를 제안하라.

조건:
- 구간에 나온 제도·항목은 묻지 않는다 (그 세부를 묻는 것도 안 된다). 문서에 아예 없을 다른 대상을 묻는다.
- 문서 분야와 가까워 헷갈리기 쉬운 대상이 좋다 (같은 분야의 다른 제도, 문서가 다루는 기간 밖의 수치 등).
- 절반쯤은 일반 상식으로 답하기 쉬운 질문으로 한다 (모델이 문서 대신 사전지식으로 답하는지 보려는 것).
- 질문하는 시점에 따라 답이 달라지는 질문은 만들지 않는다.
- keywords: 그 대상의 이름처럼, 문서가 이 질문을 다룬다면 반드시 나올 고유한 용어 1~2개. "세금", "지원" 같은 일반어는 쓰지 않는다.

JSON 객체 하나만 출력한다: {{"items": [{{"question": "...", "keywords": ["..."]}}]}}

[문서] {doc}
[구간 앞부분]
{heads}"""

QUAL_PROMPT = """아래는 문서의 한 구간이다. 이 구간을 근거로, 답하려면 여러 사실을 모아 정리해야 하는 질문 1개를 만들어라.

조건:
- subtype은 요약(여러 내용을 정리), 비교(전후·대상 간 차이), 조건 적용(사용자 상황에 규칙을 적용) 중 하나.
- 구간에 서로 다른 주제가 섞여 있으면 하나만 골라 그 주제로만 묻는다.
- question은 사용자가 실제로 물을 법한 말투(~알려주세요, ~인가요?)로 쓴다. 포인트를 질문에 나열하지 않는다 (질문이 채점 기준을 알려주면 안 된다).
- must_include: 좋은 답에 꼭 들어가야 할 포인트 3~5개. point는 한 문장, quote는 그 포인트의 근거 구절을 글자 하나 바꾸지 말고 그대로 복사한다 (100자 이내).
- must_not: 이 질문에 답할 때 저지르기 쉬운 틀린 진술 1~2개. 구간의 맞는 내용과 비슷하지만 숫자·대상·조건이 다른 진술로 쓰고, 끝에 괄호로 실제 내용을 적는다.
- 질문하는 시점에 따라 답이 달라지는 질문은 만들지 않는다.
- 질문만 보고도 무엇에 대한 질문인지 알 수 있어야 한다.
- 목차·표지처럼 사실이 부족한 구간이면 {{"skip": "이유"}}만 출력한다.

JSON 객체 하나만 출력한다:
{{"subtype": "요약|비교|조건 적용", "question": "...", "must_include": [{{"point": "...", "quote": "..."}}], "must_not": ["... (실제: ...)"]}}

[문서] {doc}
[구간]
{text}"""

CRITERIA_OUT = {"2": "문서에 없는 내용이라고 밝히고 답하지 않는다 (문서 안의 관련 내용을 다른 것으로 분명히 구분해 안내하는 것은 허용)",
                "0": "묻는 대상의 수치·기준을 제시하거나, 문서의 다른 내용을 그 답처럼 제시한다"}


@dataclass
class Window:
    ord: int
    doc: str  # document_id
    text: str
    pages: list[Page]  # 이 구간이 걸친 쪽. 쪽 없는 문서는 빈 목록

    def label(self) -> str:
        span = f"p.{self.pages[0].page}" + (f"-{self.pages[-1].page}" if len(self.pages) > 1 else "") if self.pages else f"구간 {self.ord}"
        return f"{self.doc} {span}"


def norm(s: str) -> str:
    """비교용: NFC + 공백 제거. 채점의 근거 매칭(scoring._covers)과 같은 기준이라 여기서 통과한 인용구는 거기서도 맞는다."""
    return re.sub(r"\s+", "", unicodedata.normalize("NFC", s))


def load_documents(paths: list[str]) -> tuple[list[dict], list[Window], str]:
    """(manifest용 문서 목록, 구간들, 정규화한 전체 텍스트). 추출은 문서 무관한 generic."""
    docs, windows, texts = [], [], []
    for path in map(Path, paths):
        data = path.read_bytes()
        doc_id = unicodedata.normalize("NFC", path.name)  # 적재(app.ingest)와 같은 문서 id
        pages = extract(data, doc_id, "generic")
        docs.append({"id": doc_id, "sha256": hashlib.sha256(data).hexdigest()})
        by_no = {p.page: p for p in pages}
        for c in split_chunks(pages, path.stem, WINDOW_TOKENS, 0):
            span = [by_no[n] for n in range(c.page_start, c.page_end + 1) if n in by_no] if c.page_start is not None else []
            windows.append(Window(len(windows), doc_id, c.content, span))
        texts += [p.text for p in pages]
    return docs, windows, norm("\n".join(texts))


def stratified(seq: list, k: int, seed: int) -> list:
    """seq를 앞에서부터 k등분하고 구간을 돌아가며 하나씩 뽑은 순서. 앞쪽 k개가 문서 전체에 고르게 퍼진다."""
    rng = random.Random(seed)
    k = max(1, min(k, len(seq)))
    bounds = [round(i * len(seq) / k) for i in range(k + 1)]
    queues = [rng.sample(seq[a:b], b - a) for a, b in pairwise(bounds)]
    order = []
    while any(queues):
        order += [q.pop() for q in queues if q]
    return order


def find_page(w: Window, quote: str) -> int | None:
    """인용구가 있는 쪽. 쪽 경계에 걸치거나 쪽 없는 문서면 None (근거는 인용구로 매칭된다)."""
    return next((p.page for p in w.pages if norm(quote) in norm(p.text)), None)


def overlap(question: str, text: str) -> float:
    """질문의 글자 2-gram 중 원문에 있는 비율. user 질문이 원문 표현을 베꼈는지 보는 검토 경고용."""
    q, t = (re.sub(r"[^\w]", "", s) for s in (question, text))
    grams = {q[i:i + 2] for i in range(len(q) - 1)}
    return sum(g in t for g in grams) / len(grams) if grams else 0.0


def perturb(answer: str, source: str) -> tuple[str, str] | None:
    """(틀린 정답, 바뀐 숫자+단위). 정답의 마지막 숫자를 바꾼다 (범위 "19~34세"의 앞을 바꾸면 범위가 뒤집힌다).
    날짜는 있을 수 있는 값으로만 (일은 1~28 안에서 ±7, 월은 ±1). 바꾼 정답이 원문 구간에 있으면 다음 후보로
    (우연히 맞는 값 방지). 숫자가 없으면 None."""
    found = list(re.finditer(r"\d[\d,]*(?:\.\d+)?", answer))
    if not found:
        return None
    m = found[-1]
    raw, unit = m.group(), answer[m.end():m.end() + 1]
    v = float(raw.replace(",", ""))
    if unit == "일" or (unit == "." and answer[:m.start()].endswith(".")):  # 2026.7.1. 의 1도 일
        candidates = [c for c in (v + 7, v - 7) if 1 <= c <= 28]
    elif unit == "월":
        candidates = [c for c in (v + 1, v - 1) if 1 <= c <= 12]
    elif 1900 <= v <= 2100 and unit in ("년", "."):
        candidates = [v + 1, v + 2]
    elif unit == "세":
        candidates = [v + 5, v - 5]
    elif unit == "%":
        candidates = [v * 2, v + 10] if v <= 50 else [v - 10, v - 20]
    elif v < 10 and v == int(v):
        candidates = [v + 1, v + 2]
    else:
        candidates = [v * 2, v * 3]
    decimals = len(raw.split(".")[1]) if "." in raw else 0
    for c in candidates:
        text = f"{c:,.{decimals}f}" if "," in raw else f"{c:.{decimals}f}"
        wrong = answer[:m.start()] + text + answer[m.end():]
        if text != raw and norm(wrong) not in norm(source):
            return wrong, text + unit.strip()
    return None


def ask(api, prompt: str) -> dict:
    resp = api.chat.completions.create(model=settings.llm_model, messages=[{"role": "user", "content": prompt}],
                                       response_format={"type": "json_object"}, **sampling_params("none"))
    try:  # API 오류는 잡지 않는다: 설정 문제면 첫 호출에서 바로 멈춰야 한다
        out = json.loads(resp.choices[0].message.content)
    except json.JSONDecodeError as e:
        return {"skip": f"JSON 아님: {e}"}
    return out if isinstance(out, dict) else {"skip": "JSON 객체 아님"}


def first_passing(candidates: list, n: int, make, label, accept=lambda gen, passed: None) -> tuple[list, list[dict]]:
    """후보를 앞에서부터 WORKERS개씩 병렬로 make(후보) -> (생성물, 탈락 사유)에 넣어 통과 n개를 후보 순서대로 모은다.
    accept(생성물, 지금까지 통과)는 모으는 시점의 검사 (문항 구성 상한 등). 탈락 사유를 돌려주면 탈락."""
    passed, rejects = [], []
    with ThreadPoolExecutor(WORKERS) as pool:
        for start in range(0, len(candidates), WORKERS):
            if len(passed) >= n:
                break
            batch = candidates[start:start + WORKERS]
            for cand, (gen, reason) in zip(batch, pool.map(make, batch)):
                reason = reason or accept(gen, passed)
                if reason:
                    rejects.append({"where": label(cand), "reason": reason, "generated": gen})
                    print(f"  탈락  {label(cand)} - {reason}")
                elif len(passed) < n:
                    passed.append((cand, gen))
                    print(f"  통과  {label(cand)}")
    return passed, rejects


def make_quant(api, w: Window) -> tuple[dict, str | None]:
    gen = ask(api, QUANT_PROMPT.format(doc=w.doc, text=w.text))
    if gen.get("skip"):
        return gen, f"건너뜀: {gen['skip']}"
    if any(not str(gen.get(k, "")).strip() for k in ("question_literal", "question_user", "answer", "quote")):
        return gen, "필드 누락"
    if norm(gen["quote"]) not in norm(w.text):
        return gen, "인용구가 원문에 없음"
    if norm(gen["answer"]) not in norm(gen["quote"]):
        return gen, "정답이 인용구에 없음"
    if norm(gen["question_literal"]) in norm(w.text):  # "Q. 누가 지원받을 수 있나요?" 같은 원문 소제목
        return gen, "질문이 원문 문장을 그대로 옮김"
    return gen, None


def make_trap(api, source: tuple[Window, dict, int]) -> tuple[dict, str | None]:
    w, q, _ = source
    wrong, token = perturb(q["answer"], w.text)
    gen = ask(api, TRAP_PROMPT.format(question=q["question_literal"], answer=q["answer"], wrong=wrong, quote=q["quote"]))
    gen["wrong"] = wrong
    question = str(gen.get("question", ""))
    if norm(token) not in norm(question):
        return gen, f"틀린 값({token})이 질문에 없음"
    if norm(q["answer"]) in norm(question):
        return gen, "정답이 질문에 들어감"
    return gen, None


def make_qual(api, w: Window) -> tuple[dict, str | None]:
    gen = ask(api, QUAL_PROMPT.format(doc=w.doc, text=w.text))
    if gen.get("skip"):
        return gen, f"건너뜀: {gen['skip']}"
    points = [p for p in gen.get("must_include") or []
              if isinstance(p, dict) and str(p.get("point", "")).strip() and str(p.get("quote", "")).strip()]
    gen["must_include"] = [p for p in points if norm(p["quote"]) in norm(w.text)]
    gen["dropped_points"] = [p for p in points if norm(p["quote"]) not in norm(w.text)]
    if not str(gen.get("question", "")).strip() or len(gen["must_include"]) < 2:
        return gen, "원문으로 확인된 포인트가 2개 미만"
    return gen, None


def out_of_corpus(api, docs: list[dict], windows: list[Window], full: str, k: int, seed: int) -> tuple[list[dict], list[dict]]:
    heads = "\n".join(f"- {' '.join(w.text[:150].split())}" for w in stratified(windows, 20, seed)[:20])
    gen = ask(api, OUT_PROMPT.format(k=k * 4, doc=", ".join(d["id"] for d in docs), heads=heads))
    passed, rejects = [], []
    for cand in gen.get("items") or []:
        cand = cand if isinstance(cand, dict) else {}
        keywords = [str(kw).strip() for kw in cand.get("keywords", []) if str(kw).strip()]
        found = [kw for kw in keywords if norm(kw) in full]
        reason = "필드 누락" if not str(cand.get("question", "")).strip() or not keywords else (
            f"키워드가 문서에 있음 {found}" if found else None)
        if reason:
            rejects.append({"where": "코퍼스 밖", "reason": reason, "generated": cand})
            print(f"  탈락  코퍼스 밖 {cand.get('question')} - {reason}")
        elif len(passed) < k:
            passed.append({"question": cand["question"].strip(), "keywords": keywords})
            print(f"  통과  코퍼스 밖 {cand['question']}")
    return passed, rejects


def generate(paths: list[str], out: Path, name: str | None = None, n_quant: int = 10, n_qual: int = 5,
             n_traps: int = 5, n_out: int = 5, seed: int = 7, api=None) -> dict:
    if (out / "manifest.json").exists():
        raise SystemExit(f"{out}에 이미 골드셋이 있다. 검토한 내용을 덮어쓰지 않도록 다른 --out을 쓴다.")
    api = api or llm()
    docs, windows, full = load_documents(paths)
    model = f"{VERSION}, {settings.llm_model}"
    print(f"문서 {len(docs)}개 -> 구간 {len(windows)}개 (약 {WINDOW_TOKENS}토큰)")

    def dates(gen: dict, passed: list) -> str | None:
        """날짜 정답은 20%까지. 시행일·기한이 구간마다 있는 문서에서 LLM이 쉬운 날짜 질문으로 쏠린다 (quant-v2에서 관찰)."""
        n_dates = sum(g.get("answer_kind") == "날짜" for _, g in passed)
        return "날짜 문항 상한" if gen.get("answer_kind") == "날짜" and n_dates >= max(1, n_quant // 5) else None

    print(f"정량 {n_quant}개 (문항은 literal·user 2배)")
    quant, rejects = first_passing(stratified(windows, n_quant, seed), n_quant, lambda w: make_quant(api, w), Window.label, dates)
    items: dict[str, list[dict]] = {"quant": [], "qual": [], "neg": []}
    for i, (w, q) in enumerate(quant, 1):
        evidence = [{"document_id": w.doc, "page": find_page(w, q["quote"]), "quote": q["quote"]}]
        weak = overlap(q["question_user"], w.text) >= overlap(q["question_literal"], w.text)
        for variant in ("literal", "user"):
            items["quant"].append({
                "id": f"quant-{i:02d}-{variant}", "type": "quant", "variant": variant,
                "question": q[f"question_{variant}"], "answer": q["answer"], "answer_kind": q.get("answer_kind"),
                "evidence": evidence, "status": "draft",
                "provenance": f"generated by eval/generate.py ({model}), quote and answer verified against the source text",
                **({"review_warnings": ["user 질문이 literal만큼 원문 표현을 씀"]} if weak and variant == "user" else {}),
            })

    print(f"잘못된 전제 {n_traps}개")
    sources = [(w, q, i) for i, (w, q) in enumerate(quant, 1) if perturb(q["answer"], w.text)]
    traps, more = first_passing(sources, n_traps, lambda s: make_trap(api, s), lambda s: f"{s[0].label()} (함정)")
    rejects += more
    for i, ((w, q, src), gen) in enumerate(traps, 1):
        items["neg"].append({
            "id": f"trap-{i:02d}", "type": "false_premise", "question": gen["question"].strip(), "expected_status": "answered",
            "evidence": [{"document_id": w.doc, "page": find_page(w, q["quote"]), "quote": q["quote"]}],
            "scoring": {"2": f"전제를 바로잡아 정답({q['answer']})을 답한다", "1": "정보가 불충분하다며 답하지 않는다",
                        "0": f"틀린 값({gen['wrong']})에 동의하거나 그 값을 답으로 제시한다"},
            "premise": {"wrong": gen["wrong"], "answer": q["answer"], "from": f"quant-{src:02d}"},
            "verification": "검증된 정량 정답의 숫자를 규칙으로 바꾼 값 (원문 구간에 없는 값)",
            "status": "draft", "provenance": f"generated by eval/generate.py ({model}) from a verified quant answer",
        })

    print(f"코퍼스 밖 {n_out}개")
    outs, more = out_of_corpus(api, docs, windows, full, n_out, seed)
    rejects += more
    for i, o in enumerate(outs, 1):
        items["neg"].append({
            "id": f"out-{i:02d}", "type": "unanswerable", "question": o["question"], "expected_status": "insufficient_context",
            "scoring": CRITERIA_OUT, "keywords": o["keywords"],
            "verification": f"키워드 {o['keywords']} 문서 전체 0회 - 자동 확인. 다른 표현으로 다루는지는 사람 확인",
            "status": "draft", "provenance": f"generated by eval/generate.py ({model}), keyword absence checked over the full text",
        })

    print(f"정성 {n_qual}개")
    qual, more = first_passing(stratified(windows, n_qual, seed + 1), n_qual, lambda w: make_qual(api, w), Window.label)
    rejects += more
    jobs = [(q["question"], p["point"]) for _, q in qual for p in q["must_include"]]
    with ThreadPoolExecutor(WORKERS) as pool:
        facts = iter(list(pool.map(lambda job: atomize.atomize(*job, api), jobs)))
    for i, (w, q) in enumerate(qual, 1):
        points = []
        for p in q["must_include"]:
            fs = next(facts)
            points.append({"point": p["point"], "evidence": [{"document_id": w.doc, "page": find_page(w, p["quote"]), "quote": p["quote"]}],
                           "facts": fs, "atomize_warnings": atomize.check(p["point"], fs),
                           "atomize": {"model": settings.llm_model, "prompt_version": atomize.PROMPT_VERSION}})
        items["qual"].append({
            "id": f"qual-{i:02d}", "type": "qual", "subtype": q.get("subtype"), "question": q["question"].strip(),
            "must_include": points, "must_not": [str(m) for m in q.get("must_not", [])],
            **({"dropped_points": q["dropped_points"]} if q["dropped_points"] else {}), "status": "draft",
            "provenance": f"generated by eval/generate.py ({model}), quotes verified, points split by eval/atomize.py ({atomize.PROMPT_VERSION})",
        })

    out.mkdir(parents=True, exist_ok=True)
    files = [f"{kind}.jsonl" for kind in items]
    for kind, rows in items.items():
        (out / f"{kind}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    counts = {kind: len(rows) for kind, rows in items.items()}
    manifest = {"name": name or out.name,
                "description": f"eval/generate.py 초안 ({', '.join(d['id'] for d in docs)}): 정량 {counts['quant']}, 정성 {counts['qual']}, "
                               f"답 없음·함정 {counts['neg']} - 사람 검토 전",
                "documents": docs, "files": files}
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    build = {"generator": VERSION, "model": settings.llm_model, "sampling": sampling_params("none"), "seed": seed,
             "window_tokens": WINDOW_TOKENS, "windows": len(windows),
             "requested": {"quant": n_quant, "qual": n_qual, "traps": n_traps, "out_of_corpus": n_out},
             "counts": counts, "rejects": rejects, "atomize_prompt": atomize.PROMPT_VERSION,
             "prompts": {"quant": QUANT_PROMPT, "trap": TRAP_PROMPT, "out_of_corpus": OUT_PROMPT, "qual": QUAL_PROMPT}}
    (out / "build.json").write_text(json.dumps(build, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out / "review.md").write_text(review_markdown(manifest, items, build), encoding="utf-8")
    return build


def _where(e: dict) -> str:
    return e["document_id"] + (f" p.{e['page']}" if e.get("page") is not None else "")


def review_markdown(manifest: dict, items: dict[str, list[dict]], build: dict) -> str:
    """사람 검토용. 뷰어마다 모양이 달라지는 인용(>)·접기(details)는 쓰지 않는다."""
    lines = [f"# 골드셋 초안 검토: {manifest['name']}", "",
             f"- 문서: {', '.join(d['id'] for d in manifest['documents'])} · 생성 {build['generator']} ({build['model']}) · 시드 {build['seed']}",
             f"- 탈락 {len(build['rejects'])}건은 build.json의 rejects에 사유와 함께 있다.",
             "- 모든 문항이 status draft라 아직 평가에 쓰이지 않는다. 쓸 문항은 jsonl에서 status를 approved로, 버릴 문항은 rejected로 바꾸고, 고칠 내용은 jsonl에서 직접 고친다.",
             "- 기계 검증은 통과했다 (인용구가 원문에 있음, 정답이 인용구 안에 있음, 코퍼스 밖 키워드 부재). 사람이 볼 것:",
             "  - 정량: 답이 하나로 정해지는가, 정답이 질문에 완결된 답인가, user 질문이 사용자 말투인가.",
             "  - 잘못된 전제: 틀린 값이 문서의 다른 곳에서 맞는 값은 아닌가, 질문이 자연스러운가.",
             "  - 코퍼스 밖: 문서가 다른 표현으로 이 내용을 다루지 않는가.",
             "  - 정성: 포인트가 질문에 꼭 필요한가, 빠진 포인트는 없는가, 사실 분해가 맞는가, must_not이 정말 틀린 진술인가.", "",
             f"## 정량 ({len(items['quant'])}문항)", ""]
    for literal, user in zip(items["quant"][::2], items["quant"][1::2]):
        lines += [f"### {literal['id'].removesuffix('-literal')} · {literal.get('answer_kind') or '-'} · {_where(literal['evidence'][0])}", "",
                  f"- literal: {literal['question']}", f"- user: {user['question']}"
                  + "".join(f" **경고: {w}**" for w in user.get("review_warnings", [])),
                  f"- 정답: {literal['answer']}", f"- 근거: \"{literal['evidence'][0]['quote']}\"", ""]
    lines += [f"## 답 없음·함정 ({len(items['neg'])}문항)", ""]
    for item in items["neg"]:
        lines += [f"### {item['id']} · {item['type']}", "", f"- 질문: {item['question']}"]
        if "premise" in item:
            p = item["premise"]
            lines += [f"- 정답 {p['answer']} / 틀린 값 {p['wrong']} ({p['from']}에서)",
                      f"- 근거: \"{item['evidence'][0]['quote']}\" ({_where(item['evidence'][0])})"]
        lines += [f"- 확인: {item['verification']}", ""]
    lines += [f"## 정성 ({len(items['qual'])}문항)", ""]
    for item in items["qual"]:
        lines += [f"### {item['id']} · {item.get('subtype') or '-'}", "", f"- 질문: {item['question']}"]
        for i, p in enumerate(item["must_include"]):
            lines.append(f"- 포인트 {i}: {p['point']} ({_where(p['evidence'][0])})"
                         + (f" **경고: {'; '.join(p['atomize_warnings'])}**" if p["atomize_warnings"] else ""))
            lines += [f"  - [{i}-{chr(97 + j)}] {fact}" for j, fact in enumerate(p["facts"])]
        lines += [f"- must_not: {m}" for m in item["must_not"]]
        lines += [f"- 버린 포인트 (인용구가 원문에 없음): {p['point']}" for p in item.get("dropped_points", [])]
        lines.append("")
    return "\n".join(lines)
