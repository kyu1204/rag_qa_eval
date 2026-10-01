"""Jev 판정: 답변 하나 = state 하나, 확인할 판정들을 한 요청에 묶어 병렬로 묻는다.

판정 종류 (모두 문서 무관 템플릿):
- value     정량: 답이 정답 값을 말하나 (Noul)
- fact      정성: 사실 하나가 답에 들어 있나 (Noul)
- must_not  정성: 잘못된 내용을 말하나 (Noul)
- support   답변 문장이 검색된 출처로 뒷받침되나 (Noul, 답변한 모든 문항)
- grade     답 없음·함정: 문항의 2/1/0 기준 중 무엇인가 (Choice)
거절한 문항은 판정하지 않는다 (채점 규칙이 상태만으로 정한다).
"""

import json
import re
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from app.config import settings

TEMPLATE_VERSION = "judge-ko-v1"
TEMPLATES = {
    "value": '"answer"가 "question"에 대한 답으로 다음 값을 분명히 말하는가? 값: "{target}". '
             "표기만 다른 같은 값(예: 5천만원 = 5,000만 원)은 같은 값으로 본다. 다른 숫자는 아니다.",
    "fact": '"answer"가 다음 사실을 (표현은 달라도) 담고 있는가? 사실: "{target}"',
    "must_not": '"answer"가 다음과 같은 잘못된 내용을 말하는가? 잘못된 내용: "{target}"',
    "support": 'id가 "{target}"인 문장만 "question"에 대한 답의 일부로 읽어라. 이 문장의 사실 주장이 "sources"로 '
               "완전히 뒷받침되는가? 숫자·날짜·조건이 하나라도 출처와 다르면 아니다. "
               "사실 주장이 없는 문장(인사, 안내 문구 등)은 그렇다로 답한다.",
    "grade": '"answer"가 "question"에 대해 한 응답에 가장 잘 맞는 기준을 고르라.',
}
MAX_QUESTIONS = 64  # 요청 하나에 넣는 판정 수 상한


def split_sentences(answer: str) -> list[str]:
    """채점용 문장 분리: 인용 번호와 마크다운을 지우고 줄·문장 끝으로 나눈다."""
    text = re.sub(r"\s*\[\d+\]", "", answer).replace("**", "")
    sentences = []
    for line in text.splitlines():
        line = re.sub(r"^\s*(?:[-*•]|\d+\.)\s*", "", line).strip()
        sentences += [s.strip() for s in re.split(r"(?<=[.?!])\s+(?=\S)", line)]
    return [s for s in sentences if len(s) >= 6]


def strip_explanation(text: str) -> str:
    """must_not 끝의 괄호 설명(정답과 근거 쪽)을 뗀다. 설명 속 정답이 판정을 흐리지 않게."""
    while text.endswith(")") and (start := _matching_open(text)) is not None:
        text = text[:start].rstrip()
    return text


def _matching_open(text: str) -> int | None:
    depth = 0
    for i in range(len(text) - 1, -1, -1):
        depth += {")": 1, "(": -1}.get(text[i], 0)
        if depth == 0:
            return i
    return None


def build_checks(item: dict, record: dict) -> tuple[dict, dict[str, tuple[str, str | dict]]]:
    """(state, {판정 id: (종류, 대상)}). 답변하지 않은 문항은 판정이 없다."""
    if record.get("status") != "answered":
        return {}, {}
    sentences = split_sentences(record["answer"])
    state = {"question": item["question"], "answer": record["answer"],
             "sources": [{"ref": h["ref"], "text": h["content"]} for h in record["retrieved"]],
             "sentences": [{"id": f"s{i}", "text": s} for i, s in enumerate(sentences)]}
    checks: dict[str, tuple[str, str | dict]] = {f"s{i}": ("support", f"s{i}") for i in range(len(sentences))}
    if item["type"] == "quant":
        checks["value"] = ("value", item["answer"])
    elif item["type"] == "qual":
        for pi, point in enumerate(item["must_include"]):
            checks |= {f"f{pi}_{fi}": ("fact", fact) for fi, fact in enumerate(point["facts"])}
        checks |= {f"m{mi}": ("must_not", strip_explanation(m)) for mi, m in enumerate(item.get("must_not", []))}
    else:
        checks["grade"] = ("grade", item["scoring"])
    return state, checks


def _question(kind: str, target) -> dict:
    if kind == "grade":
        return {"type": "choice", "instructions": TEMPLATES["grade"], "criteria": target}
    return {"type": "noul", "instructions": TEMPLATES[kind].format(target=target)}


class Jev:
    def __init__(self, endpoint: str | None = None, model: str | None = None, api_key: str | None = None):
        self.endpoint = endpoint or settings.judge_endpoint
        self.model = model or settings.judge_model
        self.api_key = api_key or settings.typesafe_api_key
        if not self.api_key:
            raise RuntimeError("TYPESAFE_API_KEY가 없다 (.env)")

    def ask(self, state: dict, questions: dict) -> tuple[dict, int]:
        """(answers, input_tokens). 429·5xx·네트워크 오류는 3번까지 다시 시도한다."""
        payload = json.dumps({"model": self.model, "state": state, "questions": questions}, ensure_ascii=False).encode()
        for attempt in range(4):
            request = urllib.request.Request(self.endpoint, payload, {
                "Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(request, timeout=60) as response:
                    body = json.loads(response.read())
                return body["answers"], body.get("usage", {}).get("input_tokens", 0)
            except urllib.error.HTTPError as e:
                if e.code != 429 and e.code < 500 or attempt == 3:
                    raise RuntimeError(f"Jev HTTP {e.code}: {e.read().decode()[:200]}") from e
            except urllib.error.URLError:
                if attempt == 3:
                    raise
            time.sleep(2 ** attempt)
        raise AssertionError("unreachable")


def judge_record(item: dict, record: dict, client: Jev, repeats: int) -> dict:
    """판정 결과: {판정 id: {"kind", "target", "p", "ps"}} (grade는 "choice", "probabilities", "choices").
    p는 반복 평균, ps·choices는 회차별 원값 (judge 일관성 측정용)."""
    state, checks = build_checks(item, record)
    if not checks:
        return {"checks": {}, "judge_tokens": 0}
    ids = list(checks)
    runs: dict[str, list] = {cid: [] for cid in ids}  # 회차별 noul 확률 또는 choice 확률 분포
    tokens = 0
    for _ in range(repeats):
        for start in range(0, len(ids), MAX_QUESTIONS):
            batch = {cid: _question(*checks[cid]) for cid in ids[start:start + MAX_QUESTIONS]}
            answers, used = client.ask(state, batch)
            tokens += used
            for cid, answer in answers.items():
                runs[cid].append(answer["probabilities"] if checks[cid][0] == "grade" else answer["noul"])
    out = {}
    for cid, (kind, target) in checks.items():
        if kind == "grade":
            keys = {k for dist in runs[cid] for k in dist}
            probs = {k: round(sum(d.get(k, 0.0) for d in runs[cid]) / repeats, 4) for k in keys}
            out[cid] = {"kind": kind, "choice": max(probs, key=probs.get), "probabilities": probs,
                        "choices": [max(d, key=d.get) for d in runs[cid]]}
        else:
            out[cid] = {"kind": kind, "target": target, "p": round(sum(runs[cid]) / repeats, 4),
                        "ps": [round(x, 4) for x in runs[cid]]}
    return {"checks": out, "judge_tokens": tokens}


def judge_all(items_by_id: dict[str, dict], records: list[dict], repeats: int | None = None, workers: int = 8) -> None:
    """records에 judge 결과를 채워 넣는다."""
    client = Jev()
    repeats = repeats or settings.judge_repeats
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = pool.map(lambda r: judge_record(items_by_id[r["id"]], r, client, repeats), records)
        for record, result in zip(records, results, strict=True):
            record["judge"] = result
