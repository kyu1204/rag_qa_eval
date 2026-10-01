"""Page 목록 -> Chunk 목록.

- split (기준선): 문서 구조를 모르는 범용 재귀 분할. 큰 경계(빈 줄 -> 줄 -> 문장 -> 공백)부터
  잘라 조각을 만들고, 목표 토큰까지 이어 붙이며 직전 청크 끝 조각들로 overlap을 준다.
"""

import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path

import tiktoken

from app.config import settings
from app.extract import Page, extract

ENC = tiktoken.get_encoding("o200k_base")  # 크기 산정용 근사 (임베딩 모델 토크나이저와 정확히 같지 않음)
SEPARATORS = ["\n\n", "\n", ". ", " "]


@dataclass
class Chunk:
    ord: int
    content: str
    page_start: int | None
    page_end: int | None
    embed_text: str  # 임베딩 입력 (문서 제목 등 문맥 헤더 + content)
    meta: dict = field(default_factory=dict)


def n_tokens(text: str) -> int:
    return len(ENC.encode(text))


def _pieces(text: str, max_tokens: int, seps: list[str]) -> list[str]:
    """max_tokens 이하 조각으로 자른다. 조각을 이으면 원문이 된다 (구분자는 앞 조각 끝에 남김)."""
    if n_tokens(text) <= max_tokens:
        return [text]
    if not seps:  # 공백 없는 긴 문자열: 글자 수로 자른다 (한글은 1글자 <= 1토큰이라 안전)
        return [text[i : i + max_tokens] for i in range(0, len(text), max_tokens)]
    sep, rest = seps[0], seps[1:]
    if sep not in text:
        return _pieces(text, max_tokens, rest)
    parts = text.split(sep)
    parts = [part + sep for part in parts[:-1]] + [parts[-1]]
    return [piece for part in parts if part for piece in _pieces(part, max_tokens, rest)]


def split_chunks(pages: list[Page], doc_title: str, max_tokens: int, overlap: int) -> list[Chunk]:
    # (조각, 쪽) 목록. 쪽 끝 조각에는 줄바꿈을 붙여 쪽 경계를 넘는 청크도 자연스럽게 이어진다.
    items: list[tuple[str, int | None]] = []
    for page in pages:
        pieces = _pieces(page.text, max_tokens, SEPARATORS)
        pieces[-1] += "\n"
        items += [(piece, page.page) for piece in pieces]

    groups, cur = [], []
    for item in items:
        if cur and n_tokens("".join(t for t, _ in cur + [item])) > max_tokens:
            groups.append(cur)
            tail = []  # overlap: 직전 청크 끝에서 overlap 토큰 이하가 되도록 조각을 가져온다
            for prev in reversed(cur):
                if n_tokens("".join(t for t, _ in [prev] + tail)) > overlap:
                    break
                tail.insert(0, prev)
            while tail and n_tokens("".join(t for t, _ in tail + [item])) > max_tokens:
                tail.pop(0)
            cur = tail
        cur.append(item)
    if cur:
        groups.append(cur)

    chunks = []
    for group in groups:
        content = "".join(t for t, _ in group).strip()
        pages_in = [p for _, p in group if p is not None]
        start, end = (min(pages_in), max(pages_in)) if pages_in else (None, None)
        chunks.append(
            Chunk(len(chunks), content, start, end, f"{doc_title}\n{content}", {"token_count": n_tokens(content)})
        )
    return chunks


def chunk(pages: list[Page], doc_title: str) -> list[Chunk]:
    if settings.chunk_strategy == "split":
        return split_chunks(pages, doc_title, settings.chunk_tokens, settings.chunk_overlap)
    raise ValueError(f"알 수 없는 청킹 전략: {settings.chunk_strategy}")


if __name__ == "__main__":
    # python -m app.chunking <파일>: 현재 설정(EXTRACTOR, CHUNK_STRATEGY)으로 청크 통계 출력
    path = Path(sys.argv[1])
    chunks = chunk(extract(path.read_bytes(), path.name, settings.extractor), path.stem)
    tokens = sorted(c.meta["token_count"] for c in chunks)
    print(f"extractor={settings.extractor} strategy={settings.chunk_strategy} index_version={settings.index_version()}")
    print(
        f"chunks={len(chunks)} tokens p50={statistics.median(tokens):.0f} "
        f"p95={tokens[int(0.95 * (len(tokens) - 1))]} max={tokens[-1]} min={tokens[0]} total={sum(tokens)}"
    )
