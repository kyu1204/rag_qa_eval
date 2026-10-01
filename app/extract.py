"""원본 파일 -> 정제된 정책 쪽 목록.

PDF 규칙은 「2026년 하반기부터 이렇게 달라집니다」 실측값에 맞춘 것이다 (README 참조).
상세 정책 쪽("추진배경"이 있는 쪽)만 적재 대상이고, 요약 카드·목차·부록 표는 같은 정책의
재구성이라 건너뛴다.
"""

import re
import sys
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

import pymupdf

DETAIL_MARKER = "추진배경"
FOOTER_ZONE = 40  # 쪽 하단에서 이 거리(pt) 안의 줄은 꼬리말
HEADER_ZONE = 35  # 쪽 상단에서 이 거리(pt) 안의 줄은 소관부처 머리말
CHAPTER_RE = re.compile(r"제\s*(\d+)\s*장\.\s*(\S[^\n]*)")
TITLE_RE = re.compile(r"^-\s*(.+?)\s*-$")
PAGE_NO_RE = re.compile(r"^(\d+)(?:\s|$)|•\s*(\d+)$")
BULLET_START = ("•", "-", "Q.", "*", "※", "☞", "#") + tuple("①②③④⑤⑥⑦⑧⑨⑩")
# 라벨이 내용과 같은 높이에 놓인 쪽은 표 셀처럼 " | "로 이어진다
CELL_LABEL_RE = re.compile(r"\s*(추진배경|주요내용|기대효과|시 ?행 ?일)\s*\|\s*")
SECTION_LABEL_RE = re.compile(r"^(추진배경|주요내용|기대효과|시 ?행 ?일)\s*:?\s*", re.M)
SENTENCE_END = (".", "?", "!", ")", "]", "다", "요")


@dataclass
class Page:
    """추출기 공통 출력. page는 PDF 쪽 인덱스(1부터), 쪽 개념이 없는 문서는 None."""

    page: int | None
    text: str
    meta: dict = field(default_factory=dict)


def sniff(data: bytes, name: str) -> str | None:
    """선언된 MIME이나 확장자를 믿지 않고 내용으로 판정한다. 'pdf' | 'text' | None."""
    if data.startswith(b"%PDF-"):
        return "pdf"
    if Path(name).suffix.lower() in {".md", ".txt"}:
        try:
            data.decode("utf-8")
            return "text"
        except UnicodeDecodeError:
            return None
    return None


def clean(text: str) -> str:
    """제어·서식·사설 문자 제거, 이중 글머리 정리, NFC, 공백 정리."""
    text = unicodedata.normalize("NFC", text)
    text = "".join(
        c for c in text if c in "\n\t" or unicodedata.category(c) not in ("Cc", "Cf", "Co")
    )
    text = re.sub(r"•+", "•", text)
    text = re.sub(r"(?m)^\s*-\s*-\s*", "- ", text)  # 글머리 기호 + 본문 대시
    text = re.sub(r"[ \t]+", " ", text)
    lines = (line.strip() for line in text.split("\n"))
    return "\n".join(line for line in lines if line)


def join_lines(lines: list[tuple[str, float]]) -> str:
    """한 블록 안의 (줄 텍스트, y0)를 원문대로 잇는다.

    같은 높이에 나란히 있는 줄은 표의 셀이라 " | "로 구분한다.
    InDesign은 단어 경계 줄바꿈이면 줄 끝 공백을 남기고 단어 중간 줄바꿈이면 남기지 않는다.
    그래서 끝 공백이 있으면 그대로 잇고, 다음 줄이 글머리로 시작하거나 앞 줄이 문장 끝이면
    줄을 바꾸고, 나머지(단어 중간)는 구분자 없이 잇는다.
    """
    out, prev_y = "", None
    for line, y0 in lines:
        same_row = prev_y is not None and abs(y0 - prev_y) < 2
        prev_y = y0
        if not out:
            out = line
        elif same_row:
            # 같은 줄에 붙은 글머리 기호나 문단 라벨("추진배경")은 셀이 아니다
            seg = re.sub(r"[^\w]", "", out.rsplit("\n", 1)[-1])
            sep = " " if len(seg) <= 1 or SECTION_LABEL_RE.match(seg) else " | "
            out = out.rstrip() + sep + line.lstrip()
        elif out.endswith(" "):
            out += line
        elif (
            line.lstrip().startswith(BULLET_START)
            or out.rstrip().endswith(SENTENCE_END)
            or out.rsplit("\n", 1)[-1].lstrip().startswith("#")  # 해시태그 줄
        ):
            out += "\n" + line
        else:
            out += line
    return out


def _page_lines(page: pymupdf.Page) -> list[tuple[float, float, list[tuple[str, float]]]]:
    """블록 단위 (y0, 폰트 크기, [(줄 텍스트, 줄 y0)]). 가로쓰기만."""
    blocks = []
    for b in page.get_text("dict")["blocks"]:
        lines = [l for l in b.get("lines", []) if l["dir"] == (1.0, 0.0)]
        texts = [("".join(s["text"] for s in l["spans"]), l["bbox"][1]) for l in lines]
        if lines and any(t.strip() for t, _ in texts):
            size = max(s["size"] for l in lines for s in l["spans"])
            blocks.append((lines[0]["bbox"][1], size, texts))
    return blocks


def extract_policy_book(data: bytes) -> list[Page]:
    """책자 전용 분석기: 상세 정책 쪽만, meta에 printed_page·chapter·ministry·title·effective·tags."""
    doc = pymupdf.open(stream=data, filetype="pdf")
    pages, chapter = [], ""
    for page in doc:
        raw = page.get_text()
        if m := CHAPTER_RE.search(raw):
            chapter = f"제{m.group(1)}장 {m.group(2).strip()}"
        if DETAIL_MARKER not in raw:
            continue
        height = page.rect.height
        page_no, ministry, title, body = None, "", "", []
        for y0, _size, texts in _page_lines(page):
            joined = clean(texts[0][0]) if len(texts) == 1 else None
            if y0 > height - FOOTER_ZONE:
                for t, _ in texts:
                    if pm := PAGE_NO_RE.search(t.strip()):
                        page_no = int(pm.group(1) or pm.group(2))
                continue
            if y0 < HEADER_ZONE and joined and not ministry:
                ministry = joined
                continue
            if joined and (tm := TITLE_RE.match(joined)):
                title = title or tm.group(1)
            body.append(clean(join_lines(texts)))
        text = "\n".join(t for t in body if t)
        text = CELL_LABEL_RE.sub(lambda m: "\n" + m.group(1).replace(" ", "") + ": ", text)
        text = SECTION_LABEL_RE.sub(lambda m: m.group(1).replace(" ", "") + ": ", text)
        if page_no is None or not (ministry and title):
            raise ValueError(f"PDF {page.number + 1}쪽: 쪽번호·부처·정책명 추출 실패")
        meta = {"printed_page": page_no, "chapter": chapter, "ministry": ministry, "title": title}
        pages.append(Page(page.number + 1, text, meta | _meta(text)))
    return pages


def _meta(text: str) -> dict:
    meta = {}
    if m := re.search(r"시행일:?\s*([^\n]+)", text):
        meta["effective"] = m.group(1).strip()
    if m := re.search(r"^#\s*[^\n]+$", text, re.M):
        # 해시태그 줄 뒤에 담당부서·전화가 붙어 나오는 쪽이 있어 기관명 앞에서 자른다
        tags = (re.split(r"\s+(?=\S+(?:부|처|청|위원회|본부)\s)", t)[0] for t in re.split(r"[#,]", m.group(0)))
        meta["tags"] = [t.strip() for t in tags if t.strip()]
    return meta


if __name__ == "__main__":
    path = Path(sys.argv[1])
    for p in extract_policy_book(path.read_bytes()):
        print(f"\n===== PDF p.{p.page} | {p.meta}")
        print(p.text)
