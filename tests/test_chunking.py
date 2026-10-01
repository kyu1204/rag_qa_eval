import pytest

from app.chunking import _pieces, n_tokens, policy_chunks, split_chunks
from app.extract import Page, extract_generic


def _pages():
    sentence = "노란우산공제 납입한도가 분기 300만원에서 연 1,800만원으로 확대됩니다. "
    return [Page(1, sentence * 30), Page(2, "짧은 쪽\n" + sentence * 5), Page(3, "가" * 300)]


def test_pieces_reassemble_to_original_and_fit():
    text = _pages()[0].text
    pieces = _pieces(text, 50, ["\n\n", "\n", ". ", " "])
    assert "".join(pieces) == text
    assert all(n_tokens(p) <= 50 for p in pieces)


def test_split_chunks_respect_size_cover_text_and_overlap():
    pages = _pages()
    chunks = split_chunks(pages, "문서", max_tokens=100, overlap=20)
    assert all(c.meta["token_count"] <= 100 for c in chunks)
    assert [c.ord for c in chunks] == list(range(len(chunks)))
    joined = "".join(c.content for c in chunks)
    assert all(page.text.strip()[:30] in joined for page in pages)
    # 이어지는 청크는 앞 청크의 끝부분을 다시 담는다
    assert any(chunks[i].content[-15:] in chunks[i + 1].content for i in range(len(chunks) - 1))
    assert chunks[0].page_start == 1 and chunks[-1].page_end == 3
    assert all(c.page_start <= c.page_end for c in chunks)
    assert chunks[0].embed_text.startswith("문서\n")


def test_split_chunks_text_without_pages():
    chunks = split_chunks([Page(None, "본문 " * 200)], "md", max_tokens=100, overlap=0)
    assert len(chunks) > 1 and all(c.page_start is None for c in chunks)


def test_generic_extract_text_strips_bom_and_normalizes():
    pages = extract_generic("\ufeff정책\x07 안내".encode(), "text")
    assert [(p.page, p.text) for p in pages] == [(None, "정책 안내")]


def test_policy_chunks_one_per_policy_with_header():
    meta = {"chapter": "제1장 금융·재정·조세", "ministry": "기획예산처", "title": "모두의 재정 구축"}
    chunks = policy_chunks([Page(60, "추진배경: ...", meta), Page(61, "추진배경: ...", meta)])
    assert [(c.ord, c.page_start, c.page_end) for c in chunks] == [(0, 60, 60), (1, 61, 61)]
    assert chunks[0].embed_text.startswith("모두의 재정 구축 (기획예산처, 제1장 금융·재정·조세)\n")
    assert chunks[0].meta["heading"] == "제1장 금융·재정·조세 > 기획예산처 > 모두의 재정 구축"


def test_policy_chunks_require_policy_book_pages():
    with pytest.raises(ValueError):
        policy_chunks([Page(1, "본문")])
