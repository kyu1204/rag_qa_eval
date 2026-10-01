import unicodedata
from pathlib import Path

from app.extract import clean, extract_policy_book, join_lines, sniff

CORPUS_PDF = Path("data/corpus/2026년 하반기부터 이렇게 달라집니다.pdf")


def test_sniff_trusts_content_not_name():
    assert sniff(b"%PDF-1.4 ...", "report.txt") == "pdf"
    assert sniff("정책".encode(), "a.md") == "text"
    assert sniff(b"\xff\xfe\x00", "a.md") is None  # UTF-8 아님
    assert sniff(b"PK\x03\x04", "a.docx") is None


def test_clean_removes_control_chars_and_normalizes():
    nfd = unicodedata.normalize("NFD", "추진배경")
    out = clean(f"*\x04참고 ••항목 ‌\x07{nfd}\n\n  ")
    assert out == "*참고 •항목 추진배경"
    assert unicodedata.is_normalized("NFC", out)


def test_join_lines_follows_indesign_line_breaks():
    lines = [("AI를 활용한 분석도 ", 0), ("어려워 지난 15", 10), ("년간 확대", 20), ("• 다음 항목", 30)]
    assert join_lines(lines) == "AI를 활용한 분석도 어려워 지난 15년간 확대\n• 다음 항목"


def test_join_lines_separates_table_cells_but_not_labels():
    assert join_lines([("구 분", 0), ("제도 시행 전", 0.5), ("제도 시행 후", 0)]) == "구 분 | 제도 시행 전 | 제도 시행 후"
    assert join_lines([("추진배경", 0), ("배경 설명", 0)]) == "추진배경 배경 설명"


def test_policy_book_extraction_regression():
    pages = extract_policy_book(CORPUS_PDF.read_bytes())
    assert len(pages) == 245
    numbers = [p.page for p in pages]
    assert numbers == sorted(numbers) and len(set(numbers)) == 245
    assert all(p.page - p.meta["printed_page"] == 42 for p in pages)  # 이 책은 인쇄 쪽 = PDF 쪽 - 42
    assert all(p.meta["chapter"] and p.meta["ministry"] and p.meta["title"] for p in pages)
    first = next(p for p in pages if p.meta["printed_page"] == 18)
    assert first.meta == {
        "printed_page": 18,
        "chapter": "제1장 금융·재정·조세",
        "ministry": "기획예산처",
        "title": "통합재정정보 플랫폼 ‘모두의 재정’ 구축",
        "effective": "2026년 12월",
        "tags": ["통합재정정보", "AI 재정플랫폼", "정보공개"],
    }
    assert "구 분 | 제도 시행 전 | 제도 시행 후" in first.text
    text = "".join(p.text for p in pages)
    assert "••" not in text and not any(unicodedata.category(c) in ("Cc", "Cf", "Co") for c in text if c != "\n")

