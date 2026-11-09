import pytest

from app.chunking import chunk_pages
from app.models import Page
from app.pdf_loader import PdfError, extract_pages


def test_extract_pages_keeps_page_numbers_and_umlauts(sample_pdf):
    pages = extract_pages(sample_pdf)
    assert [p.number for p in pages] == [1, 2, 3, 4, 5, 6]
    assert "30 Urlaubstage" in pages[2].text
    assert "Weiterbildungsbudget" in pages[3].text
    assert "Passwörter" in pages[5].text


def test_extract_pages_rejects_non_pdf():
    with pytest.raises(PdfError):
        extract_pages(b"das ist kein pdf")


def test_chunks_respect_size_and_never_cross_pages():
    text = "\n\n".join(f"Absatz {i}. " + "Wort " * 40 for i in range(10))
    pages = [Page(1, text), Page(2, "Kurze zweite Seite.")]
    chunks = chunk_pages(pages, "doc", "a.pdf", size=300, overlap=50)

    assert len(chunks) > 3
    assert all(len(c.text) <= 300 + 50 for c in chunks)
    assert chunks[-1].page == 2 and chunks[-1].text == "Kurze zweite Seite."
    assert len({c.id for c in chunks}) == len(chunks)


def test_chunks_overlap_carries_context():
    para_a = "Erster Absatz mit wichtigem Kontext am Ende Schlusswort."
    para_b = "Zweiter Absatz."
    chunks = chunk_pages([Page(1, f"{para_a}\n\n{para_b}")], "d", "a.pdf", size=60, overlap=20)
    assert len(chunks) == 2
    assert "Schlusswort." in chunks[1].text


def test_very_long_sentence_is_hard_split():
    chunks = chunk_pages([Page(1, "x" * 2500)], "d", "a.pdf", size=1000, overlap=0)
    assert [len(c.text) for c in chunks] == [1000, 1000, 500]


def test_overlap_must_be_smaller_than_size():
    with pytest.raises(ValueError):
        chunk_pages([Page(1, "text")], "d", "a.pdf", size=100, overlap=100)
