import re

from app.models import Chunk, Page


def split_long_paragraph(text, size):
    parts = []
    current = ""
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        while len(sentence) > size:
            parts.append(sentence[:size])
            sentence = sentence[size:]
        if current and len(current) + 1 + len(sentence) > size:
            parts.append(current)
            current = sentence
        else:
            current = (current + " " + sentence).strip()
    if current:
        parts.append(current)
    return parts


def chunk_pages(pages: list[Page], doc_id, filename, size=900, overlap=150) -> list[Chunk]:
    # chunks gehen nie über eine seite hinaus, sonst stimmt die seitenzahl beim zitat nicht
    if overlap >= size:
        raise ValueError("overlap muss kleiner als size sein")

    chunks = []

    def add(page_nr, text):
        chunks.append(Chunk(f"{doc_id}:{len(chunks)}", doc_id, filename, page_nr, text))

    for page in pages:
        pieces = []
        for para in re.split(r"\n\s*\n", page.text):
            para = " ".join(para.split())
            if para:
                pieces += split_long_paragraph(para, size)

        current = ""
        for piece in pieces:
            if current and len(current) + 2 + len(piece) > size:
                add(page.number, current)
                tail = current[-overlap:] if overlap else ""
                if " " in tail:
                    tail = tail[tail.find(" ") + 1 :]
                current = (tail + " " + piece).strip()
            elif current:
                current += "\n\n" + piece
            else:
                current = piece
        if current:
            add(page.number, current)

    return chunks
