import io
import re

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from app.models import Page


class PdfError(ValueError):
    pass


def clean_text(text):
    text = text.replace("­", "")
    text = re.sub(r"-\n(?=[a-zäöüß])", "", text)  # silbentrennung am zeilenende
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_pages(data: bytes) -> list[Page]:
    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted:
            raise PdfError("Verschlüsselte PDFs gehen leider nicht.")
        pages = []
        for i, page in enumerate(reader.pages, start=1):
            text = clean_text(page.extract_text() or "")
            if text:
                pages.append(Page(i, text))
    except PdfReadError as e:
        raise PdfError(f"Kein gültiges PDF: {e}") from e

    if not pages:
        raise PdfError("Kein Text im PDF gefunden (vielleicht ein Scan?)")
    return pages
