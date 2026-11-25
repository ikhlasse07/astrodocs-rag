"""Extract PDF pages while retaining original page numbers and text."""

from hashlib import sha256
from pathlib import Path
import re
import unicodedata

import pymupdf


def clean_text(text: str) -> str:
    """Apply conservative whitespace and soft-hyphen cleanup."""
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Join words split by a soft hyphen at a printed line break.
    text = re.sub(r"\u00ad[ \t]*\n[ \t]*", "", text)
    text = text.replace("\u00ad", "")

    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")]

    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def extract_formatted_text(page) -> tuple[str, int]:
    """Preserve detected numeric superscripts using caret notation.

    Formatting is read from PDF spans, rather than inferred from numbers.
    Superscript detection is heuristic; raw text is retained separately.
    """
    data = page.get_text(
        "dict",
        flags=pymupdf.TEXTFLAGS_TEXT,
        sort=False,
    )

    blocks = []
    superscript_count = 0

    for block in data["blocks"]:
        if block.get("type") != 0:
            continue

        lines = []

        for line in block["lines"]:
            parts = []

            for span in line["spans"]:
                text = span["text"]

                is_superscript = bool(span["flags"] & 1)
                is_numeric = re.fullmatch(
                    r"[+\-\u2212]?\d+",
                    text.strip(),
                )

                if is_superscript and is_numeric:
                    text = "^" + text.strip()
                    superscript_count += 1

                parts.append(text)

            lines.append("".join(parts))

        blocks.append("\n".join(lines))

    return "\n".join(blocks), superscript_count


def extract_pdf_pages(pdf_path: str | Path) -> list[dict]:
    """Return every physical PDF page, including empty and short pages.

    Native PDF text order is retained. Multi-column layouts and tables
    still require manual review.
    """
    pdf_path = Path(pdf_path)

    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    document_id = sha256(pdf_path.read_bytes()).hexdigest()
    pages = []

    with pymupdf.open(pdf_path) as document:
        if document.needs_pass:
            raise ValueError(f"Password-protected PDF: {pdf_path.name}")

        for page_number, page in enumerate(document, start=1):
            raw_text = page.get_text("text", sort=False)

            formatted_text, superscript_count = extract_formatted_text(page)

            text = clean_text(formatted_text)

            if not text:
                status = "empty"
            elif len(text) < 300:
                status = "short"
            else:
                status = "ok"

            pages.append(
                {
                    "document_id": document_id,
                    "source": pdf_path.name,
                    "page": page_number,
                    "raw_text": raw_text,
                    "text": text,
                    "char_count": len(text),
                    "status": status,
                    "superscript_spans": superscript_count,
                }
            )

    return pages
