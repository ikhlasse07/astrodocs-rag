"""Extract all PDFs into inspectable, page-level records."""

import argparse
from importlib.metadata import version
import json
from pathlib import Path

from app.rag.pdf_loader import extract_pdf_pages


BACKEND_DIR = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--input-dir",
        type=Path,
        default=BACKEND_DIR / "data" / "documents",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=BACKEND_DIR / "data" / "processed",
    )

    args = parser.parse_args()

    if not args.input_dir.is_dir():
        raise SystemExit(f"Input folder not found: {args.input_dir}")

    pdf_files = sorted(
        (
            path
            for path in args.input_dir.iterdir()
            if path.is_file() and path.suffix.lower() == ".pdf"
        ),
        key=lambda path: path.name.casefold(),
    )

    if not pdf_files:
        raise SystemExit(f"No PDF files found in: {args.input_dir}")

    all_pages = []
    files = []

    # Read every PDF before writing the output files.
    # An unreadable PDF stops the run with a visible error.
    for pdf_path in pdf_files:
        try:
            pages = extract_pdf_pages(pdf_path)
        except Exception as error:
            raise SystemExit(f"Could not extract {pdf_path.name}: {error}") from error

        summary = {
            "source": pdf_path.name,
            "document_id": (pages[0]["document_id"] if pages else None),
            "pages": len(pages),
            "empty_pages": [
                page["page"] for page in pages if page["status"] == "empty"
            ],
            "short_pages": [
                page["page"] for page in pages if page["status"] == "short"
            ],
        }

        files.append(summary)
        all_pages.extend(pages)

        print(
            f"{pdf_path.name}: {len(pages)} pages | "
            f"empty={len(summary['empty_pages'])} | "
            f"short={len(summary['short_pages'])}",
            flush=True,
        )

    report = {
        "pymupdf_version": version("PyMuPDF"),
        "extraction_method": (
            "raw text plus dict spans with numeric superscripts written using ^"
        ),
        "page_numbering": ("1-based physical PDF page, not the printed page label"),
        "short_page_threshold_characters": 300,
        "note": (
            "Status ok measures text length only, "
            "not extraction accuracy. "
            "Review columns and tables."
        ),
        "total_files": len(files),
        "total_pages": len(all_pages),
        "empty_pages": sum(page["status"] == "empty" for page in all_pages),
        "short_pages": sum(page["status"] == "short" for page in all_pages),
        "superscript_spans": sum(page["superscript_spans"] for page in all_pages),
        "files": files,
    }

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    json_outputs = [
        ("pages.json", all_pages),
        ("extraction_report.json", report),
    ]

    for filename, content in json_outputs:
        output_path = args.output_dir / filename

        output_path.write_text(
            json.dumps(
                content,
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    review_parts = []

    for page in all_pages:
        review_parts.append(
            f"SOURCE: {page['source']}\n"
            f"PDF PAGE: {page['page']}\n"
            f"STATUS: {page['status']}\n\n"
            f"{page['text']}\n\n"
            "----------------------------------------\n\n"
        )

    review_path = args.output_dir / "extracted_text.txt"

    review_path.write_text(
        "".join(review_parts),
        encoding="utf-8",
    )

    print(f"\nTotal files: {report['total_files']}")
    print(f"Total PDF pages: {report['total_pages']}")
    print(f"Empty pages: {report['empty_pages']}")
    print(f"Short pages to review: {report['short_pages']}")
    print(f"Numeric superscript spans retained: {report['superscript_spans']}")
    print(f"Output folder: {args.output_dir.resolve()}")
    print("Saved: pages.json, extraction_report.json, extracted_text.txt")


if __name__ == "__main__":
    main()
