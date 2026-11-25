"""Build token-based chunks from the extracted PDF pages."""

from collections import Counter
from hashlib import sha256
import json
from pathlib import Path

from transformers import AutoTokenizer

from app.rag.chunker import chunk_pages


BACKEND_DIR = Path(__file__).resolve().parent

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

MAX_SEQ_LENGTH = 128
CHUNK_SIZE = 100
OVERLAP = 20


def main():
    processed_dir = BACKEND_DIR / "data" / "processed"
    input_path = processed_dir / "pages.json"

    if not input_path.is_file():
        raise SystemExit(f"Input file not found: {input_path}")

    pages = json.loads(input_path.read_text(encoding="utf-8-sig"))

    if not isinstance(pages, list) or not pages:
        raise SystemExit("pages.json must contain a non-empty list of page records.")

    print(
        "Loading the embedding tokenizer...",
        flush=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        use_fast=True,
    )

    chunks = chunk_pages(
        pages,
        tokenizer,
        CHUNK_SIZE,
        OVERLAP,
        MAX_SEQ_LENGTH,
    )

    if not chunks:
        raise SystemExit("No chunks were created. Check the extracted page text.")

    counts = Counter(chunk["source"] for chunk in chunks)

    maximum = max(chunk["token_count"] for chunk in chunks)

    report = {
        "embedding_model": MODEL_NAME,
        "model_max_tokens": MAX_SEQ_LENGTH,
        "chunk_size_content_tokens": CHUNK_SIZE,
        "overlap_window_tokens": OVERLAP,
        "special_tokens_per_chunk": (tokenizer.num_special_tokens_to_add(pair=False)),
        "input_pages_sha256": sha256(input_path.read_bytes()).hexdigest(),
        "total_input_pages": len(pages),
        "empty_pages_skipped": sum(not page["text"].strip() for page in pages),
        "total_chunks": len(chunks),
        "max_chunk_tokens_including_special_tokens": maximum,
        "all_chunks_within_model_limit": (maximum <= MAX_SEQ_LENGTH),
        "chunks_per_source": dict(counts),
    }

    for filename, content in [
        ("chunks.json", chunks),
        ("chunking_report.json", report),
    ]:
        output_path = processed_dir / filename

        output_path.write_text(
            json.dumps(
                content,
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    preview = []

    for chunk in chunks:
        preview.append(
            f"SOURCE: {chunk['source']}\n"
            f"PDF PAGE: {chunk['page']}\n"
            f"CHUNK: {chunk['chunk_index']}\n"
            f"TOKENS INCLUDING SPECIAL TOKENS: "
            f"{chunk['token_count']}\n\n"
            f"{chunk['text']}\n\n"
            "----------------------------------------\n\n"
        )

    preview_path = processed_dir / "chunks_preview.txt"

    preview_path.write_text(
        "".join(preview),
        encoding="utf-8",
    )

    print(f"\nInput pages: {len(pages)}")

    for source, count in counts.items():
        print(f"{source}: {count} chunks")

    print(f"Total chunks: {len(chunks)}")

    print(f"Maximum chunk tokens including special tokens: {maximum}")

    print(f"Embedding model token limit: {MAX_SEQ_LENGTH}")

    print(
        f"All chunks fit the embedding model: {report['all_chunks_within_model_limit']}"
    )

    print("Saved: chunks.json, chunking_report.json, chunks_preview.txt")


if __name__ == "__main__":
    main()
