"""Split page text into overlapping chunks using the embedding tokenizer."""

from hashlib import sha256


def split_text(
    text,
    tokenizer,
    chunk_size=100,
    overlap=20,
    max_seq_length=128,
):
    """Preserve source substrings and verify their final encoded lengths."""
    special_tokens = tokenizer.num_special_tokens_to_add(pair=False)

    if chunk_size <= 0 or chunk_size + special_tokens > max_seq_length:
        raise ValueError("Chunk size exceeds the embedding model's token budget.")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("Overlap must be non-negative and smaller than chunk size.")

    if not tokenizer.is_fast:
        raise ValueError("A fast tokenizer is required for character offsets.")

    if not text.strip():
        return []

    encoded = tokenizer(
        text,
        add_special_tokens=False,
        return_offsets_mapping=True,
        truncation=False,
        verbose=False,
    )

    offsets = [pair for pair in encoded["offset_mapping"] if pair[1] > pair[0]]

    chunks = []
    start = 0

    while start < len(offsets):
        end = min(start + chunk_size, len(offsets))

        # Re-encoding a substring can change its boundary tokens.
        # Check the actual saved text and shorten it when necessary.
        while end > start:
            char_start = offsets[start][0]
            char_end = offsets[end - 1][1]

            chunk_text = text[char_start:char_end]

            token_count = len(
                tokenizer.encode(
                    chunk_text,
                    add_special_tokens=True,
                    truncation=False,
                    verbose=False,
                )
            )

            if token_count <= chunk_size + special_tokens:
                break

            end -= 1

        else:
            raise ValueError("A text segment could not fit within the token budget.")

        chunks.append(
            {
                "text": chunk_text,
                "char_start": char_start,
                "char_end": char_end,
                "token_count": token_count,
            }
        )

        if end == len(offsets):
            break

        # Always advance, even when a window was shortened.
        start = max(start + 1, end - overlap)

    return chunks


def chunk_pages(
    pages,
    tokenizer,
    chunk_size=100,
    overlap=20,
    max_seq_length=128,
):
    """Keep document and page metadata with every chunk."""
    chunks = []
    seen_pages = set()

    for page in pages:
        page_key = (
            page["document_id"],
            page["page"],
        )

        if page_key in seen_pages:
            raise ValueError(
                f"Duplicate document page: {page['source']}, {page['page']}"
            )

        seen_pages.add(page_key)

        parts = split_text(
            page["text"],
            tokenizer,
            chunk_size,
            overlap,
            max_seq_length,
        )

        for chunk_index, part in enumerate(parts, start=1):
            identity = (
                f"{page['document_id']}:{page['page']}:"
                f"{part['char_start']}:{part['char_end']}:"
                f"{part['text']}"
            )

            chunk_id = sha256(identity.encode("utf-8")).hexdigest()

            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "document_id": page["document_id"],
                    "source": page["source"],
                    "page": page["page"],
                    "chunk_index": chunk_index,
                    **part,
                }
            )

    return chunks
