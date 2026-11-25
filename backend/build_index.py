"""Embed the chunks, save a FAISS index, and verify reloading."""

import json
from pathlib import Path
from time import perf_counter

import numpy as np

from app.rag.embedder import Embedder
from app.rag.vector_store import load_index, save_index


BACKEND_DIR = Path(__file__).resolve().parent


def main():
    chunks_path = BACKEND_DIR / "data" / "processed" / "chunks.json"

    index_dir = BACKEND_DIR / "data" / "index"

    if not chunks_path.is_file():
        raise SystemExit(f"Chunks file not found: {chunks_path}")

    chunks = json.loads(chunks_path.read_text(encoding="utf-8-sig"))

    if not isinstance(chunks, list) or not chunks:
        raise SystemExit("The chunks file must contain a non-empty list.")

    texts = [chunk["text"] for chunk in chunks]

    print(
        f"Chunks to index: {len(chunks)}",
        flush=True,
    )

    print(
        "Loading the embedding model on CPU...",
        flush=True,
    )

    start = perf_counter()
    embedder = Embedder()

    print(
        f"Model loaded in {perf_counter() - start:.2f} seconds",
        flush=True,
    )

    print(f"Embedding dimension: {embedder.dimension}")

    print(f"Model token limit: {embedder.max_seq_length}")

    start = perf_counter()
    vectors = embedder.encode(texts)

    print(f"Embedding shape: {vectors.shape}")

    print(f"Embedding time: {perf_counter() - start:.2f} seconds")

    save_index(
        vectors,
        chunks,
        index_dir,
        embedder.model_name,
        embedder.max_seq_length,
    )

    loaded_index, loaded_chunks, config = load_index(index_dir)

    restored_vectors = loaded_index.reconstruct_n(
        0,
        loaded_index.ntotal,
    )

    if loaded_chunks != chunks or not np.allclose(restored_vectors, vectors):
        raise RuntimeError("Reload verification failed: vector or chunk order changed.")

    scores, positions = loaded_index.search(
        vectors[:1],
        1,
    )

    if positions[0, 0] < 0 or not np.isclose(
        scores[0, 0],
        1.0,
        atol=1e-4,
    ):
        raise RuntimeError("Index self-similarity check failed.")

    print(f"\nIndexed vectors: {config['vector_count']}")

    print(f"Saved chunks: {len(loaded_chunks)}")

    print("Vector normalization: passed")
    print("File hash checks: passed")
    print("Reload and ordering check: passed")
    print("Index self-similarity check: passed")
    print(f"Index folder: {index_dir}")

    print("Saved: rag_index.faiss, chunks.json, index_config.json")


if __name__ == "__main__":
    main()
