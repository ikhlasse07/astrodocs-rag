"""Save and load a FAISS index with its ordered chunk metadata."""

from hashlib import sha256
import json
from pathlib import Path

import faiss
import numpy as np


def file_hash(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def save_index(
    vectors,
    chunks,
    index_dir,
    model_name,
    max_seq_length,
):
    vectors = np.ascontiguousarray(
        vectors,
        dtype=np.float32,
    )

    if vectors.ndim != 2 or vectors.shape[0] != len(chunks) or not chunks:
        raise ValueError("Expected one vector for every chunk, in the same order.")

    chunk_ids = {chunk["chunk_id"] for chunk in chunks}

    if len(chunk_ids) != len(chunks):
        raise ValueError("Chunk IDs must be unique.")

    if not np.isfinite(vectors).all():
        raise ValueError("Vectors contain non-finite values.")

    norms = np.linalg.norm(vectors, axis=1)

    if not np.allclose(norms, 1.0, atol=1e-4):
        raise ValueError("Normalize vectors before building the index.")

    index_dir = Path(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)

    index_path = index_dir / "rag_index.faiss"
    chunks_path = index_dir / "chunks.json"

    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    faiss.write_index(
        index,
        str(index_path),
    )

    chunks_path.write_text(
        json.dumps(
            chunks,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    config = {
        "format_version": 1,
        "embedding_model": model_name,
        "model_max_tokens": int(max_seq_length),
        "embedding_dimension": int(vectors.shape[1]),
        "vector_count": int(index.ntotal),
        "index_type": "IndexFlatIP",
        "similarity": "cosine",
        "embeddings_normalized": True,
        "index_sha256": file_hash(index_path),
        "chunks_sha256": file_hash(chunks_path),
    }

    # Write the config last.
    # Hash checks detect incomplete or mixed builds.
    config_path = index_dir / "index_config.json"

    config_path.write_text(
        json.dumps(
            config,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    return config


def load_index(index_dir):
    index_dir = Path(index_dir)

    index_path = index_dir / "rag_index.faiss"
    chunks_path = index_dir / "chunks.json"
    config_path = index_dir / "index_config.json"

    for path in [
        index_path,
        chunks_path,
        config_path,
    ]:
        if not path.is_file():
            raise FileNotFoundError(f"Index file not found: {path}")

    config = json.loads(config_path.read_text(encoding="utf-8"))

    if config["format_version"] != 1:
        raise ValueError("Unsupported index format. Rebuild the index.")

    if file_hash(index_path) != config["index_sha256"]:
        raise ValueError("Index hash mismatch. Rebuild the index.")

    if file_hash(chunks_path) != config["chunks_sha256"]:
        raise ValueError("Chunk metadata hash mismatch. Rebuild the index.")

    index = faiss.read_index(str(index_path))

    chunks = json.loads(chunks_path.read_text(encoding="utf-8"))

    if index.ntotal != len(chunks) or index.ntotal != config["vector_count"]:
        raise ValueError("Index and chunk counts do not match.")

    if index.d != config["embedding_dimension"]:
        raise ValueError("Embedding dimension does not match the config.")

    if index.metric_type != faiss.METRIC_INNER_PRODUCT:
        raise ValueError("Unexpected FAISS similarity metric.")

    return index, chunks, config
