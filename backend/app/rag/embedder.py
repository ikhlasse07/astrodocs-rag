"""Create normalized embeddings without silently truncating input text."""

import numpy as np
from sentence_transformers import SentenceTransformer


DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


class Embedder:
    def __init__(self, model_name=DEFAULT_MODEL):
        self.model_name = model_name

        self.model = SentenceTransformer(
            model_name,
            device="cpu",
        )

        self.max_seq_length = self.model.max_seq_length

        self.dimension = self.model.get_embedding_dimension()

    def encode(
        self,
        texts,
        batch_size=16,
        show_progress_bar=True,
    ):
        if not texts or any(
            not isinstance(text, str) or not text.strip() for text in texts
        ):
            raise ValueError("Expected a non-empty list of non-empty text strings.")

        tokens = self.model.tokenizer(
            texts,
            add_special_tokens=True,
            truncation=False,
            padding=False,
            verbose=False,
        )["input_ids"]

        for position, token_ids in enumerate(tokens):
            if len(token_ids) > self.max_seq_length:
                raise ValueError(
                    f"Text at position {position} has "
                    f"{len(token_ids)} tokens; "
                    f"the model limit is {self.max_seq_length}. "
                    "Rebuild the chunks."
                )

        vectors = self.model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=show_progress_bar,
        )

        vectors = np.ascontiguousarray(
            vectors,
            dtype=np.float32,
        )

        if vectors.shape != (len(texts), self.dimension):
            raise ValueError("Unexpected embedding matrix shape.")

        if not np.isfinite(vectors).all():
            raise ValueError("Embeddings contain non-finite values.")

        norms = np.linalg.norm(vectors, axis=1)

        if not np.allclose(norms, 1.0, atol=1e-4):
            raise ValueError("Embeddings must have unit length.")

        return vectors
