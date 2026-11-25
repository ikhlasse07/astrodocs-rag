"""Retrieve FAISS candidates and rerank them with a CrossEncoder."""

import numpy as np
from sentence_transformers import CrossEncoder
import torch

from app.rag.embedder import Embedder
from app.rag.vector_store import load_index


RERANKER_NAME = "cross-encoder/ms-marco-MiniLM-L6-v2"


class Retriever:
    def __init__(self, index_dir):
        self.index, self.chunks, self.config = load_index(index_dir)

        if self.index.ntotal == 0:
            raise ValueError("The index is empty. Build the index first.")

        if not self.config.get("embeddings_normalized"):
            raise ValueError("Expected normalized document embeddings.")

        self.embedder = Embedder(self.config["embedding_model"])

        if self.embedder.dimension != self.index.d:
            raise ValueError("Embedding dimensions differ. Rebuild the index.")

        if self.embedder.max_seq_length != self.config["model_max_tokens"]:
            raise ValueError("Model token limits differ. Rebuild the index.")

        self.reranker_name = RERANKER_NAME

        print(
            "Loading the reranking model on CPU...",
            flush=True,
        )

        self.reranker = CrossEncoder(
            self.reranker_name,
            device="cpu",
            max_length=512,
            activation_fn=torch.nn.Identity(),
        )

    def retrieve(
        self,
        question,
        top_k=4,
        candidate_k=20,
    ):
        if not isinstance(question, str) or not question.strip():
            raise ValueError("Enter a non-empty question.")

        for name, value in [
            ("top_k", top_k),
            ("candidate_k", candidate_k),
        ]:
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")

        question = question.strip()

        tokens = self.embedder.model.tokenizer.encode(
            question,
            add_special_tokens=True,
            truncation=False,
            verbose=False,
        )

        if len(tokens) > self.embedder.max_seq_length:
            raise ValueError(
                f"Question has {len(tokens)} tokens; "
                f"the limit is {self.embedder.max_seq_length}. "
                "Shorten the question."
            )

        query = self.embedder.encode(
            [question],
            show_progress_bar=False,
        )

        pool_size = min(
            max(candidate_k, top_k),
            self.index.ntotal,
        )

        scores, positions = self.index.search(
            query,
            pool_size,
        )

        candidates = []

        for dense_rank, (score, position) in enumerate(
            zip(scores[0], positions[0]),
            start=1,
        ):
            if position < 0:
                continue

            item = dict(self.chunks[int(position)])

            item["score"] = float(score)
            item["dense_rank"] = dense_rank

            candidates.append(item)

        if not candidates:
            return []

        pairs = [(question, item["text"]) for item in candidates]

        relevance = np.asarray(
            self.reranker.predict(
                pairs,
                batch_size=16,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
        ).reshape(-1)

        if len(relevance) != len(candidates) or not np.isfinite(relevance).all():
            raise ValueError("The reranker returned invalid scores.")

        order = sorted(
            range(len(candidates)),
            key=lambda i: float(relevance[i]),
            reverse=True,
        )

        results = []

        for i in order[:top_k]:
            item = dict(candidates[i])

            item["rank"] = len(results) + 1
            item["rerank_score"] = float(relevance[i])

            results.append(item)

        return results
