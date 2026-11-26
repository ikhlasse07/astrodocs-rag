"""Combine dense and lexical candidates before cross-encoder reranking."""

from collections import Counter
import math
import re
import unicodedata

import numpy as np

from app.rag.retriever import Retriever


def lexical_tokens(text):
    text = unicodedata.normalize("NFKC", text).casefold()
    return re.findall(r"[^\W_]+", text, flags=re.UNICODE)


class HybridRetriever(Retriever):
    def __init__(self, index_dir):
        super().__init__(index_dir)

        self.term_counts = [
            Counter(lexical_tokens(chunk["text"])) for chunk in self.chunks
        ]

        self.lengths = [sum(counts.values()) for counts in self.term_counts]

        self.mean_length = (sum(self.lengths) / len(self.lengths)) or 1.0

        frequencies = Counter()

        for counts in self.term_counts:
            frequencies.update(counts.keys())

        self.document_frequency = frequencies
        count = len(self.chunks)

        self.idf = {
            term: math.log(1 + (count - frequency + 0.5) / (frequency + 0.5))
            for term, frequency in frequencies.items()
        }

        self.last_candidate_count = 0

    def lexical_scores(self, question):
        scores = np.zeros(
            len(self.chunks),
            dtype=np.float64,
        )

        for term in set(lexical_tokens(question)):
            weight = self.idf.get(term)

            if weight is None:
                continue

            for position, counts in enumerate(self.term_counts):
                frequency = counts.get(term, 0)

                if frequency:
                    length_factor = 1.2 * (
                        0.25 + 0.75 * self.lengths[position] / self.mean_length
                    )

                    scores[position] += (
                        weight * frequency * 2.2 / (frequency + length_factor)
                    )

        return scores

    def retrieve(self, question, top_k=4, candidate_k=20):
        if not isinstance(question, str) or not question.strip():
            raise ValueError("Enter a non-empty question.")

        for name, value in (
            ("top_k", top_k),
            ("candidate_k", candidate_k),
        ):
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
            raise ValueError("The question exceeds the embedding model token limit.")

        query = self.embedder.encode(
            [question],
            show_progress_bar=False,
        )

        count = min(
            max(candidate_k, top_k),
            self.index.ntotal,
        )

        scores, positions = self.index.search(query, count)

        dense = {
            int(position): (rank, float(score))
            for rank, (score, position) in enumerate(
                zip(scores[0], positions[0]),
                start=1,
            )
            if position >= 0
        }

        lexical = self.lexical_scores(question)

        ranked_lexical = sorted(
            (int(position) for position in np.flatnonzero(lexical > 0)),
            key=lambda position: float(lexical[position]),
            reverse=True,
        )

        query_terms = set(lexical_tokens(question)) & self.idf.keys()

        anchors = set()

        if query_terms:
            rarest = min(self.document_frequency[term] for term in query_terms)

            anchors = {
                term for term in query_terms if self.document_frequency[term] == rarest
            }

        # Reserve rare-term matches inside the lexical budget.
        protected = [
            position
            for position in ranked_lexical
            if anchors & self.term_counts[position].keys()
        ][:count]

        protected_set = set(protected)

        lexical_order = (
            protected
            + [position for position in ranked_lexical if position not in protected_set]
        )[:count]

        lexical_rank = {
            position: rank
            for rank, position in enumerate(
                lexical_order,
                start=1,
            )
        }

        pool = list(dense)

        pool.extend(position for position in lexical_order if position not in dense)

        self.last_candidate_count = len(pool)

        candidates = []

        for position in pool:
            item = dict(self.chunks[position])

            dense_rank, similarity = dense.get(
                position,
                (None, None),
            )

            if similarity is None:
                similarity = float(
                    np.dot(
                        query[0],
                        self.index.reconstruct(position),
                    )
                )

            item.update(
                {
                    "score": similarity,
                    "dense_rank": dense_rank,
                    "lexical_rank": lexical_rank.get(position),
                    "lexical_score": float(lexical[position]),
                }
            )

            candidates.append(item)

        if not candidates:
            return []

        relevance = np.asarray(
            self.reranker.predict(
                [(question, item["text"]) for item in candidates],
                batch_size=16,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
        ).reshape(-1)

        if len(relevance) != len(candidates) or not np.isfinite(relevance).all():
            raise ValueError("The reranker returned invalid scores.")

        order = sorted(
            range(len(candidates)),
            key=lambda position: float(relevance[position]),
            reverse=True,
        )

        results = []

        for position in order[:top_k]:
            item = dict(candidates[position])

            item.update(
                rank=len(results) + 1,
                rerank_score=float(relevance[position]),
            )

            results.append(item)

        return results
