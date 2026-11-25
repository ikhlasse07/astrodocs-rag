"""Connect retrieval, excerpt generation, and citation verification."""

from time import perf_counter

from app.rag.generator import AnswerGenerator, check_citations
from app.rag.retriever import Retriever


class RAGPipeline:
    def __init__(self, index_dir):
        print("Loading the retriever...", flush=True)
        self.retriever = Retriever(index_dir)

        print("Loading the answer generator on CPU...", flush=True)
        self.generator = AnswerGenerator(reranker=self.retriever.reranker)

    def ask(self, question, top_k=4, candidate_k=20):
        if not isinstance(question, str) or not question.strip():
            raise ValueError("The question must be a non-empty string.")

        question = question.strip()

        retrieval_start = perf_counter()

        passages = self.retriever.retrieve(
            question,
            top_k=top_k,
            candidate_k=candidate_k,
        )

        sources = [
            {**passage, "source_id": position}
            for position, passage in enumerate(passages, start=1)
        ]

        retrieval_seconds = perf_counter() - retrieval_start

        generation_start = perf_counter()

        generated = self.generator.generate(
            question,
            sources,
        )

        generation_seconds = perf_counter() - generation_start

        citations = check_citations(
            generated["answer"],
            sources,
        )

        cited_ids = set(citations["cited_source_ids"])

        return {
            "question": question,
            **generated,
            **citations,
            "generation_model": self.generator.model_name,
            "embedding_model": self.retriever.config["embedding_model"],
            "reranker_model": self.retriever.reranker_name,
            "retrieval_seconds": round(retrieval_seconds, 4),
            "generation_seconds": round(generation_seconds, 4),
            "cited_sources": [
                source for source in sources if source["source_id"] in cited_ids
            ],
            "retrieved_sources": sources,
        }
