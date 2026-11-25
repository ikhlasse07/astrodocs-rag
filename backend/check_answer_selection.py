"""Replay saved model selections without loading the language model."""

import json
from pathlib import Path

import torch
from sentence_transformers import CrossEncoder

from app.rag.generator import (
    add_previous_sentence,
    check_citations,
    parse_selection,
    screen_relevance,
    verify_selection,
)


BACKEND_DIR = Path(__file__).resolve().parent

ANSWERS_PATH = BACKEND_DIR / "data" / "processed" / "rag_answers.json"

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"


def main():
    if not ANSWERS_PATH.is_file():
        raise SystemExit(f"Saved answers were not found: {ANSWERS_PATH}")

    records = json.loads(ANSWERS_PATH.read_text(encoding="utf-8-sig"))

    if not isinstance(records, list) or not records:
        raise SystemExit("Expected a non-empty list of saved answers.")

    print("Loading the reranking model on CPU...", flush=True)

    reranker = CrossEncoder(
        RERANKER_MODEL,
        device="cpu",
        max_length=512,
        activation_fn=torch.nn.Identity(),
    )

    for record in records:
        question = record["question"]
        sources = record["retrieved_sources"]

        print(f"\nQuestion: {question}", flush=True)

        try:
            result = verify_selection(
                parse_selection(record["raw_generation"]),
                sources,
            )

            result = add_previous_sentence(
                question,
                result,
                sources,
            )

            result = screen_relevance(
                question,
                result,
                sources,
                reranker,
            )

            citations = check_citations(
                result["answer"],
                sources,
            )

        except ValueError as error:
            print(f"Replay failed: {error}")
            continue

        print(f"Answer: {result['answer']}")
        print(f"Status: {citations['status']}")
        print(f"Action: {result['relevance_action']}")
        print(
            "Decision reason: "
            f"{result.get('relevance_decision_reason', 'not_applicable')}"
        )
        print(f"Shared question terms: {result.get('shared_question_terms', [])}")

    print("\nReplay completed. Saved answer files were not modified.")


if __name__ == "__main__":
    main()
