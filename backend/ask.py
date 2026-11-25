"""Ask questions and receive generated answers with PDF references."""

import argparse
import json
from pathlib import Path
from time import perf_counter

from app.rag.pipeline import RAGPipeline


BACKEND_DIR = Path(__file__).resolve().parent


def run_question(
    question,
    pipeline,
    args,
    output_path,
    history,
):
    result = pipeline.ask(
        question,
        top_k=args.top_k,
        candidate_k=args.candidate_k,
    )

    print(f"\nQuestion: {result['question']}")
    print(f"Answer: {result['answer']}")
    print(f"Status: {result['status']}")

    print(f"Retrieval time: {result['retrieval_seconds']:.3f} seconds")

    print(f"Generation time: {result['generation_seconds']:.3f} seconds")

    print("\nCited sources:")

    if not result["cited_sources"]:
        print("No cited sources.")

    for source in result["cited_sources"]:
        print(
            f"[{source['source_id']}] "
            f"{source['source']} | "
            f"PDF page {source['page']} | "
            f"Chunk {source['chunk_index']}"
        )

    if not result["citation_format_ok"]:
        print("Review note: citation labels are missing or invalid.")

    if result["reached_token_budget"]:
        print("Review note: the answer reached the token budget. Check completeness.")

    if args.show_passages:
        print("\nRetrieved passages:")

        for source in result["retrieved_sources"]:
            print(
                f"\n[{source['source_id']}] "
                f"{source['source']} | "
                f"PDF page {source['page']}"
            )

            print(source["text"])

    history.append(result)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        json.dumps(
            history,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"\nSaved session answers: {output_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--question",
        help="Run one question and exit.",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--candidate-k",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--show-passages",
        action="store_true",
    )

    args = parser.parse_args()

    if args.top_k < 1 or args.candidate_k < 1:
        parser.error("--top-k and --candidate-k must be positive integers.")

    print(
        "Starting AstroDocs RAG...",
        flush=True,
    )

    start = perf_counter()

    pipeline = RAGPipeline(BACKEND_DIR / "data" / "index")

    print(
        f"Pipeline ready in {perf_counter() - start:.2f} seconds",
        flush=True,
    )

    output_path = BACKEND_DIR / "data" / "processed" / "rag_answers.json"

    history = []

    if args.question is not None:
        try:
            run_question(
                args.question,
                pipeline,
                args,
                output_path,
                history,
            )

        except ValueError as error:
            raise SystemExit(str(error)) from error

        return

    print("\nEnter a question. Type exit or quit to stop.")

    while True:
        try:
            question = input("\nQuestion: ").strip()

        except (EOFError, KeyboardInterrupt):
            print("\nQuestion-answering session ended.")
            break

        if question.lower() in {"exit", "quit"}:
            print("Question-answering session ended.")
            break

        if not question:
            continue

        try:
            run_question(
                question,
                pipeline,
                args,
                output_path,
                history,
            )

        except ValueError as error:
            print(f"Input error: {error}")


if __name__ == "__main__":
    main()
