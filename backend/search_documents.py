"""Run retrieval with CrossEncoder reranking and save session results."""

import argparse
import json
from pathlib import Path
from time import perf_counter

from app.rag.retriever import Retriever


BACKEND_DIR = Path(__file__).resolve().parent


def run_search(
    question,
    retriever,
    top_k,
    candidate_k,
    output_path,
    history,
):
    start = perf_counter()

    results = retriever.retrieve(
        question,
        top_k=top_k,
        candidate_k=candidate_k,
    )

    elapsed = perf_counter() - start

    pool_size = min(
        max(candidate_k, top_k),
        retriever.index.ntotal,
    )

    print(f"\nQuestion: {question.strip()}")

    print(f"Search and reranking time: {elapsed:.3f} seconds")

    print(f"Candidate chunks: {pool_size}")
    print(f"Returned chunks: {len(results)}")

    for item in results:
        print(f"\n[{item['rank']}] Relevance score: {item['rerank_score']:.4f}")

        print(f"Semantic similarity: {item['score']:.4f}")

        print(f"Initial rank: {item['dense_rank']}")
        print(f"Source: {item['source']}")
        print(f"PDF page: {item['page']}")
        print(f"Chunk: {item['chunk_index']}")
        print(f"Text:\n{item['text']}")

    history.append(
        {
            "question": question.strip(),
            "candidate_count": pool_size,
            "requested_top_k": top_k,
            "returned_chunks": len(results),
            "retrieval_seconds": round(elapsed, 4),
            "reranker_model": retriever.reranker_name,
            "results": results,
        }
    )

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

    print(f"\nSaved session results: {output_path}")


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

    args = parser.parse_args()

    if args.top_k < 1 or args.candidate_k < 1:
        parser.error("--top-k and --candidate-k must be positive integers.")

    index_dir = BACKEND_DIR / "data" / "index"

    output_path = BACKEND_DIR / "data" / "processed" / "retrieval_reranked_results.json"

    print(
        "Loading retrieval models...",
        flush=True,
    )

    start = perf_counter()
    retriever = Retriever(index_dir)

    print(f"Retriever ready in {perf_counter() - start:.2f} seconds")

    print(f"Indexed chunks: {retriever.index.ntotal}")

    print(f"Candidate target: {args.candidate_k}")

    print(f"Results per question: {args.top_k}")

    history = []

    if args.question is not None:
        try:
            run_search(
                args.question,
                retriever,
                args.top_k,
                args.candidate_k,
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
            print("\nSearch session ended.")
            break

        if question.lower() in {"exit", "quit"}:
            print("Search session ended.")
            break

        if not question:
            continue

        try:
            run_search(
                question,
                retriever,
                args.top_k,
                args.candidate_k,
                output_path,
                history,
            )

        except ValueError as error:
            print(f"Input error: {error}")


if __name__ == "__main__":
    main()
