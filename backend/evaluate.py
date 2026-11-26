"""Evaluate the running API and save evidence for manual answer review."""

import argparse
import csv
import hashlib
import json
import platform
import re
import statistics
import unicodedata
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from time import perf_counter
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


BACKEND_DIR = Path(__file__).resolve().parent
DATASET_PATH = BACKEND_DIR / "data" / "evaluation" / "questions.json"
RESULTS_DIR = BACKEND_DIR / "data" / "evaluation" / "results"
MANUAL_FIELDS = ("answer_score", "citations_relevant", "notes")


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def file_hash(path):
    if not path.is_file():
        return None

    digest = hashlib.sha256()

    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def snapshot():
    paths = (
        "app/rag/generator.py",
        "app/rag/retriever.py",
        "app/rag/pipeline.py",
        "app/rag/embedder.py",
        "app/rag/vector_store.py",
        "app/main.py",
        "app/schemas.py",
        "data/index/index_config.json",
        "data/index/rag_index.faiss",
        "data/index/chunks.json",
        "evaluate.py",
        "app/rag/hybrid_retriever.py",
        "app/rag/evidence_selector.py",
    )

    packages = {}

    for name in (
        "fastapi",
        "pydantic",
        "torch",
        "transformers",
        "sentence-transformers",
        "faiss-cpu",
        "numpy",
    ):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None

    return {
        "files": {name: file_hash(BACKEND_DIR / name) for name in paths},
        "python": platform.python_version(),
        "packages": packages,
    }


def api_json(url, payload=None, timeout=180):
    body = None if payload is None else json.dumps(payload).encode("utf-8")

    headers = {"Accept": "application/json"}

    if body is not None:
        headers["Content-Type"] = "application/json"

    request = Request(url, data=body, headers=headers)

    try:
        with urlopen(request, timeout=timeout) as response:
            return json.load(response)

    except HTTPError as error:
        detail = error.read().decode(
            "utf-8",
            errors="replace",
        )[:400]

        raise RuntimeError(f"HTTP {error.code}: {detail}") from error

    except (URLError, TimeoutError, OSError) as error:
        raise RuntimeError(f"API connection failed: {error}") from error


def normalize(text):
    return re.sub(
        r"\s+",
        " ",
        unicodedata.normalize("NFC", text),
    ).strip()


def literal_support(result):
    """Check quotation membership, not relevance or correctness."""
    if result["status"] == "abstained":
        return None

    text = result["answer"].strip()

    sources = {source["source_id"]: source for source in result["cited_sources"]}

    matches = list(
        re.finditer(
            r"(.+?)\s+\[(\d+)\](?=\s|$)",
            text,
            re.DOTALL,
        )
    )

    if not matches or text[matches[-1].end() :].strip():
        return False

    for match in matches:
        source = sources.get(int(match.group(2)))
        quote = normalize(match.group(1))

        if not source or not quote or quote not in normalize(source["text"]):
            return False

    return True


def page_hit(sources, case):
    if not case["answerable"]:
        return None

    return any(
        source["source"] == case["expected_source"]
        and source["page"] == case["expected_page"]
        for source in sources
    )


def save_run(run, path):
    """Checkpoint after each answer and preserve CSV review fields."""
    temporary = path.with_suffix(".tmp")

    temporary.write_text(
        json.dumps(run, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    temporary.replace(path)

    csv_path = path.with_suffix(".csv")
    manual = {}

    if csv_path.is_file():
        with csv_path.open(
            encoding="utf-8-sig",
            newline="",
        ) as stream:
            for row in csv.DictReader(stream):
                manual[row["id"]] = {key: row.get(key, "") for key in MANUAL_FIELDS}

    columns = [
        "id",
        "question",
        "answerable",
        "expected_answer",
        "expected_source",
        "expected_page",
        "reference_quote",
        "answer",
        "status",
        "reference_page_retrieved",
        "reference_page_cited",
        "literal_support_ok",
        "total_seconds",
        *MANUAL_FIELDS,
    ]

    with csv_path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=columns,
        )

        writer.writeheader()

        for record in run["results"]:
            result = record["api_response"]

            row = {key: record.get(key, "") for key in columns}

            row.update(
                {
                    "answer": result["answer"],
                    "status": result["status"],
                    "total_seconds": result.get("total_seconds", ""),
                }
            )

            row.update(manual.get(record["id"], {}))
            writer.writerow(row)


def summarize(run):
    records = run["results"]

    positive = [record for record in records if record["answerable"]]

    negative = [record for record in records if not record["answerable"]]

    answered = [
        record for record in records if record["api_response"]["status"] != "abstained"
    ]

    times = [record["api_response"]["total_seconds"] for record in records]

    retrieved_count = sum(record["reference_page_retrieved"] for record in positive)

    negative_abstained = sum(
        record["api_response"]["status"] == "abstained" for record in negative
    )

    positive_abstained = sum(
        record["api_response"]["status"] == "abstained" for record in positive
    )

    supported_count = sum(bool(record["literal_support_ok"]) for record in answered)

    return {
        "completed_cases": len(records),
        "reference_page_retrieved": (f"{retrieved_count}/{len(positive)}"),
        "unanswerable_cases_abstained": (f"{negative_abstained}/{len(negative)}"),
        "answerable_cases_abstained": (f"{positive_abstained}/{len(positive)}"),
        "literal_support_among_non_abstained": (f"{supported_count}/{len(answered)}"),
        "mean_api_seconds": (round(statistics.mean(times), 3) if times else None),
        "request_errors": len(run["errors"]),
        "answer_correctness": "Manual review required.",
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8000",
    )
    parser.add_argument("--run-name", default="baseline_v1")
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--candidate-k", type=int, default=20)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--resume", action="store_true")

    args = parser.parse_args()

    if not re.fullmatch(
        r"[A-Za-z0-9_-]{1,60}",
        args.run_name,
    ):
        raise SystemExit("Use letters, digits, underscores, or hyphens for run-name.")

    if (
        not 1 <= args.top_k <= 6
        or not args.top_k <= args.candidate_k <= 50
        or args.timeout < 1
    ):
        raise SystemExit("Invalid retrieval settings or timeout.")

    if not DATASET_PATH.is_file():
        raise SystemExit(f"Dataset not found: {DATASET_PATH}")

    dataset = json.loads(DATASET_PATH.read_text(encoding="utf-8-sig"))
    cases = dataset["cases"]

    if not cases or len({case["id"] for case in cases}) != len(cases):
        raise SystemExit("The dataset must contain cases with unique IDs.")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output = RESULTS_DIR / f"{args.run_name}.json"

    if output.exists() and not args.resume:
        raise SystemExit("Run already exists. Use --resume or a new --run-name.")

    if args.resume and not output.exists():
        raise SystemExit("There is no saved run to resume.")

    base_url = args.base_url.rstrip("/")

    print("Checking the running API...", flush=True)

    try:
        health = api_json(
            f"{base_url}/health",
            timeout=15,
        )
    except RuntimeError as error:
        raise SystemExit(f"Start the FastAPI backend first. {error}") from error

    if health.get("status") != "ok" or health.get("busy"):
        raise SystemExit("The API is not ready or is busy. Try again later.")

    settings = {
        "base_url": base_url,
        "top_k": args.top_k,
        "candidate_k": args.candidate_k,
    }

    environment = snapshot()

    models = {
        key: health[key]
        for key in (
            "embedding_model",
            "generation_model",
            "reranker_model",
        )
    }

    if args.resume:
        run = json.loads(output.read_text(encoding="utf-8"))

        if (
            run["dataset_sha256"] != file_hash(DATASET_PATH)
            or run["settings"] != settings
            or run["snapshot"] != environment
            or run["models"] != models
            or run["health"]["indexed_chunks"] != health["indexed_chunks"]
        ):
            raise SystemExit(
                "Dataset, code, index, models, or settings changed. Use a new run-name."
            )

    else:
        run = {
            "run_name": args.run_name,
            "created_at_utc": utc_now(),
            "dataset_name": dataset["name"],
            "dataset_sha256": file_hash(DATASET_PATH),
            "settings": settings,
            "models": models,
            "health": health,
            "snapshot": environment,
            "results": [],
            "errors": [],
        }

    save_run(run, output)

    done = {record["id"] for record in run["results"]}

    for position, case in enumerate(cases, start=1):
        if case["id"] in done:
            continue

        print(
            f"\n[{position}/{len(cases)}] {case['id']}: {case['question']}",
            flush=True,
        )

        started = perf_counter()

        try:
            result = api_json(
                f"{base_url}/ask",
                {
                    "question": case["question"],
                    "top_k": args.top_k,
                    "candidate_k": args.candidate_k,
                },
                timeout=args.timeout,
            )

            for key in (
                "answer",
                "status",
                "cited_sources",
                "retrieved_sources",
                "total_seconds",
            ):
                if key not in result:
                    raise RuntimeError(f"API response is missing: {key}")

        except (
            RuntimeError,
            ValueError,
            KeyboardInterrupt,
        ) as error:
            run["errors"].append(
                {
                    "id": case["id"],
                    "at_utc": utc_now(),
                    "message": str(error) or "Interrupted by user.",
                }
            )

            save_run(run, output)

            raise SystemExit(
                f"Stopped; progress saved. {error}\nResume after the API is ready."
            ) from error

        record = {
            **case,
            "client_seconds": round(
                perf_counter() - started,
                4,
            ),
            "reference_page_retrieved": page_hit(
                result["retrieved_sources"],
                case,
            ),
            "reference_page_cited": page_hit(
                result["cited_sources"],
                case,
            ),
            "literal_support_ok": literal_support(result),
            "api_response": result,
        }

        run["results"].append(record)
        run["updated_at_utc"] = utc_now()

        save_run(run, output)

        print(
            f"Status: {result['status']} | Time: {result['total_seconds']:.2f}s",
            flush=True,
        )
        print(f"Answer: {result['answer']}", flush=True)

    run["summary"] = summarize(run)
    save_run(run, output)

    print("\nAutomatic checks (not answer accuracy):")
    print(json.dumps(run["summary"], indent=2))

    print(f"\nSaved: {output}\nSaved: {output.with_suffix('.csv')}")


if __name__ == "__main__":
    main()
