"""FastAPI application for AstroDocs RAG."""

import gc
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from threading import Lock
from time import perf_counter

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool

from app.rag.pipeline import RAGPipeline
from app.schemas import (
    AskRequest,
    AskResponse,
    HealthResponse,
    SourceResponse,
)


BACKEND_DIR = Path(__file__).resolve().parents[1]
INDEX_DIR = BACKEND_DIR / "data" / "index"

logger = logging.getLogger("uvicorn.error")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load shared resources before accepting requests."""
    app.state.pipeline = None
    app.state.inference_lock = Lock()
    app.state.load_seconds = 0.0

    required_files = (
        "rag_index.faiss",
        "chunks.json",
        "index_config.json",
    )

    for filename in required_files:
        path = INDEX_DIR / filename

        if not path.is_file():
            raise FileNotFoundError(
                f"Missing index file: {path}. "
                "Run backend/build_index.py before starting the API."
            )

    logger.info("Loading AstroDocs RAG...")
    started = perf_counter()

    try:
        app.state.pipeline = await run_in_threadpool(
            RAGPipeline,
            INDEX_DIR,
        )

        app.state.load_seconds = round(
            perf_counter() - started,
            4,
        )

        logger.info(
            "AstroDocs RAG is ready in %.2f seconds.",
            app.state.load_seconds,
        )

        yield

    finally:
        app.state.pipeline = None
        gc.collect()
        logger.info("AstroDocs RAG resources released.")


app = FastAPI(
    title="AstroDocs RAG API",
    description=(
        "Answer questions using indexed astronomy PDFs and source-verified excerpts."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:4173",
        "http://127.0.0.1:4173",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
    expose_headers=["Retry-After"],
)


def get_pipeline(request: Request) -> RAGPipeline:
    pipeline = getattr(
        request.app.state,
        "pipeline",
        None,
    )

    if pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="The RAG pipeline is not ready.",
        )

    return pipeline


@app.get("/", tags=["System"])
def root():
    return {
        "name": "AstroDocs RAG API",
        "version": "0.1.0",
        "docs": "/docs",
        "health": "/health",
    }


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["System"],
)
def health(request: Request):
    pipeline = get_pipeline(request)

    return HealthResponse(
        status="ok",
        busy=request.app.state.inference_lock.locked(),
        indexed_chunks=int(pipeline.retriever.index.ntotal),
        load_seconds=request.app.state.load_seconds,
        embedding_model=pipeline.retriever.config["embedding_model"],
        generation_model=pipeline.generator.model_name,
        reranker_model=pipeline.retriever.reranker_name,
    )


@app.post(
    "/ask",
    response_model=AskResponse,
    tags=["Questions"],
    responses={
        422: {"description": "Invalid question or retrieval settings."},
        503: {"description": "The pipeline is unavailable or busy."},
        500: {"description": "An unexpected processing error occurred."},
    },
)
def ask(payload: AskRequest, request: Request):
    """Process one question at a time using the shared pipeline."""
    pipeline = get_pipeline(request)
    inference_lock = request.app.state.inference_lock

    if not inference_lock.acquire(blocking=False):
        raise HTTPException(
            status_code=503,
            detail=(
                "The system is processing another question. Please try again shortly."
            ),
            headers={"Retry-After": "5"},
        )

    started = perf_counter()

    try:
        try:
            result = pipeline.ask(
                question=payload.question,
                top_k=payload.top_k,
                candidate_k=payload.candidate_k,
            )
        except ValueError as error:
            raise HTTPException(
                status_code=422,
                detail=str(error),
            ) from error

        cited_sources = [
            SourceResponse.model_validate(source) for source in result["cited_sources"]
        ]

        retrieved_sources = [
            SourceResponse.model_validate(source)
            for source in result["retrieved_sources"]
        ]

        return AskResponse(
            question=result["question"],
            answer=result["answer"],
            status=result["status"],
            answer_mode=result["answer_mode"],
            evidence_verified=result["evidence_verified"],
            citation_format_ok=result["citation_format_ok"],
            verification_reason=result["verification_reason"],
            relevance_action=result.get(
                "relevance_action",
                "not_checked",
            ),
            cited_sources=cited_sources,
            retrieved_sources=retrieved_sources,
            retrieval_seconds=result["retrieval_seconds"],
            generation_seconds=result["generation_seconds"],
            total_seconds=round(
                perf_counter() - started,
                4,
            ),
            reached_token_budget=result["reached_token_budget"],
        )

    except HTTPException:
        raise

    except Exception as error:
        logger.exception("Question processing failed.")

        raise HTTPException(
            status_code=500,
            detail=("Question processing failed. Check the server logs for details."),
        ) from error

    finally:
        inference_lock.release()
