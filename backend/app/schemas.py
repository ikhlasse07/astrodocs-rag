"""Request and response models for the AstroDocs API."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class AskRequest(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        str_strip_whitespace=True,
        json_schema_extra={
            "examples": [
                {
                    "question": "What is the purpose of MOXIE?",
                    "top_k": 4,
                    "candidate_k": 20,
                }
            ]
        },
    )

    question: str = Field(
        min_length=3,
        max_length=500,
        description="The question to answer using the indexed PDFs.",
    )

    top_k: int = Field(
        default=4,
        ge=1,
        le=6,
        description="Maximum number of passages sent to the answer generator.",
    )

    candidate_k: int = Field(
        default=20,
        ge=1,
        le=50,
        description="Number of dense retrieval candidates to rerank.",
    )

    @model_validator(mode="after")
    def validate_candidate_count(self):
        if self.candidate_k < self.top_k:
            raise ValueError("candidate_k must be greater than or equal to top_k.")

        return self


class SourceResponse(BaseModel):
    source_id: int
    chunk_id: str
    source: str
    page: int
    chunk_index: int
    text: str
    score: float | None = None
    rerank_score: float | None = None


class AskResponse(BaseModel):
    question: str
    answer: str

    status: Literal[
        "verified_excerpt",
        "abstained",
        "needs_review",
    ]

    answer_mode: str

    evidence_verified: bool = Field(
        description=(
            "Displayed excerpts match their cited source text. "
            "This does not guarantee relevance or completeness."
        )
    )

    citation_format_ok: bool
    verification_reason: str
    relevance_action: str

    cited_sources: list[SourceResponse]
    retrieved_sources: list[SourceResponse]

    retrieval_seconds: float
    generation_seconds: float
    total_seconds: float

    reached_token_budget: bool


class HealthResponse(BaseModel):
    status: Literal["ok"]
    busy: bool
    indexed_chunks: int
    load_seconds: float
    embedding_model: str
    generation_model: str
    reranker_model: str
