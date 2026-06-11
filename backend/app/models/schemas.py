from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class DocumentInfo(BaseModel):
    id: str
    filename: str
    file_type: str
    chunk_count: int
    created_at: datetime


class DocumentListResponse(BaseModel):
    documents: list[DocumentInfo]
    total: int


class IngestResponse(BaseModel):
    document_id: str
    filename: str
    chunks_created: int
    message: str


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=4000)
    language: str | None = Field(
        default=None,
        description="Response language: ar or en (from OLD session language switcher)",
    )
    document_id: str | None = Field(
        default=None,
        description="Restrict retrieval to a specific document",
    )
    top_k: int | None = Field(default=None, ge=1, le=20)
    use_reranker: bool | None = None
    use_hybrid: bool | None = Field(
        default=None,
        description="TurboVec + BM25 hybrid search (OLD pipeline, without FAISS)",
    )
    portfolio_fast: bool = Field(
        default=False,
        description="Low-latency profile for public portfolio embeds",
    )
    stream: bool = Field(
        default=False,
        description="Stream SSE pings while processing (for Netlify/proxy timeouts)",
    )


class SourceChunk(BaseModel):
    chunk_id: str
    document_id: str
    filename: str
    page: int | None = None
    score: float
    text: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceChunk]
    model: str


class HealthResponse(BaseModel):
    status: str
    ollama: dict[str, Any]
    vector_store: dict[str, Any]
    database: str
