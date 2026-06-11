"""Post-retrieval context filtering (domain-agnostic)."""

from __future__ import annotations

from langchain_core.documents import Document

from app.config import Settings


def filter_by_absolute_score(
    chunks: list[tuple[Document, float]],
    min_score: float,
) -> list[tuple[Document, float]]:
    if not chunks:
        return chunks
    filtered = [(doc, score) for doc, score in chunks if score >= min_score]
    return filtered or [chunks[0]]


def filter_by_relative_score(
    chunks: list[tuple[Document, float]],
    margin: float,
) -> list[tuple[Document, float]]:
    """Keep chunks within `margin` of the top reranker score."""
    if not chunks:
        return chunks
    top_score = chunks[0][1]
    filtered = [(doc, score) for doc, score in chunks if score >= top_score - margin]
    return filtered or [chunks[0]]


def apply_context_filters(
    chunks: list[tuple[Document, float]],
    settings: Settings,
) -> list[tuple[Document, float]]:
    if not chunks:
        return chunks

    filtered = chunks
    if settings.reranker_min_score is not None:
        filtered = filter_by_absolute_score(filtered, settings.reranker_min_score)
    if settings.context_score_margin > 0:
        filtered = filter_by_relative_score(filtered, settings.context_score_margin)
    return filtered[: settings.max_context_chunks]
