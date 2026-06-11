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
    min_keep: int = 3,
) -> list[tuple[Document, float]]:
    """Keep chunks within `margin` of the top reranker score."""
    if not chunks:
        return chunks
    top_score = chunks[0][1]
    filtered = [(doc, score) for doc, score in chunks if score >= top_score - margin]
    if len(filtered) >= min_keep:
        return filtered
    return chunks[:min_keep]


def apply_context_filters(
    chunks: list[tuple[Document, float]],
    settings: Settings,
    portfolio_fast: bool = False,
) -> list[tuple[Document, float]]:
    if not chunks:
        return chunks

    max_chunks = (
        settings.portfolio_max_context_chunks
        if portfolio_fast
        else settings.max_context_chunks
    )
    min_keep = min(max_chunks, len(chunks))

    filtered = chunks
    if settings.reranker_min_score is not None:
        absolute = filter_by_absolute_score(filtered, settings.reranker_min_score)
        if len(absolute) >= min_keep:
            filtered = absolute
    if settings.context_score_margin > 0:
        relative = filter_by_relative_score(
            filtered, settings.context_score_margin, min_keep=min_keep
        )
        if len(relative) >= min_keep:
            filtered = relative
    return filtered[:max_chunks]
