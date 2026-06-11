"""Backward-compatible wrapper around universal query expansion."""

from __future__ import annotations

from app.config import Settings
from app.services.query_expansion import QueryExpander, has_latin_tokens

__all__ = ["QueryReformulator", "has_latin_tokens"]


class QueryReformulator:
    def __init__(self, settings: Settings) -> None:
        self._expander = QueryExpander(settings)

    async def retrieval_queries(self, question: str) -> list[str]:
        return await self._expander.expand(question)
