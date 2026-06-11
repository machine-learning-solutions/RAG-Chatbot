"""Universal query expansion for RAG (multi-query / RAG-Fusion pattern)."""

from __future__ import annotations

import re

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from app.config import Settings
from app.services.hybrid_search import is_arabic_question

LATIN_TOKEN_RE = re.compile(r"[a-zA-Z]{2,}")

MULTI_QUERY_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You generate search queries to retrieve relevant passages from a document corpus.\n\n"
            "Given a user question, output exactly {count} search queries, one per line:\n"
            "1. A concise rephrasing of the question.\n"
            "2. An alternative phrasing emphasizing key entities and terms.\n"
            "3. An English query preserving all proper nouns (translate if needed).\n"
            "{extra_line}"
            "Rules:\n"
            "- One query per line only.\n"
            "- No numbering, bullets, or explanations.\n"
            "- Keep company names, project names, technologies, and dates intact.",
        ),
        ("human", "Question:\n{question}"),
    ]
)


def has_latin_tokens(text: str) -> bool:
    return bool(LATIN_TOKEN_RE.search(text))


def _dedupe_queries(queries: list[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for query in queries:
        normalized = query.strip().lower()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        unique.append(query.strip())
    return unique


class QueryExpander:
    """RAG-Fusion style multi-query expansion (domain-agnostic)."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._llm = ChatOllama(
            base_url=settings.ollama_base_url,
            model=settings.ollama_model,
            temperature=0.0,
            num_predict=128,
            num_ctx=1024,
        ).bind(think=False)

    async def expand(self, question: str) -> list[str]:
        """Return unique search queries: original + LLM-generated variations."""
        original = question.strip()
        queries = [original]

        if not self.settings.query_expansion_enabled:
            return queries

        count = self.settings.multi_query_count
        extra = ""
        if count > 3:
            extra = "4. A short keyword-style query.\n"

        chain = MULTI_QUERY_PROMPT | self._llm
        response = await chain.ainvoke(
            {"question": original, "count": count, "extra_line": extra}
        )
        lines = [
            line.strip().lstrip("0123456789.-)• ").strip("\"'")
            for line in (response.content or "").splitlines()
            if line.strip()
        ]
        queries.extend(lines[:count])
        return _dedupe_queries(queries)

    def lexical_queries(self, queries: list[str]) -> list[str]:
        """Queries suitable for BM25 (must contain Latin tokens)."""
        lexical = [query for query in queries if has_latin_tokens(query)]
        if lexical:
            return _dedupe_queries(lexical)
        if is_arabic_question(queries[0]):
            return []
        return queries[:1]
