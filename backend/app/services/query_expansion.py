"""Universal query expansion for RAG (multi-query / RAG-Fusion pattern)."""

from __future__ import annotations

import asyncio
import logging
import re

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from app.config import Settings
from app.services.hybrid_search import is_arabic_question

logger = logging.getLogger(__name__)

LATIN_TOKEN_RE = re.compile(r"[a-zA-Z]{2,}")

CERT_INTENT_RE = re.compile(r"شهاد|ترخيص|certif|license", re.IGNORECASE)
SKILLS_INTENT_RE = re.compile(r"مهار|skills", re.IGNORECASE)
PROJECTS_INTENT_RE = re.compile(r"مشاريع|مشروع|projects", re.IGNORECASE)
EDUCATION_INTENT_RE = re.compile(r"تعليم|education", re.IGNORECASE)
EXPERIENCE_INTENT_RE = re.compile(r"خبرة|experience|عمل", re.IGNORECASE)
CONTACT_INTENT_RE = re.compile(
    r"تواصل|contact|email|phone|هاتف|بريد|linkedin", re.IGNORECASE
)

# Deterministic English retrieval queries when portfolio intent is known (no LLM).
PORTFOLIO_INTENT_SEARCH: list[tuple[re.Pattern[str], str]] = [
    (CERT_INTENT_RE, "Licenses Certifications Purple SystemVerilog"),
    (SKILLS_INTENT_RE, "Technical Skills Software Skills Hardware Skills Ladder VFD"),
    (
        PROJECTS_INTENT_RE,
        "EXPERIENCE WeFix Nuqayyem Quantalytics Olive Blinx 4Tech CSC Beyond",
    ),
    (EDUCATION_INTENT_RE, "Education"),
    (EXPERIENCE_INTENT_RE, "EXPERIENCE"),
    (CONTACT_INTENT_RE, "Contact LinkedIn email phone"),
]

ENGLISH_SEARCH_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Translate the user question into one short English search query "
            "for retrieving relevant resume/CV passages. "
            "Include section nouns when appropriate (certifications, skills, "
            "projects, experience, education). Output only the query.",
        ),
        ("human", "{question}"),
    ]
)

MULTI_QUERY_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You generate short search queries to retrieve relevant passages.\n\n"
            "Output exactly {count} queries, one per line:\n"
            "1. Concise rephrasing.\n"
            "2. Alternative phrasing with key entities.\n"
            "3. English query with section-relevant nouns when applicable "
            "(e.g. certifications, skills, projects, experience).\n"
            "No numbering or explanation. One query per line only.",
        ),
        ("human", "Question:\n{question}"),
    ]
)


def has_latin_tokens(text: str) -> bool:
    return bool(LATIN_TOKEN_RE.search(text))


def has_sufficient_latin_terms(text: str) -> bool:
    """Skip LLM expansion when the query already names entities in Latin."""
    tokens = LATIN_TOKEN_RE.findall(text)
    if len(tokens) >= 2:
        return True
    return len(tokens) == 1 and len(tokens[0]) >= 5


def resolve_portfolio_intent_search_query(question: str) -> str | None:
    """Return a deterministic English search query when question intent is known."""
    for pattern, search_query in PORTFOLIO_INTENT_SEARCH:
        if pattern.search(question):
            return search_query
    return None


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
            num_predict=64,
            num_ctx=512,
        ).bind(think=False)

    async def _english_search_query(self, question: str) -> str:
        chain = ENGLISH_SEARCH_PROMPT | self._llm
        response = await chain.ainvoke({"question": question})
        return (response.content or "").strip().splitlines()[0].strip("\"'")

    async def _llm_variations(self, question: str) -> list[str]:
        count = self.settings.multi_query_count
        chain = MULTI_QUERY_PROMPT | self._llm
        response = await chain.ainvoke({"question": question, "count": count})
        lines = [
            line.strip().lstrip("0123456789.-)• ").strip("\"'")
            for line in (response.content or "").splitlines()
            if line.strip()
        ]
        return lines[:count]

    def _expansion_enabled(self, portfolio_fast: bool) -> bool:
        if self.settings.query_expansion_enabled:
            return True
        return portfolio_fast and self.settings.portfolio_query_expansion_enabled

    async def extra_queries(
        self, question: str, portfolio_fast: bool = False
    ) -> list[str]:
        """Additional search queries beyond the original (may be empty)."""
        if not self._expansion_enabled(portfolio_fast):
            return []

        if has_sufficient_latin_terms(question):
            logger.debug("Skipping LLM query expansion: Latin terms present")
            return []

        if not is_arabic_question(question):
            return []

        timeout = (
            self.settings.portfolio_query_expansion_timeout_seconds
            if portfolio_fast and not self.settings.query_expansion_enabled
            else self.settings.query_expansion_timeout_seconds
        )
        try:
            return await asyncio.wait_for(self._llm_variations(question), timeout=timeout)
        except asyncio.TimeoutError:
            logger.warning(
                "Query expansion timed out after %.1fs; using original query only",
                timeout,
            )
            return []

    async def expand(self, question: str, portfolio_fast: bool = False) -> list[str]:
        """Return unique search queries: original + optional LLM variations."""
        original = question.strip()

        if portfolio_fast:
            intent_query = resolve_portfolio_intent_search_query(original)
            if intent_query:
                logger.debug(
                    "Using deterministic portfolio intent query (2-query cap)"
                )
                return _dedupe_queries([original, intent_query])[:2]

        extras = await self.extra_queries(original, portfolio_fast=portfolio_fast)
        queries = _dedupe_queries([original, *extras])

        if (
            portfolio_fast
            and is_arabic_question(original)
            and not any(has_latin_tokens(query) for query in queries)
        ):
            timeout = self.settings.portfolio_query_expansion_timeout_seconds
            try:
                english = await asyncio.wait_for(
                    self._english_search_query(original),
                    timeout=timeout,
                )
                if english:
                    queries = _dedupe_queries([*queries, english])
            except asyncio.TimeoutError:
                logger.warning("English search query timed out; using Arabic only")

        return queries

    def lexical_queries(self, queries: list[str]) -> list[str]:
        """Queries suitable for BM25 (must contain Latin tokens)."""
        lexical = [query for query in queries if has_latin_tokens(query)]
        if lexical:
            return _dedupe_queries(lexical)
        if is_arabic_question(queries[0]):
            return []
        return queries[:1]
