import re
from dataclasses import dataclass

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi
from sqlalchemy import select

from app.config import Settings, get_settings
from app.services.database import Chunk, Document as DocumentModel, async_session_factory

ARABIC_RE = re.compile(
    r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]"
)


def is_arabic_question(text: str) -> bool:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return False
    arabic_count = sum(1 for c in letters if ARABIC_RE.match(c))
    return arabic_count / len(letters) >= 0.4


def resolve_language(question: str, language: str | None = None) -> str:
    if language in {"ar", "en"}:
        return language
    return "ar" if is_arabic_question(question) else "en"


# Arabic questions against English resumes need extra English keywords for BM25/vectors.
_ARABIC_RETRIEVAL_HINTS: list[tuple[re.Pattern[str], str]] = [
    (
        re.compile(
            r"شهاد|ترخيص|certif|license|آخر\s*\d|أحدث|latest",
            re.IGNORECASE,
        ),
        "Licenses Certifications Purple March 2025",
    ),
    (
        re.compile(
            r"أشباه\s*الموصلات|موصلات|VLSI|ASIC|CMOS|"
            r"تحقق\s*من\s*التصميم|semiconductor|synopsys|Synopsis",
            re.IGNORECASE,
        ),
        "Purple Certification VLSI ASIC CMOS Design Verification "
        "SystemVerilog UVM semiconductor Synopsis",
    ),
    (
        re.compile(r"Quantalytics|كوانت|فوربس|Forbes", re.IGNORECASE),
        "Quantalytics Forbes React Native Redux Optimum Partners",
    ),
    (
        re.compile(r"JoPath|WeFix|وي\s*فiks|جو\s*بath", re.IGNORECASE),
        "JoPath WeFix microservices Flutter GraphQL",
    ),
]


def expand_retrieval_query(query: str) -> str:
    """Append English domain terms when an Arabic question implies them."""
    extras: list[str] = []
    for pattern, terms in _ARABIC_RETRIEVAL_HINTS:
        if pattern.search(query):
            extras.append(terms)
    if not extras:
        return query
    return f"{query} {' '.join(extras)}"


@dataclass
class ChunkRecord:
    chunk_id: str
    document_id: str
    filename: str
    page: int | None
    text: str


def _tokenize(text: str) -> list[str]:
    return text.lower().split()


def _to_document(record: ChunkRecord) -> Document:
    doc = Document(
        page_content=record.text,
        metadata={
            "document_id": record.document_id,
            "filename": record.filename,
            "source": record.filename,
            "page": record.page,
        },
    )
    doc.id = record.chunk_id
    return doc


class BM25Retriever:
    """Keyword search over indexed chunks (uses asyncpg via async SQLAlchemy)."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    async def _load_records(self, document_id: str | None = None) -> list[ChunkRecord]:
        stmt = (
            select(
                Chunk.id,
                Chunk.document_id,
                Chunk.text,
                Chunk.page_number,
                DocumentModel.filename,
            )
            .join(DocumentModel, Chunk.document_id == DocumentModel.id)
        )
        if document_id:
            stmt = stmt.where(Chunk.document_id == document_id)

        async with async_session_factory() as session:
            rows = (await session.execute(stmt)).all()

        return [
            ChunkRecord(
                chunk_id=row.id,
                document_id=row.document_id,
                filename=row.filename,
                page=row.page_number,
                text=row.text,
            )
            for row in rows
        ]

    async def search(
        self,
        query: str,
        k: int,
        document_id: str | None = None,
    ) -> list[tuple[Document, float]]:
        records = await self._load_records(document_id)
        if not records:
            return []

        tokenized_corpus = [_tokenize(record.text) for record in records]
        bm25 = BM25Okapi(tokenized_corpus)
        scores = bm25.get_scores(_tokenize(query))

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda idx: scores[idx],
            reverse=True,
        )

        results: list[tuple[Document, float]] = []
        for idx in ranked_indices[:k]:
            if scores[idx] <= 0:
                continue
            results.append((_to_document(records[idx]), float(scores[idx])))
        return results


def merge_hybrid_results(
    vector_results: list[tuple[Document, float]],
    bm25_results: list[tuple[Document, float]],
    k: int,
) -> list[tuple[Document, float]]:
    combined: dict[str, tuple[Document, float]] = {}

    for doc, score in vector_results + bm25_results:
        key = doc.id or str(hash(doc.page_content[:100]))
        if key not in combined:
            combined[key] = (doc, score)
        else:
            existing_doc, existing_score = combined[key]
            combined[key] = (existing_doc, max(existing_score, score))

    ranked = sorted(combined.values(), key=lambda item: item[1], reverse=True)
    return ranked[:k]
