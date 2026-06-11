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


def _chunk_key(doc: Document) -> str:
    return doc.id or str(hash(doc.page_content[:100]))


def reciprocal_rank_fusion(
    ranked_lists: list[list[tuple[Document, float]]],
    k: int,
    rrf_constant: int = 60,
) -> list[tuple[Document, float]]:
    """Fuse multiple ranked lists with reciprocal rank fusion (scale-invariant)."""
    if not ranked_lists:
        return []

    scores: dict[str, float] = {}
    docs: dict[str, Document] = {}

    for ranked in ranked_lists:
        for rank, (doc, _) in enumerate(ranked, start=1):
            key = _chunk_key(doc)
            docs[key] = doc
            scores[key] = scores.get(key, 0.0) + 1.0 / (rrf_constant + rank)

    fused = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    return [(docs[key], score) for key, score in fused[:k]]


def merge_hybrid_results(
    vector_results: list[tuple[Document, float]],
    bm25_results: list[tuple[Document, float]],
    k: int,
    rrf_constant: int = 60,
) -> list[tuple[Document, float]]:
    return reciprocal_rank_fusion([vector_results, bm25_results], k, rrf_constant)


@dataclass
class ChunkRecord:
    chunk_id: str
    document_id: str
    filename: str
    page: int | None
    text: str


def _tokenize(text: str) -> list[str]:
    return [token for token in re.findall(r"\w+", text.lower()) if len(token) > 1]


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
        self._records_cache: list[ChunkRecord] | None = None
        self._cache_document_id: str | None = None

    async def _load_records(self, document_id: str | None = None) -> list[ChunkRecord]:
        if self._records_cache is not None and self._cache_document_id == document_id:
            return self._records_cache

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

        records = [
            ChunkRecord(
                chunk_id=row.id,
                document_id=row.document_id,
                filename=row.filename,
                page=row.page_number,
                text=row.text,
            )
            for row in rows
        ]
        self._records_cache = records
        self._cache_document_id = document_id
        return records

    def invalidate_cache(self) -> None:
        self._records_cache = None
        self._cache_document_id = None

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
