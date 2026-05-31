from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from app.config import Settings
from app.models.schemas import SourceChunk
from app.services.hybrid_search import BM25Retriever, merge_hybrid_results
from app.services.language import resolve_language
from app.services.reranker import Reranker
from app.services.vector_store import VectorStoreManager

# Arabic prompt adapted from Chatbot - OLD (rag_pipeline.py) + strict language rules
ARABIC_RAG_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "أنت مساعد ذكي متخصص في الإجابة على الأسئلة بناءً على المعلومات المقدمة "
            "من قاعدة المعرفة.\n\n"
            "**تعليمات مهمة:**\n"
            "1. اقرأ المعلومات المقدمة بعناية واستخدمها للإجابة على السؤال.\n"
            "2. قدم إجابة شاملة ومفصلة بالعربية الفصحى بناءً على المعلومات المتوفرة.\n"
            "3. ترجم المصطلحات الأجنبية إلى العربية. استخدم الاختصارات اللاتينية "
            "فقط للمصطلحات الشائعة (مثل EMG، AI).\n"
            "4. لا تُدخل كلمات إسبانية أو إنجليزية داخل الجمل العربية.\n"
            "5. لا تنسخ أخطاء OCR أو نصوص مشوهة من السياق.\n"
            "6. استخدم الترقيم العربي: ، ؛ ؟\n"
            "7. إذا لم تجد أي معلومات ذات صلة، قل: "
            "«لا أستطيع العثور على إجابة في المعلومات المتوفرة.»",
        ),
        (
            "human",
            "**المعلومات من قاعدة المعرفة:**\n{context}\n\n"
            "**السؤال:**\n{question}\n\n"
            "**الإجابة (يجب أن تكون شاملة ومبنية على المعلومات المقدمة):**",
        ),
    ]
)

ENGLISH_RAG_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an intelligent assistant specialized in answering questions based "
            "on the provided knowledge base information.\n\n"
            "Important Instructions:\n"
            "1. Read the provided information carefully and use it to answer.\n"
            "2. Provide a comprehensive answer based on the available information.\n"
            "3. If the information contains the answer, use it directly with appropriate context.\n"
            "4. If the information is partial, clearly state what is available.\n"
            "5. Only if you cannot find any relevant information, say: "
            '"I cannot find an answer in the provided information."',
        ),
        (
            "human",
            "**Information from Knowledge Base:**\n{context}\n\n"
            "**Question:**\n{question}\n\n"
            "**Answer (must be comprehensive and based on the provided information):**",
        ),
    ]
)


class RetrievalService:
    def __init__(
        self,
        vector_manager: VectorStoreManager,
        settings: Settings,
        reranker: Reranker | None = None,
        bm25: BM25Retriever | None = None,
    ) -> None:
        self.vector_manager = vector_manager
        self.settings = settings
        self.reranker = reranker
        self.bm25 = bm25 or BM25Retriever(settings)

    async def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        document_id: str | None = None,
        use_reranker: bool | None = None,
        use_hybrid: bool | None = None,
    ) -> list[tuple[Document, float]]:
        k = top_k or self.settings.top_k
        retrieval_k = max(k, self.settings.retrieval_min_k)

        should_rerank = (
            use_reranker
            if use_reranker is not None
            else self.settings.reranker_enabled
        )
        should_hybrid = (
            use_hybrid
            if use_hybrid is not None
            else self.settings.hybrid_search_enabled
        )

        fetch_k = retrieval_k * 3 if should_rerank and self.reranker else retrieval_k

        if should_hybrid:
            vector_results = self.vector_manager.search(
                query, k=fetch_k, document_id=document_id
            )
            bm25_results = await self.bm25.search(
                query, k=fetch_k, document_id=document_id
            )
            results = merge_hybrid_results(vector_results, bm25_results, fetch_k)
        else:
            results = self.vector_manager.search(
                query, k=fetch_k, document_id=document_id
            )

        if should_rerank and self.reranker and results:
            return self.reranker.rerank(query, results, k)

        return results[:k]


class GenerationService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.llm = ChatOllama(
            base_url=settings.ollama_base_url,
            model=settings.ollama_model,
            temperature=settings.llm_temperature,
        )

    def build_context(
        self,
        chunks: list[tuple[Document, float]],
        language: str = "en",
    ) -> str:
        parts: list[str] = []
        for index, (doc, _) in enumerate(chunks, start=1):
            source = doc.metadata.get("filename", "unknown")
            page = doc.metadata.get("page")

            if language == "ar":
                header = f"[مصدر {index}: {source}]"
            else:
                header = f"[Source {index}: {source}]"

            if page:
                header += f" (page {page})" if language != "ar" else f" (صفحة {page})"

            parts.append(f"{header}\n{doc.page_content.strip()}")

        return "\n\n---\n\n".join(parts)

    def to_source_chunks(
        self, chunks: list[tuple[Document, float]]
    ) -> list[SourceChunk]:
        sources: list[SourceChunk] = []
        for doc, score in chunks:
            sources.append(
                SourceChunk(
                    chunk_id=doc.id or "",
                    document_id=doc.metadata.get("document_id", ""),
                    filename=doc.metadata.get("filename", ""),
                    page=doc.metadata.get("page"),
                    score=round(float(score), 4),
                    text=doc.page_content[:500],
                )
            )
        return sources

    async def generate(
        self,
        question: str,
        chunks: list[tuple[Document, float]],
        language: str | None = None,
    ) -> str:
        lang = resolve_language(question, language)
        context = self.build_context(chunks, language=lang)
        prompt = ARABIC_RAG_PROMPT if lang == "ar" else ENGLISH_RAG_PROMPT
        chain = prompt | self.llm
        response = await chain.ainvoke({"context": context, "question": question})
        return response.content
