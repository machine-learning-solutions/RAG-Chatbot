import re

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from app.config import Settings
from app.models.schemas import SourceChunk
from app.services.hybrid_search import BM25Retriever, expand_retrieval_query, merge_hybrid_results
from app.services.language import (
    needs_arabic_polish,
    normalize_phone_numbers,
    resolve_language,
    sanitize_arabic_answer,
)
from app.services.question_intent import (
    NOT_FOUND_AR,
    is_degraded_arabic_answer,
    try_static_answer,
)
from app.services.reranker import Reranker
from app.services.resume_certs import format_certifications_answer
from app.services.vector_store import VectorStoreManager

ARABIC_RAG_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "أنت مساعد يجيب بالعربية الفصحى بناءً على معلومات قاعدة المعرفة.\n\n"
            "القواعد:\n"
            "1. المصادر قد تكون بأي لغة، لكن الإجابة عربية فصحى بالكامل.\n"
            "2. ترجم المصطلحات والأسماء الأجنبية إلى العربية، لكن انسخ عناوين "
            "البريد الإلكتروني وروابط LinkedIn وGitHub والموقع الشخصي كاملة "
            "كما في المصدر (مثل: name@email.com أو https://...).\n"
            "3. لا تدمج حروفاً لاتينية داخل كلمات عربية.\n"
            "4. لا تنسخ أخطاء OCR. لا تضع مراجعاً أو أقواساً في الإجابة "
            "(المصادر تُعرض منفصلة في الواجهة).\n"
            "5. عند السرد استخدم قائمة واضحة؛ ضع رقم القائمة والعنوان في "
            "نفس السطر (مثل: ١. تطوير التطبيقات) وليس الرقم في سطر منفصل.\n"
            "6. للتعريف العام أو التحية: 3-4 جمل مهذبة فقط، دون سرد السيرة كاملة.\n"
            "7. للأسئلة المحددة: إجابة وافية بالتفاصيل ذات الصلة فقط "
            "(المسميات، الشركات، التواريخ، التقنيات) دون حشو.\n"
            "8. ترجم كل المصطلحات التقنية إلى العربية (React → رياكت، "
            "TypeScript → تايب سكريبت، Node.js → نود، GraphQL → جراف كيو إل، "
            "PostgreSQL → بوستجري). لا تترك كلمات إنجليزية ولا قوائم فارغة "
            "(ممنوع: «باستخدام، و، و» أو «مثل و و»).\n"
            "9. اكتب أرقام الهاتف بالتنسيق الدولي مع + في البداية "
            "(مثل: +962 77 700 2130) ولا تعكس ترتيب الأرقام.\n"
            "10. إن لم تجد المعلومة المطلوبة في المصادر، أجب بإيجاز أنها غير متوفرة "
            "ولا تسرد خبرات أو مشاريع غير مرتبطة. لا تخمّن ولا تستنتج.\n"
            "11. عند ذكر الشهادات: اكتب اسم كل شهادة كاملاً بالعربية ثم التاريخ "
            "(مثل: شهادة البنفسجي في التحقق من التصميم باستخدام يو في إم - مارس 2025). "
            "لا تكتب التاريخ وحده.\n"
            "12. شهادات Purple وVLSI وASIC وCMOS وSystemVerilog وDesign Verification "
            "تخص مجال أشباه الموصلات والتحقق من التصميم.\n"
            "13. عند السؤال عن «آخر شهادة» (مفرد) اذكر الأحدث زمنياً فقط؛ "
            "وعند «آخر 5» اذكر خمساً مرتبة من الأحدث للأقدم.\n"
            "14. «كم بده/بكم/السعر/التكلفة» تعني التسعير والأتعاب وليس عدد المشاريع.\n"
            "15. إن سُئل عن التسعير أو عدد المشاريع ولم يُذكر في المصادر، "
            "قل بوضوح أن السيرة لا تتضمن هذه المعلومة واقترح التواصل عبر البريد أو الهاتف.",
        ),
        (
            "human",
            "**المعلومات من قاعدة المعرفة:**\n{context}\n\n"
            "**السؤال:**\n{question}\n\n"
            "**الإجابة:**",
        ),
    ]
)

ARABIC_POLISH_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "أنت محرر لغوي عربي. مهمتك إعادة صياغة مسودة إجابة لتصبح عربية فصحى "
            "سليمة نحوياً وبلاغياً.\n\n"
            "القواعد:\n"
            "1. التزم بحقائق مرجع المعرفة والمسودة؛ لا تضف ولا تحذف معلومات.\n"
            "2. أزل كل الكلمات الإنجليزية واستبدلها بترجمة عربية كاملة "
            "(إلا: AI، ML، API، IoT، JWT). لا تترك حروف عطف دون كلمات.\n"
            "3. أصلح الأسماء والمصطلحات المختلطة أو المشوّهة.\n"
            "4. حسّن البنية: جمل واضحة، وقوائم منظمة عند الحاجة.\n"
            "5. أزل الأقواس الفارغة وأي مراجع بين قوسين.\n"
            "6. عند الشهادات: احتفظ باسم كل شهادة بالعربية مع تاريخها؛ "
            "لا تُبقِ التاريخ دون اسم.\n"
            "7. أعد النص النهائي فقط دون مقدمات أو تعليقات.",
        ),
        (
            "human",
            "**مرجع المعرفة:**\n{context}\n\n"
            "**السؤال:**\n{question}\n\n"
            "**المسودة:**\n{draft}\n\n"
            "**الإجابة المنقّحة:**",
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
            "2. Provide a concise, unified answer (at most two short paragraphs).\n"
            "3. If the information contains the answer, use it directly with appropriate context.\n"
            "4. If the information is partial, clearly state what is available.\n"
            "5. Provide ONE consolidated answer only. Do not repeat the same "
            "information in different wording.\n"
            "6. Only if you cannot find any relevant information, say: "
            '"I cannot find an answer in the provided information."',
        ),
        (
            "human",
            "**Information from Knowledge Base:**\n{context}\n\n"
            "**Question:**\n{question}\n\n"
            "**Answer (one cohesive response, no repetition):**",
        ),
    ]
)


def _word_set(text: str) -> set[str]:
    return {word for word in re.findall(r"\w+", text.lower()) if len(word) > 2}


def _jaccard_similarity(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def deduplicate_chunks(
    chunks: list[tuple[Document, float]],
    similarity_threshold: float = 0.65,
) -> list[tuple[Document, float]]:
    unique: list[tuple[Document, float]] = []
    for doc, score in chunks:
        words = _word_set(doc.page_content)
        if any(
            _jaccard_similarity(words, _word_set(existing.page_content))
            >= similarity_threshold
            for existing, _ in unique
        ):
            continue
        unique.append((doc, score))
    return unique


def _paragraph_prefix(text: str, words: int = 5) -> str:
    tokens = re.findall(r"\w+", text.lower())
    return " ".join(tokens[:words])


def deduplicate_answer(text: str, similarity_threshold: float = 0.45) -> str:
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text.strip()) if part.strip()]
    if len(paragraphs) <= 1:
        paragraphs = [part.strip() for part in text.split("\n") if part.strip()]

    unique: list[str] = []
    seen_prefixes: set[str] = set()
    for paragraph in paragraphs:
        prefix = _paragraph_prefix(paragraph)
        words = _word_set(paragraph)
        if prefix in seen_prefixes:
            continue
        if any(
            _jaccard_similarity(words, _word_set(kept)) >= similarity_threshold
            for kept in unique
        ):
            continue
        seen_prefixes.add(prefix)
        unique.append(paragraph)

    return "\n\n".join(unique)


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
        search_query = expand_retrieval_query(query)

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
                search_query, k=fetch_k, document_id=document_id
            )
            bm25_results = await self.bm25.search(
                search_query, k=fetch_k, document_id=document_id
            )
            results = merge_hybrid_results(vector_results, bm25_results, fetch_k)
        else:
            results = self.vector_manager.search(
                search_query, k=fetch_k, document_id=document_id
            )

        if should_rerank and self.reranker and results:
            return self.reranker.rerank(query, results, k)

        return results[:k]


class GenerationService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        # Gemma 4 defaults to thinking mode; with num_predict capped it can
        # exhaust the budget on reasoning and return empty content.
        self.llm = ChatOllama(
            base_url=settings.ollama_base_url,
            model=settings.ollama_model,
            temperature=settings.llm_temperature,
            num_predict=settings.llm_num_predict,
            num_ctx=2048,
        ).bind(think=False)

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

            content = doc.page_content.strip()
            max_chars = self.settings.context_chunk_max_chars
            if len(content) > max_chars:
                content = content[:max_chars].rstrip() + "..."
            parts.append(f"{header}\n{content}")

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

    async def _polish_arabic(
        self, draft: str, question: str, context: str
    ) -> str:
        chain = ARABIC_POLISH_PROMPT | self.llm
        response = await chain.ainvoke(
            {"draft": draft, "question": question, "context": context}
        )
        return response.content

    async def generate(
        self,
        question: str,
        chunks: list[tuple[Document, float]],
        language: str | None = None,
    ) -> str:
        lang = resolve_language(question, language)

        static = try_static_answer(question, lang)
        if static:
            return static if lang != "ar" else sanitize_arabic_answer(static)

        all_distinct = deduplicate_chunks(chunks)
        distinct_chunks = all_distinct[: self.settings.max_context_chunks]
        context = self.build_context(distinct_chunks, language=lang)

        if lang == "ar":
            structured = format_certifications_answer(question, all_distinct)
            if structured:
                return sanitize_arabic_answer(structured)

        prompt = ARABIC_RAG_PROMPT if lang == "ar" else ENGLISH_RAG_PROMPT
        chain = prompt | self.llm
        response = await chain.ainvoke({"context": context, "question": question})
        answer = deduplicate_answer(response.content)
        if lang == "ar" and answer:
            if self.settings.arabic_polish_enabled and (
                needs_arabic_polish(answer) or is_degraded_arabic_answer(answer)
            ):
                answer = await self._polish_arabic(answer, question, context)
            answer = sanitize_arabic_answer(answer)
            if is_degraded_arabic_answer(answer):
                answer = sanitize_arabic_answer(NOT_FOUND_AR)
        elif answer:
            answer = normalize_phone_numbers(answer)
        return answer
