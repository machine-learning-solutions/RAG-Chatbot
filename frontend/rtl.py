import html
import re

ARABIC_RE = re.compile(
    r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]"
)


def contains_arabic(text: str) -> bool:
    return bool(ARABIC_RE.search(text))


def format_text_breaks(text: str) -> str:
    """Add line breaks after sentences and bullet points for readability."""
    lines = text.split("\n")
    formatted: list[str] = []
    for index, line in enumerate(lines):
        if line.lstrip() and not line.lstrip().startswith("-"):
            line = re.sub(r"\.([ \t]+)(?=\S)", ".\n", line)
        formatted.append(line)
        if line.lstrip().startswith("-") and line.rstrip().endswith("."):
            if index + 1 < len(lines) and lines[index + 1].strip():
                formatted.append("")
    return "\n".join(formatted)


def render_bidi_text(text: str, *, monospace: bool = False) -> None:
    import streamlit as st

    safe = html.escape(format_text_breaks(text)).replace("\n", "<br>")
    mono = "font-family: monospace; white-space: pre-wrap;" if monospace else ""
    st.markdown(
        f'<div dir="auto" class="bidi-text" style="{mono}">{safe}</div>',
        unsafe_allow_html=True,
    )


BIDI_CSS = """
.bidi-text,
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"],
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] p,
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] li,
[data-testid="stExpander"] [data-testid="stMarkdownContainer"] {
    direction: auto !important;
    text-align: start !important;
    unicode-bidi: plaintext !important;
}

[data-testid="stChatInput"] textarea {
    direction: auto !important;
    text-align: start !important;
    unicode-bidi: plaintext !important;
    min-height: 3.5rem !important;
    line-height: 1.45 !important;
    padding-top: 0.75rem !important;
    padding-bottom: 0.75rem !important;
}

[data-testid="stChatInput"] > div {
    min-height: 3.5rem !important;
    align-items: center !important;
}
"""

RTL_PAGE_CSS = """
html, body, [data-testid="stAppViewContainer"], [data-testid="stSidebar"] {
    direction: rtl !important;
    text-align: right !important;
    font-family: "Noto Sans Arabic", Tahoma, "Segoe UI", Arial, sans-serif;
}

[data-testid="stSidebar"] .stMarkdown,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span {
    text-align: right !important;
}

[data-testid="stChatMessage"] code,
[data-testid="stChatMessage"] pre {
    direction: ltr !important;
    text-align: left !important;
}
"""

LTR_PAGE_CSS = """
html, body, [data-testid="stAppViewContainer"] {
    font-family: "Segoe UI", Tahoma, Arial, sans-serif;
}
"""


PORTFOLIO_EMBED_CSS = """
[data-testid="stSidebar"],
[data-testid="stSidebarCollapsedControl"],
[data-testid="collapsedControl"],
section[data-testid="stSidebar"] {
    display: none !important;
}

[data-testid="stAppViewContainer"] > section.main > div.block-container {
    padding-top: 0.75rem !important;
    padding-bottom: 0.5rem !important;
    max-width: 100% !important;
}

[data-testid="stToolbar"],
header[data-testid="stHeader"] {
    display: none !important;
}

[data-testid="stDecoration"] {
    display: none !important;
}

[data-testid="stMainBlockContainer"] {
    padding-top: 0.25rem !important;
}

@media (max-width: 767px) {
    [data-testid="stBottomBlockContainer"] {
        padding-bottom: 0.5rem !important;
    }

    [data-testid="stAppViewContainer"] > section.main > div.block-container {
        padding-bottom: 0.25rem !important;
    }
}
"""


def page_css(language: str, *, portfolio: bool = False) -> str:
    base = BIDI_CSS + (RTL_PAGE_CSS if language == "ar" else LTR_PAGE_CSS)
    if portfolio:
        base += PORTFOLIO_EMBED_CSS
    return f"<style>{base}</style>"


UI_STRINGS = {
    "ar": {
        "page_title": "مساعد ذكي — Offline",
        "title": "💬 اسأل مستنداتك",
        "caption": "مدعوم بـ multilingual-e5-large و TurboVec و Ollama (Gemma 4)",
        "sidebar_title": "📚 مساعد RAG",
        "sidebar_caption": "100% Offline — TurboVec + Ollama",
        "chat_input": "اكتب سؤالك بالعربية أو الإنجليزية...",
        "portfolio_chat_input": "اسأل عن خبرتي، مهاراتي، شهاداتي، أو مشاريعي...",
        "upload": "رفع ملف",
        "upload_types": "PDF · Word · Markdown · TXT · JSON",
        "ingest": "فهرسة المستند",
        "filter": "تصفية المستندات",
        "manage": "إدارة",
        "no_docs": "لا توجد مستندات بعد. ارفع ملفاً للبدء.",
        "top_k": "عدد النتائج Top-K",
        "reranker": "إعادة ترتيب Cross-encoder",
        "hybrid": "بحث هجين (TurboVec + BM25)",
        "sources": "المصادر",
        "all_docs": "كل المستندات",
        "lang_en": "English",
        "lang_ar": "العربية",
        "welcome": "مرحباً! أنا مساعدك الذكي. ارفع مستنداً واسأل عنه.",
        "portfolio_welcome": (
            "مرحباً! أنا مساعد محفظة جهاد أبو عواد. "
            "اسألني عن خبرتي، مهاراتي، شهاداتي، مشاريعي، أو أي معلومات عني."
        ),
        "searching": "جاري البحث والتوليد...",
        "system_status": "حالة النظام",
    },
    "en": {
        "page_title": "RAG Chatbot — Offline",
        "title": "💬 Ask your documents",
        "caption": "Powered by multilingual-e5-large, TurboVec, and Ollama (Gemma 4)",
        "sidebar_title": "📚 RAG Chatbot",
        "sidebar_caption": "100% Offline — TurboVec + Ollama",
        "chat_input": "Ask a question in Arabic or English...",
        "portfolio_chat_input": "Ask about experience, skills, certifications, projects...",
        "upload": "Upload Documents",
        "upload_types": "PDF · Word · Markdown · TXT · JSON",
        "ingest": "Ingest Document",
        "filter": "Document Filter",
        "manage": "Manage",
        "no_docs": "No documents yet. Upload a file to start.",
        "top_k": "Top-K results",
        "reranker": "Cross-encoder reranker",
        "hybrid": "Hybrid search (TurboVec + BM25)",
        "sources": "Sources",
        "all_docs": "All documents",
        "lang_en": "English",
        "lang_ar": "العربية",
        "welcome": "Hello! Upload a document and ask questions about it.",
        "portfolio_welcome": (
            "Hello! I'm Jehad Abu Awwad's portfolio assistant. "
            "Ask about my experience, skills, certifications, projects, or background."
        ),
        "searching": "Searching & generating...",
        "system_status": "System status",
    },
}


def t(language: str, key: str) -> str:
    return UI_STRINGS.get(language, UI_STRINGS["en"]).get(key, key)
