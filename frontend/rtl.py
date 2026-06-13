import html
import re

ARABIC_RE = re.compile(
    r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]"
)


def contains_arabic(text: str) -> bool:
    return bool(ARABIC_RE.search(text))


NUM_ONLY_LINE = re.compile(r"^([٠-٩]+|\d+)\.\s*$")
NUM_HEADING_LINE = re.compile(r"^([٠-٩]+|\d+)\.\s*(.+)$")
ARABIC_LIST_SPLIT = re.compile(r"(?=^(?:[٠-٩]+|\d+)\.)", re.MULTILINE)
INLINE_AR_NUM = re.compile(r"(?<=[^\n])\s+([٠-٩]{1,2})\.\s+")
INLINE_EN_NUM = re.compile(r"(?<=[^\n])\s+(\d{1,2})\.\s+")


def split_inline_numbered_items(text: str) -> str:
    if not text:
        return text
    if contains_arabic(text):
        return INLINE_AR_NUM.sub(r"\n\1. ", text)
    return INLINE_EN_NUM.sub(r"\n\1. ", text)


def merge_orphan_list_numbers(text: str) -> str:
    """Join «٣.» on its own line with the next line so RTL layout stays correct."""
    lines = text.split("\n")
    merged: list[str] = []
    index = 0
    while index < len(lines):
        line = lines[index]
        stripped = line.strip()
        if (
            NUM_ONLY_LINE.match(stripped)
            and index + 1 < len(lines)
            and lines[index + 1].strip()
        ):
            merged.append(f"{stripped} {lines[index + 1].strip()}")
            index += 2
            continue
        merged.append(line)
        index += 1
    return "\n".join(merged)


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


def prepare_arabic_text(text: str) -> str:
    return merge_orphan_list_numbers(
        format_text_breaks(split_inline_numbered_items(text))
    )


# Phone numbers, emails, and URLs must stay LTR inside Arabic paragraphs.
LTR_ISOLATE_RE = re.compile(
    r"""
    \+[\d\s\-().]{6,30}\d
    |\b(?:\+?962|00962)[\s\-]?\d{2}[\s\-]?\d{3}[\s\-]?\d{4}\b
    |\b0\d{2}[\s\-]?\d{3}[\s\-]?\d{4}\b
    |[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}
    |https?://[^\s<>"']+
    |\b\d{1,4}(?:[\s\-]\d{2,5}){2,4}\b
    """,
    re.VERBOSE,
)


def wrap_ltr_isolates(text: str) -> str:
    parts: list[str] = []
    last = 0
    for match in LTR_ISOLATE_RE.finditer(text):
        parts.append(html.escape(text[last : match.start()]))
        segment = html.escape(match.group(0).strip())
        parts.append(
            '<span dir="ltr" class="ltr-isolate" '
            'style="unicode-bidi:isolate; direction:ltr; display:inline-block;">'
            f"{segment}</span>"
        )
        last = match.end()
    parts.append(html.escape(text[last:]))
    return "".join(parts)


def _render_plain_arabic_html(text: str) -> str:
    safe = wrap_ltr_isolates(text).replace("\n", "<br>")
    return f'<div dir="rtl" class="bidi-text bidi-text-rtl">{safe}</div>'


def _render_arabic_numbered_list_html(text: str) -> str:
    """Render Arabic ١. ٢. sections as a proper RTL ordered list."""
    blocks = ARABIC_LIST_SPLIT.split(text)
    parts: list[str] = ['<div dir="rtl" class="bidi-text bidi-text-rtl">']

    preamble = blocks[0].strip() if blocks else ""
    if preamble:
        parts.append(
            f'<div class="bidi-preamble">{wrap_ltr_isolates(preamble).replace(chr(10), "<br>")}</div>'
        )

    items: list[str] = []
    for block in blocks[1:]:
        block = block.strip()
        if not block:
            continue
        lines = block.split("\n")
        heading_match = NUM_HEADING_LINE.match(lines[0].strip())
        if not heading_match:
            continue
        num = heading_match.group(1)
        title = heading_match.group(2).strip()
        body = "\n".join(lines[1:]).strip()
        title_html = wrap_ltr_isolates(title)
        if body:
            body_html = wrap_ltr_isolates(body).replace("\n", "<br>")
            items.append(
                f'<li><span class="ar-list-num"><strong>{html.escape(num)}.</strong></span> '
                f'<span class="ar-list-title">{title_html}</span>'
                f'<div class="ar-list-body">{body_html}</div></li>'
            )
        else:
            items.append(
                f'<li><span class="ar-list-num"><strong>{html.escape(num)}.</strong></span> '
                f'<span class="ar-list-title">{title_html}</span></li>'
            )

    if items:
        parts.append('<ol dir="rtl" class="ar-num-list">')
        parts.extend(items)
        parts.append("</ol>")

    parts.append("</div>")
    return "".join(parts)


def build_bidi_html(text: str) -> str:
    prepared = prepare_arabic_text(text)
    if contains_arabic(prepared) and re.search(
        r"^(?:[٠-٩]+|\d+)\.", prepared, re.MULTILINE
    ):
        return _render_arabic_numbered_list_html(prepared)
    if contains_arabic(prepared):
        return _render_plain_arabic_html(prepared)
    safe = wrap_ltr_isolates(prepared).replace("\n", "<br>")
    return f'<div dir="auto" class="bidi-text">{safe}</div>'


def render_bidi_text(text: str, *, monospace: bool = False) -> None:
    import streamlit as st

    html_body = build_bidi_html(text)
    if monospace:
        html_body = html_body.replace(
            'class="bidi-text',
            'class="bidi-text" style="font-family: monospace; white-space: pre-wrap;"',
            1,
        )
    st.markdown(html_body, unsafe_allow_html=True)


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

.ltr-isolate {
    direction: ltr !important;
    unicode-bidi: isolate !important;
    display: inline-block !important;
    text-align: left !important;
}

.bidi-text-rtl,
.bidi-text[dir="rtl"] {
    direction: rtl !important;
    text-align: right !important;
    unicode-bidi: embed !important;
}

.bidi-preamble {
    margin-bottom: 0.75rem;
}

ol.ar-num-list {
    direction: rtl !important;
    text-align: right !important;
    list-style: none;
    padding: 0 0.25rem 0 0 !important;
    margin: 0.5rem 0 0.75rem !important;
}

ol.ar-num-list li {
    direction: rtl !important;
    text-align: right !important;
    unicode-bidi: embed !important;
    margin-bottom: 0.85rem;
    display: block;
}

.ar-list-num {
    font-weight: 700;
    margin-left: 0.35rem;
}

.ar-list-title {
    font-weight: 600;
}

.ar-list-body {
    margin-top: 0.35rem;
    font-weight: 400;
    padding-right: 0.25rem;
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
            "مرحباً! أنا مساعد ملف نماذج الاعمال لـجهاد أبو عواد. "
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
