import re

from app.services.hybrid_search import is_arabic_question, resolve_language

__all__ = [
    "is_arabic_question",
    "resolve_language",
    "sanitize_arabic_answer",
    "strip_empty_numbered_items",
    "strip_meta_source_phrases",
    "needs_arabic_polish",
    "normalize_phone_numbers",
]

_KB_META_PHRASES: tuple[str, ...] = (
    r"بناءً على المعلومات المتوفرة(?: في قاعدة المعرفة)?(?: لدي)?",
    r"بناءً على قاعدة المعرفة(?: المتوفرة| المتاحة)?(?: لدي)?",
    r"من قاعدة المعرفة(?: المتوفرة| المتاحة)?",
    r"في قاعدة المعرفة(?: المتوفرة| المتاحة)?",
    r"قاعدة المعرفة(?: المتوفرة| المتاحة)?",
    r"بناءً على المعلومات المتاحة(?: لدي)?",
    r"أستند إلى المعلومات المتوفرة",
    r"يجيب(?: عن سيرة جهاد(?: أبو عواد)?)? من قاعدة المعرفة",
    r"يجيب عن سيرة جهاد(?: أبو عواد)? بناءً على [^.،]+",
    r"based on the (?:provided )?knowledge base",
    r"from the knowledge base",
    r"in the (?:provided )?knowledge base",
    r"the knowledge base",
)

# List marker only — no body text (truncation artifact, e.g. "**12.**" or "١٢.")
_EMPTY_NUMBERED_LINE_RE = re.compile(
    r"^\s*"
    r"(?:[-•]\s*)?"
    r"(?:\*{0,2})"
    r"(?:\d+|[\u0660-\u0669]+)"
    r"(?:\.|\)|-|–)"
    r"(?:\*{0,2})?"
    r"\s*$"
)

ARABIC_RE = re.compile(
    r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]"
)
LATIN_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9.\-/&]*")
ALLOWED_LATIN_ACRONYMS = frozenset(
    {"AI", "ML", "EMG", "API", "UI", "UX", "IOT", "IOV", "RAG", "SQL", "PDF"}
)
ALLOWED_LATIN_TERMS = frozenset(
    {
        "purple",
        "uvm",
        "systemverilog",
        "verilog",
        "vhdl",
        "asic",
        "vlsi",
        "cmos",
        "rtl",
        "vdsm",
        "synopsys",
        "iot",
        "jwt",
        "otp",
        "vfd",
        "vvvf",
        "plc",
        "scada",
        "pca",
        "lda",
    }
)

# English tech terms → Arabic transliteration (applied before stripping Latin).
TECH_GLOSSARY: dict[str, str] = {
    "typescript": "تايب سكريبت",
    "javascript": "جافاسكريبت",
    "node.js": "نود",
    "nodejs": "نود",
    "node": "نود",
    "react": "رياكت",
    "react.js": "رياكت",
    "next.js": "نيكست",
    "nextjs": "نيكست",
    "express": "إكسبريس",
    "express.js": "إكسبريس",
    "graphql": "جراف كيو إل",
    "postgresql": "بوستجري إس كيو إل",
    "postgres": "بوستجري",
    "mysql": "ماي إس كيو إل",
    "sequelize": "سيكوالايز",
    "firebase": "فايربيس",
    "flutter": "فلاتر",
    "django": "جانغو",
    "python": "بايثون",
    "dart": "دارت",
    "pytorch": "بايتورش",
    "tensorflow": "تينسورفلو",
    "scikit-learn": "سكيكيت ليرن",
    "sklearn": "سكيكيت ليرن",
    "pandas": "باندا",
    "matplotlib": "ماتبلوتليب",
    "netlify": "نتلايف",
    "heroku": "هيروكو",
    "redux": "ريدكس",
    "solid": "سوليد",
    "apollo": "أبولو",
    "server": "سيرفر",
    "docker": "دوكر",
    "compose": "كومبوز",
    "signalr": "سيجنال آر",
    "konva": "كونفا",
    "mongoose": "مونجووس",
    "mongodb": "مونجو دي بي",
    "n8n": "إن eight إن",
    "llama": "لاما",
    "llm": "نموذج لغوي",
    "bm25": "بي إم 25",
    "streamlit": "ستريمليت",
    "fastapi": "فاست إيه بي آي",
    "ollama": "أولاما",
    "turbovec": "تيربو فيك",
    "branch": "برانش",
    "redux": "ريدكس",
    "atomic": "ذري",
    "design": "تصميم",
    "pattern": "نمط",
    "clean": "نظيف",
    "architecture": "بنية",
    "microservices": "خدمات مصغرة",
    "restful": "ريست",
    "rest": "ريست",
    "orm": "أو آر إم",
    "ci/cd": "سي آي/سي دي",
    "devops": "ديف أوبس",
    "maddpg": "مادد بي جي",
    "greedy": "جريدي",
    "blca": "بي إل سي إيه",
    "iov": "إنترنت المركبات",
    "iot": "إنترنت الأشياء",
    "vfd": "محرك تردد متغير",
    "vvvf": "محرك جهد وتردد متغير",
    "plc": "تحكم منطقي قابل للبرمجة",
    "scada": "سكادا",
    "matlab": "ماتلاب",
    "opencv": "أوبن سي في",
}
KEEP_CHARS = set(".,،؛؟!-:()«»+[]/@#\n\t ") | set("0123456789")

URL_RE = re.compile(
    r"https?://[^\s\)\]،؛\"'<>]+"
    r"|www\.[^\s\)\]،؛\"'<>]+"
)
EMAIL_RE = re.compile(
    r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"
)
CONTACT_TOKEN_PATTERNS = (URL_RE, EMAIL_RE)


def _is_allowed_latin(word: str) -> bool:
    stripped = word.strip(".,;:!?()[]«»\"'")
    if stripped.upper() in ALLOWED_LATIN_ACRONYMS:
        return True
    return stripped.lower() in ALLOWED_LATIN_TERMS


def _strip_disallowed_scripts(text: str) -> str:
    """Keep Arabic letters, digits, punctuation, whitespace, and Latin acronyms only."""
    result: list[str] = []
    for char in text:
        if ARABIC_RE.match(char) or char in KEEP_CHARS:
            result.append(char)
        elif char.isascii() and char.isalpha():
            result.append(char)
    return "".join(result)


def _normalize_mixed_script_tokens(text: str) -> str:
    """Words mixing Arabic and Latin lose their Latin portion; polish pass restores meaning."""
    tokens = re.split(r"(\s+)", text)
    normalized: list[str] = []
    for token in tokens:
        if not token.strip():
            normalized.append(token)
            continue
        has_arabic = bool(ARABIC_RE.search(token))
        has_latin = bool(re.search(r"[A-Za-z]", token))
        if has_arabic and has_latin:
            arabic_only = "".join(ARABIC_RE.findall(token))
            normalized.append(arabic_only or token)
            continue
        normalized.append(token)
    return "".join(normalized)


def _strip_latin_parentheticals(text: str) -> str:
    """Remove (English text) groups instead of leaving empty parentheses."""

    def replace(match: re.Match[str]) -> str:
        inner = match.group(1)
        if ARABIC_RE.search(inner):
            return match.group(0)
        if LATIN_WORD_RE.search(inner):
            return ""
        return match.group(0)

    return re.sub(r"\(([^)]*)\)", replace, text)


def _glossary_key(word: str) -> str:
    return word.strip(".,;:!?()[]«»\"'-*").lower()


def _replace_latin_words(text: str) -> str:
    """Translate known tech terms; drop unknown Latin only as last resort."""

    def replace(match: re.Match[str]) -> str:
        word = match.group(0)
        if _is_allowed_latin(word):
            return word
        key = _glossary_key(word)
        if key in TECH_GLOSSARY:
            return TECH_GLOSSARY[key]
        for term, arabic in sorted(TECH_GLOSSARY.items(), key=lambda x: -len(x[0])):
            if key == term or key.startswith(term + ".") or term in key:
                return arabic
        return ""

    return LATIN_WORD_RE.sub(replace, text)


def _repair_orphan_conjunctions(text: str) -> str:
    """Fix lists gutted by stripped Latin, e.g. «باستخدام، و، و»."""
    text = re.sub(r"(?:،\s*و\s*){2,}", "، ", text)
    text = re.sub(r"باستخدام\s*،\s*(?:و\s*،?\s*)+", "باستخدام ", text)
    text = re.sub(r"مثل\s*(?:و\s*،?\s*)+", "مثل ", text)
    text = re.sub(r":\s*(?:و\s*،?\s*)+", ": ", text)
    text = re.sub(r"\(\s*و\s*\)", "", text)
    text = re.sub(r"\s+و\s+و\s+", " ", text)
    text = re.sub(r"،\s*و\s*([،.؟!]|$)", r"\1", text)
    return text


def _remove_empty_parentheses(text: str) -> str:
    cleaned = text
    while True:
        updated = re.sub(r"\(\s*\)", "", cleaned)
        if updated == cleaned:
            return updated
        cleaned = updated


def _mask_contact_tokens(text: str) -> tuple[str, dict[str, str]]:
    """Replace URLs and emails with placeholders before Latin-stripping passes."""
    store: dict[str, str] = {}
    masked = text
    for pattern in CONTACT_TOKEN_PATTERNS:
        def repl(match: re.Match[str], _store: dict[str, str] = store) -> str:
            token = f"[[{len(_store)}]]"
            _store[token] = match.group(0)
            return token

        masked = pattern.sub(repl, masked)
    return masked, store


def _unmask_contact_tokens(text: str, store: dict[str, str]) -> str:
    for token, value in store.items():
        text = text.replace(token, value)
    return text


def _normalize_whitespace(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"\s+([،؛.؟!])", r"\1", text)
    text = re.sub(r"،{2,}", "،", text)
    return text.strip()


JORDAN_INTL_PHONE_RE = re.compile(
    r"(?<![+\d])"
    r"(?:00962|962)"
    r"[\s\-]?"
    r"(\d{2})"
    r"[\s\-]?"
    r"(\d{3})"
    r"[\s\-]?"
    r"(\d{4})"
    r"(?!\d)"
)

JORDAN_LOCAL_PHONE_RE = re.compile(
    r"(?<![+\d])"
    r"0"
    r"(\d{2})"
    r"[\s\-]?"
    r"(\d{3})"
    r"[\s\-]?"
    r"(\d{4})"
    r"(?!\d)"
)


def normalize_phone_numbers(text: str) -> str:
    """Ensure Jordan phone numbers use the +962 international prefix."""

    def repl(match: re.Match[str]) -> str:
        return f"+962 {match.group(1)} {match.group(2)} {match.group(3)}"

    text = JORDAN_LOCAL_PHONE_RE.sub(repl, text)
    return JORDAN_INTL_PHONE_RE.sub(repl, text)


def needs_arabic_polish(text: str) -> bool:
    if re.search(r"\(\s*\)", text):
        return True
    if re.search(r"(?:،\s*و\s*){2,}|باستخدام\s*،\s*و", text):
        return True
    for token in re.split(r"\s+", text):
        if not token:
            continue
        has_arabic = bool(ARABIC_RE.search(token))
        has_latin = bool(re.search(r"[A-Za-z]", token))
        if has_arabic and has_latin:
            return True
        if has_latin and not _is_allowed_latin(token):
            return True
    return False


_REPETITION_RUN_RE = re.compile(r"(.{1,16}?)\1{5,}")


def strip_repetitive_tail(text: str) -> str:
    """Truncate answers that degenerate into repeated token loops."""
    if not text:
        return text
    match = _REPETITION_RUN_RE.search(text)
    if match:
        text = text[: match.start()].rstrip()
    cleaned_lines: list[str] = []
    numbered_line = re.compile(r"^[٠-٩0-9]+[\.\)]")
    for line in text.split("\n"):
        if len(line) > 500 and not numbered_line.match(line.strip()):
            line = line[:500].rsplit(" ", 1)[0] + "…"
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines).rstrip()


def strip_trailing_latin_block(text: str) -> str:
    """Remove English tail paragraphs accidentally appended to Arabic answers."""
    if not text or not ARABIC_RE.search(text):
        return text
    paragraphs = re.split(r"\n\n+", text.strip())
    while paragraphs:
        tail = paragraphs[-1].strip()
        if not tail:
            paragraphs.pop()
            continue
        latin = len(re.findall(r"[A-Za-z]", tail))
        arabic = len(ARABIC_RE.findall(tail))
        if latin > 60 and latin > arabic * 2:
            paragraphs.pop()
            continue
        break
    return "\n\n".join(paragraphs).strip()


def strip_empty_numbered_items(text: str) -> str:
    """Drop numbered list lines that contain only a marker with no content."""
    if not text:
        return text
    lines = [line for line in text.split("\n") if not _EMPTY_NUMBERED_LINE_RE.match(line)]
    return "\n".join(lines).rstrip()


def strip_meta_source_phrases(text: str) -> str:
    """Remove meta phrases about knowledge base / retrieved sources from answers."""
    if not text:
        return text
    cleaned = text
    for pattern in _KB_META_PHRASES:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    cleaned = re.sub(r"،\s*،+", "،", cleaned)
    cleaned = re.sub(r"\.\s*\.+", ".", cleaned)
    return cleaned.strip()


def sanitize_arabic_answer(text: str, *, light: bool = False) -> str:
    """Post-process Arabic answers. Use light=True for long numbered lists."""
    if light:
        cleaned = strip_meta_source_phrases(text)
        cleaned = strip_empty_numbered_items(cleaned)
        cleaned = _normalize_whitespace(cleaned)
        return normalize_phone_numbers(cleaned)

    masked, token_store = _mask_contact_tokens(text)
    cleaned = _strip_disallowed_scripts(masked)
    cleaned = _strip_latin_parentheticals(cleaned)
    cleaned = _normalize_mixed_script_tokens(cleaned)
    cleaned = _replace_latin_words(cleaned)
    cleaned = _remove_empty_parentheses(cleaned)
    cleaned = _repair_orphan_conjunctions(cleaned)
    cleaned = strip_empty_numbered_items(cleaned)
    cleaned = strip_trailing_latin_block(cleaned)
    cleaned = strip_repetitive_tail(cleaned)
    cleaned = strip_meta_source_phrases(cleaned)
    cleaned = _normalize_whitespace(cleaned)
    cleaned = normalize_phone_numbers(cleaned)
    return _unmask_contact_tokens(cleaned, token_store)
