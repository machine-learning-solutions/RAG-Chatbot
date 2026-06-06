import re

from app.services.hybrid_search import is_arabic_question, resolve_language

__all__ = [
    "is_arabic_question",
    "resolve_language",
    "sanitize_arabic_answer",
    "needs_arabic_polish",
    "normalize_phone_numbers",
]

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
    }
)
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


def _remove_stray_latin_words(text: str) -> str:
    def replace(match: re.Match[str]) -> str:
        word = match.group(0)
        return word if _is_allowed_latin(word) else ""

    return LATIN_WORD_RE.sub(replace, text)


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


def sanitize_arabic_answer(text: str) -> str:
    """Minimal post-processing: script cleanup only, no phrase-specific rules."""
    masked, token_store = _mask_contact_tokens(text)
    cleaned = _strip_disallowed_scripts(masked)
    cleaned = _strip_latin_parentheticals(cleaned)
    cleaned = _normalize_mixed_script_tokens(cleaned)
    cleaned = _remove_stray_latin_words(cleaned)
    cleaned = _remove_empty_parentheses(cleaned)
    cleaned = _normalize_whitespace(cleaned)
    cleaned = normalize_phone_numbers(cleaned)
    return _unmask_contact_tokens(cleaned, token_store)
