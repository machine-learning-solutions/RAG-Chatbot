import re

from app.services.hybrid_search import is_arabic_question, resolve_language

__all__ = ["is_arabic_question", "resolve_language", "sanitize_arabic_answer"]

ARABIC_RE = re.compile(
    r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]"
)
LATIN_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9.\-/&]*")
ALLOWED_LATIN_ACRONYMS = frozenset(
    {"AI", "ML", "EMG", "API", "UI", "UX", "IOT", "IOV", "RAG", "SQL", "PDF"}
)
KEEP_CHARS = set(".,،؛؟!-:()«»/\n\t ") | set("0123456789")


def _is_allowed_latin(word: str) -> bool:
    return word.strip(".,;:!?()[]«»\"'").upper() in ALLOWED_LATIN_ACRONYMS


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


def _remove_stray_latin_words(text: str) -> str:
    def replace(match: re.Match[str]) -> str:
        word = match.group(0)
        return word if _is_allowed_latin(word) else ""

    return LATIN_WORD_RE.sub(replace, text)


def _normalize_whitespace(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"\s+([،؛.؟!])", r"\1", text)
    text = re.sub(r"،{2,}", "،", text)
    return text.strip()


def sanitize_arabic_answer(text: str) -> str:
    """Minimal post-processing: script cleanup only, no phrase-specific rules."""
    cleaned = _strip_disallowed_scripts(text)
    cleaned = _normalize_mixed_script_tokens(cleaned)
    cleaned = _remove_stray_latin_words(cleaned)
    return _normalize_whitespace(cleaned)
