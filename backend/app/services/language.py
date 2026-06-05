import re

from app.services.hybrid_search import is_arabic_question, resolve_language

__all__ = ["is_arabic_question", "resolve_language", "sanitize_arabic_answer"]

ARABIC_RE = re.compile(
    r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]"
)
LATIN_RE = re.compile(r"[A-Za-z]")
ALLOWED_LATIN_ACRONYMS = {
    "AI",
    "ML",
    "EMG",
    "API",
    "UI",
    "UX",
    "IoT",
    "IoV",
    "RAG",
    "SQL",
    "PDF",
}

# Whole-word / phrase replacements (case-insensitive).
ENGLISH_TO_ARABIC_PHRASES: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bmechatronics\s+engineer\b", re.I), "مهندس ميكاترونيكس"),
    (re.compile(r"\bmechatronics\b", re.I), "ميكاترونيكس"),
    (re.compile(r"\belectronics\s+technician\b", re.I), "فني إلكترونيات"),
    (re.compile(r"\bsoftware\s+developer\b", re.I), "مطور برمجيات"),
    (re.compile(r"\bfull[\s-]*stack\b", re.I), "متكامل"),
    (re.compile(r"\bmobile\s+developer\b", re.I), "مطور تطبيقات جوال"),
    (re.compile(r"\bmachine\s+learning\b", re.I), "تعلم الآلة"),
    (re.compile(r"\bweb\s+development\b", re.I), "تطوير الويب"),
    (re.compile(r"\bexperienced\b", re.I), "ذو خبرة"),
    (re.compile(r"\bengineer\b", re.I), "مهندس"),
    (re.compile(r"\bdeveloper\b", re.I), "مطور"),
    (re.compile(r"\btechnician\b", re.I), "فني"),
    (re.compile(r"\bmaintenance\b", re.I), "صيانة"),
    (re.compile(r"\bquality\b", re.I), "جودة"),
    (re.compile(r"\bcode\s+fellows\b", re.I), "معهد كود فيلوز"),
]

# Broken mixed-script tokens produced by small LLMs.
MIXED_TOKEN_REPLACEMENTS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"ميكانtroniks?", re.I), "ميكاترونيكس"),
    (re.compile(r"ميكانtronك", re.I), "ميكاترونيكس"),
    (re.compile(r"ميكانtron", re.I), "ميكاترونيكس"),
    (re.compile(r"ميكانيكs?", re.I), "ميكاترونيكس"),
    (re.compile(r"engineer", re.I), "مهندس"),
    (re.compile(r"Experienced", re.I), "ذو خبرة"),
]


def _is_mixed_script(token: str) -> bool:
    return bool(ARABIC_RE.search(token) and LATIN_RE.search(token))


def _is_allowed_latin_token(token: str) -> bool:
    cleaned = token.strip(".,;:!?()[]«»\"'")
    return cleaned.upper() in ALLOWED_LATIN_ACRONYMS


def _fix_mixed_token(token: str) -> str:
    for pattern, replacement in MIXED_TOKEN_REPLACEMENTS:
        if pattern.search(token):
            return pattern.sub(replacement, token)
    arabic_chars = "".join(ARABIC_RE.findall(token))
    if "ميكان" in arabic_chars:
        return "ميكاترونيكس"
    return ARABIC_RE.sub("", token) or token


def _replace_stray_latin_words(text: str) -> str:
    def replace_word(match: re.Match[str]) -> str:
        word = match.group(0)
        if _is_allowed_latin_token(word):
            return word
        return ""

    return re.sub(r"[A-Za-z][A-Za-z.\-/&]*", replace_word, text)


def sanitize_arabic_answer(text: str) -> str:
    cleaned = text
    for pattern, replacement in ENGLISH_TO_ARABIC_PHRASES:
        cleaned = pattern.sub(replacement, cleaned)
    for pattern, replacement in MIXED_TOKEN_REPLACEMENTS:
        cleaned = pattern.sub(replacement, cleaned)

    tokens = re.split(r"(\s+)", cleaned)
    fixed_tokens: list[str] = []
    for token in tokens:
        if not token.strip():
            fixed_tokens.append(token)
            continue
        if _is_mixed_script(token):
            fixed_tokens.append(_fix_mixed_token(token))
            continue
        fixed_tokens.append(token)

    cleaned = "".join(fixed_tokens)
    cleaned = _replace_stray_latin_words(cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    cleaned = re.sub(r"\s+([،؛.؟!])", r"\1", cleaned)
    return cleaned.strip()
