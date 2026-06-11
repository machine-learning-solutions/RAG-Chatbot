"""Detect degraded LLM output and portfolio greeting intent."""

from __future__ import annotations

import re

PURE_GREETING_RE = re.compile(
    r"^(?:"
    r"مرحبا|مرحباً|أهلاً?|اهلاً?|هلا|السلام عليكم|سلام عليكم|سلام|"
    r"صباح الخير|مساء الخير|"
    r"hello|hi|hey|greetings|good\s+(?:morning|afternoon|evening)"
    r")[\s!?.,،]*$",
    re.IGNORECASE,
)

INTRO_GREETING_RE = re.compile(
    r"من\s+أنت|من\s+انت|who\s+are\s+you|"
    r"عرفني\s+عن|قدّم|قدم\s+نفسك|تعريف\s+ب",
    re.IGNORECASE,
)


def is_greeting_question(question: str) -> bool:
    """Short welcome / who-is-this — not a factual CV lookup."""
    q = question.strip()
    if PURE_GREETING_RE.match(q):
        return True
    return bool(INTRO_GREETING_RE.search(q))


def is_degraded_arabic_answer(text: str) -> bool:
    """Detect answers gutted by sanitization or incomplete LLM output."""
    if not text or len(text) < 40:
        return True
    degraded_patterns = (
        r"\(\s*و\s*\)",
        r"في مشروع\s*$",
        r"منصة\s*$",
        r"باستخدام\s*:?\s*$",
        r"مطور ويب \(\s*و\s*\)",
        r"عمل كمطور تطبيقات باستخدام\s*:",
        r"(?:،\s*و\s*){2,}",
        r"باستخدام\s*،\s*و",
        r":\s*و\s*و",
        r"مثل\s*و\s*و",
    )
    return any(re.search(p, text) for p in degraded_patterns)
