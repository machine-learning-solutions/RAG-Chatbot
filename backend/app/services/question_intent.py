"""Detect degraded LLM output (domain-agnostic)."""

from __future__ import annotations

import re


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
