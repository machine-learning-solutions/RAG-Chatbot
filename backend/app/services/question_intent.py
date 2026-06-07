"""Detect questions the resume cannot answer; return helpful static responses."""

from __future__ import annotations

import re

# Jordanian/colloquial: «كم بده» = how much does he charge (NOT how many projects).
PRICING_RE = re.compile(
    r"كم\s*بده|بكم|السعر|التكلف|الأتعاب|أتعاب|تسعير|price|rate|cost|charge|fee|budget",
    re.IGNORECASE,
)

PROJECT_COUNT_RE = re.compile(
    r"كم\s*(?:عدد\s*)?مشروع|عدد\s*المشاريع|how\s+many\s+projects",
    re.IGNORECASE,
)

NOT_FOUND_AR = (
    "لا تتضمن المعلومات المتوفرة إجابةً عن هذا السؤال تحديداً. "
    "للاستفسار يمكن التواصل مع جهاد أبو عواد عبر:\n"
    "- البريد: jehadabuawwad@outlook.com\n"
    "- الهاتف: +962 77 700 2130\n"
    "- الموقع: www.jehadabuawwad.com"
)

PRICING_ANSWER_AR = (
    "لا تتضمن السيرة الذاتية معلومات عن أسعار المشاريع أو الأتعاب.\n\n"
    "التسعير يعتمد على نطاق المشروع ومدته وتقنياته. "
    "للاستفسار عن التعاون أو عرض سعر، يُفضّل التواصل مباشرة:\n"
    "- البريد: jehadabuawwad@outlook.com\n"
    "- الهاتف: +962 77 700 2130\n"
    "- الموقع: www.jehadabuawwad.com"
)

PROJECT_COUNT_ANSWER_AR = (
    "لا يُذكر في السيرة الذاتية عدد محدد للمشاريع التي عمل عليها جهاد أبو عواد.\n\n"
    "تتضمن السيرة خبراته في شركات ومشاريع متعددة مثل JoPath (WeFix)، "
    "CSC Beyond، Optimum Partners (Blinx، Olive، Quantalytics)، وNuqayyem، وغيرها. "
    "لتفاصيل مشروع معيّن أو نطاق تعاون، يُمكن التواصل عبر "
    "jehadabuawwad@outlook.com أو +962 77 700 2130."
)


def try_static_answer(question: str, language: str) -> str | None:
    """Return a canned answer when the resume cannot contain the requested info."""
    if language != "ar" and not PRICING_RE.search(question):
        return None

    if PRICING_RE.search(question):
        return PRICING_ANSWER_AR if language == "ar" else _pricing_en()

    if PROJECT_COUNT_RE.search(question):
        return PROJECT_COUNT_ANSWER_AR if language == "ar" else _project_count_en()

    return None


def _pricing_en() -> str:
    return (
        "The resume does not include project pricing or rate information.\n\n"
        "For collaboration or a quote, contact Jehad directly:\n"
        "- Email: jehadabuawwad@outlook.com\n"
        "- Phone: +962 77 700 2130\n"
        "- Website: www.jehadabuawwad.com"
    )


def _project_count_en() -> str:
    return (
        "The resume does not state an exact number of projects.\n\n"
        "It lists experience at JoPath (WeFix), CSC Beyond, Optimum Partners "
        "(Blinx, Olive, Quantalytics), Nuqayyem, and others. "
        "Contact jehadabuawwad@outlook.com for project-specific details."
    )


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
    )
    return any(re.search(p, text) for p in degraded_patterns)
