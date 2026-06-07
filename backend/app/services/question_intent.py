"""Detect question intent; return concise static responses when appropriate."""

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

GREETING_RE = re.compile(
    r"^(?:مرحبا|مرحباً|السلام\s+عليكم|هلا|أهلاً?|صباح\s+الخير|مساء\s+الخير|"
    r"hello|hi|hey|good\s+(?:morning|evening))\s*[!؟?.]*\s*$",
    re.IGNORECASE,
)

INTRO_ABOUT_RE = re.compile(
    r"(?:"
    r"من\s+أنت|من\s+هو(?:\s+جهاد)?|مين\s+(?:جهاد|أنت|هو)|"
    r"عرفني(?:\s+عن)?(?:\s+نفسك|\s+جهاد|\s+نفسه)?|"
    r"احكيلي\s+عن|خبرني\s+عن|قلّ?ي\s+عن\s+نفسك|"
    r"قدّم\s+نفسك|قدم\s+نفسك|نبذة\s+عن|تعريف\s+(?:عن|مختصر)?|"
    r"who\s+are\s+you|who\s+is\s+jehad|introduce\s+yourself|"
    r"tell\s+me\s+about\s+(?:you|jehad|yourself)"
    r")",
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

GREETING_ANSWER_AR = (
    "أهلاً وسهلاً! أنا مساعد جهاد أبو عواد، وأجيب من سيرته الذاتية.\n\n"
    "اسألني عن خبرته، مهاراته، شهاداته، مشاريعه، أو كيفية التواصل معه."
)

INTRO_ANSWER_AR = (
    "أهلاً بك!\n\n"
    "جهاد أبو عواد مهندس ميكاترونكس ومطور برمجيات من عمّان، "
    "يجمع بين هندسة الأنظمة الصناعية وتطوير الويب والموبايل والذكاء الاصطناعي.\n\n"
    "يمكنني الإجابة عن خبرته العملية، مهاراته التقنية، شهاداته، ومشاريعه. "
    "ما الذي تود معرفته تحديداً؟"
)


def is_greeting(question: str) -> bool:
    return bool(GREETING_RE.match(question.strip()))


def is_intro_question(question: str) -> bool:
    text = question.strip()
    if is_greeting(text):
        return True
    if INTRO_ABOUT_RE.search(text):
        return True
    return False


def try_static_answer(question: str, language: str) -> str | None:
    """Return a canned answer for known intents or missing resume fields."""
    text = question.strip()

    if is_greeting(text):
        return GREETING_ANSWER_AR if language == "ar" else _greeting_en()

    if is_intro_question(text):
        return INTRO_ANSWER_AR if language == "ar" else _intro_en()

    if PRICING_RE.search(text):
        return PRICING_ANSWER_AR if language == "ar" else _pricing_en()

    if PROJECT_COUNT_RE.search(text):
        return PROJECT_COUNT_ANSWER_AR if language == "ar" else _project_count_en()

    return None


def _greeting_en() -> str:
    return (
        "Hello! I'm Jehad Abu Awwad's portfolio assistant.\n\n"
        "Ask me about his experience, skills, certifications, projects, or how to reach him."
    )


def _intro_en() -> str:
    return (
        "Hello!\n\n"
        "Jehad Abu Awwad is a Mechatronics Engineer and software developer based in Amman, "
        "combining industrial systems engineering with web, mobile, and AI development.\n\n"
        "I can answer questions about his work experience, technical skills, certifications, "
        "and projects. What would you like to know?"
    )


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
        r"(?:،\s*و\s*){2,}",
        r"باستخدام\s*،\s*و",
        r":\s*و\s*و",
        r"مثل\s*و\s*و",
    )
    return any(re.search(p, text) for p in degraded_patterns)
