"""Structured certification extraction from resume-style markdown chunks."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime

CERT_LINE_RE = re.compile(
    r"[-●]\s*\*\*([^*]+)\*\*\s*[—–-]\s*([^\n]+)",
    re.MULTILINE,
)

SEMICONDUCTOR_RE = re.compile(
    r"purple|vlsi|asic|cmos|systemverilog|verilog|vhdl|vdsm|design verification|"
    r"rtl|uvm|synops|semiconductor|submicron",
    re.IGNORECASE,
)

CERT_QUESTION_RE = re.compile(r"شهاد|ترخيص|certif|license", re.IGNORECASE)

LAST_N_RE = re.compile(
    r"آخر\s*(?:(\d+)|(?:خمس|5)|(?:أربع|4)|(?:ثلاث|3)|(?:اثنت|2))"
    r"|last\s*(\d+)",
    re.IGNORECASE,
)

SINGLE_LATEST_RE = re.compile(
    r"آخر\s+شهاد(?:ة|ات)?(?!\s*(?:\d|خمس|أربع|ثلاث|اثنت))",
    re.IGNORECASE,
)

MONTHS = {
    "jan": 1,
    "january": 1,
    "feb": 2,
    "february": 2,
    "mar": 3,
    "march": 3,
    "apr": 4,
    "april": 4,
    "may": 5,
    "jun": 6,
    "june": 6,
    "jul": 7,
    "july": 7,
    "aug": 8,
    "august": 8,
    "sep": 9,
    "sept": 9,
    "september": 9,
    "oct": 10,
    "october": 10,
    "nov": 11,
    "november": 11,
    "dec": 12,
    "december": 12,
}

MONTHS_AR = {
    1: "يناير",
    2: "فبراير",
    3: "مارس",
    4: "أبريل",
    5: "مايو",
    6: "يونيو",
    7: "يوليو",
    8: "أغسطس",
    9: "سبتمبر",
    10: "أكتوبر",
    11: "نوفمبر",
    12: "ديسمبر",
}

# Stable Arabic labels for resume certification titles (no LLM hallucination).
CERT_TITLE_AR: dict[str, str] = {
    "Purple Certification: Design Verification Track": (
        "شهادة Purple - مسار التحقق من التصميم"
    ),
    "Purple Certification: SystemVerilog Verification using UVM": (
        "شهادة Purple - التحقق من SystemVerilog باستخدام UVM"
    ),
    "Purple Certification: SystemVerilog Assertions for Formal Verification": (
        "شهادة Purple - تأكيدات SystemVerilog للتحقق الرسمي"
    ),
    "Purple Certification: SystemVerilog Testbench": (
        "شهادة Purple - منصة اختبار SystemVerilog"
    ),
    "Purple Certification:  SystemVerilog for RTL Design": (
        "شهادة Purple - SystemVerilog لتصميم RTL"
    ),
    "Purple Certification: System Verlog Assertion": (
        "شهادة Purple - تأكيدات SystemVerilog"
    ),
    "Purple Certification: Very Deep Submicron (VDSM) Fundamentals": (
        "شهادة Purple - أساسيات التقنية فائقة الدقة (VDSM)"
    ),
    "SystemVerilog Refresher": "مراجعة SystemVerilog",
    "VHDL Refresher": "مراجعة VHDL",
    "Verilog Refresher": "مراجعة Verilog",
    "Design Verification: Comprehensive": "التحقق من التصميم - دورة شاملة",
    "Purple Certification: VLSI Basics": "شهادة Purple - أساسيات VLSI",
    "Purple Certification: Digital Design Fundamentals": (
        "شهادة Purple - أساسيات التصميم الرقمي"
    ),
    "Purple Certification: ASIC Design Flow": (
        "شهادة Purple - مسار تصميم ASIC"
    ),
    "Purple Certification: CMOS Fundamentals": (
        "شهادة Purple - أساسيات CMOS"
    ),
}


@dataclass(frozen=True)
class Certification:
    title: str
    date: datetime
    raw_date: str

    @property
    def is_semiconductor(self) -> bool:
        return bool(SEMICONDUCTOR_RE.search(self.title))

    def title_ar(self) -> str:
        normalized = " ".join(self.title.split())
        for key, label in CERT_TITLE_AR.items():
            if key.replace("  ", " ") == normalized or key in self.title:
                return label
        if self.title.startswith("Purple Certification:"):
            detail = self.title.split(":", 1)[1].strip()
            return f"شهادة Purple — {detail}"
        return self.title


def _parse_date(raw: str) -> datetime | None:
    match = re.search(
        r"(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
        r"Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|"
        r"Dec(?:ember)?)\s+(\d{4})",
        raw,
        re.IGNORECASE,
    )
    if not match:
        return None
    month = MONTHS.get(match.group(1).lower()[:3], 0)
    if not month and match.group(1).lower() in MONTHS:
        month = MONTHS[match.group(1).lower()]
    year = int(match.group(2))
    if not month:
        return None
    return datetime(year, month, 1)


def _format_date_ar(when: datetime, raw: str) -> str:
    month = MONTHS_AR.get(when.month, raw)
    return f"{month} {when.year}"


def extract_certifications(text: str) -> list[Certification]:
    found: list[Certification] = []
    seen: set[str] = set()
    for match in CERT_LINE_RE.finditer(text):
        title = match.group(1).strip()
        raw_date = match.group(2).strip()
        when = _parse_date(raw_date)
        if not when:
            continue
        key = f"{title}|{when.isoformat()}"
        if key in seen:
            continue
        seen.add(key)
        found.append(Certification(title=title, date=when, raw_date=raw_date))
    found.sort(key=lambda item: item.date, reverse=True)
    return found


def extract_certifications_from_chunks(chunks: list) -> list[Certification]:
    merged: list[Certification] = []
    seen: set[str] = set()
    for doc, _score in chunks:
        for cert in extract_certifications(doc.page_content):
            key = f"{cert.title}|{cert.date.isoformat()}"
            if key in seen:
                continue
            seen.add(key)
            merged.append(cert)
    merged.sort(key=lambda item: item.date, reverse=True)
    return merged


def parse_cert_limit(question: str) -> int | None:
    match = LAST_N_RE.search(question)
    if not match:
        return None
    for group in match.groups():
        if group and group.isdigit():
            return int(group)
    if re.search(r"خمس|5", question):
        return 5
    if re.search(r"أربع|4", question):
        return 4
    if re.search(r"ثلاث|3", question):
        return 3
    if re.search(r"اثنت|2", question):
        return 2
    return 5


def is_certification_question(question: str) -> bool:
    return bool(CERT_QUESTION_RE.search(question))


def is_semiconductor_cert_question(question: str) -> bool:
    return bool(
        re.search(
            r"أشباه\s*الموصلات|موصلات|VLSI|ASIC|CMOS|semiconductor|"
            r"تحقق\s*من\s*التصميم",
            question,
            re.IGNORECASE,
        )
    )


def format_certifications_answer(question: str, chunks: list) -> str | None:
    """Return a deterministic Arabic answer for certification list questions."""
    if not is_certification_question(question):
        return None

    certs = extract_certifications_from_chunks(chunks)
    if not certs:
        return None

    if is_semiconductor_cert_question(question):
        certs = [cert for cert in certs if cert.is_semiconductor]
        if not certs:
            return None
        limit = 1 if SINGLE_LATEST_RE.search(question) else parse_cert_limit(question)
    else:
        limit = parse_cert_limit(question)

    if limit is None and SINGLE_LATEST_RE.search(question):
        limit = 1

    if limit is None:
        return None

    selected = certs[:limit]
    if not selected:
        return None

    if limit == 1:
        cert = selected[0]
        date_ar = _format_date_ar(cert.date, cert.raw_date)
        if is_semiconductor_cert_question(question):
            intro = (
                "آخر شهادة حصل عليها جهاد أبو عواد في مجال أشباه الموصلات "
                "والتحقق من التصميم هي:"
            )
        else:
            intro = "آخر شهادة مسجلة في قاعدة المعرفة هي:"
        return f"{intro}\n\n- {cert.title_ar()} - {date_ar}"

    lines = [
        f"{index}. {cert.title_ar()} - {_format_date_ar(cert.date, cert.raw_date)}"
        for index, cert in enumerate(selected, start=1)
    ]
    intro = "آخر الشهادات المسجلة في قاعدة المعرفة (من الأحدث إلى الأقدم):"
    return f"{intro}\n\n" + "\n".join(lines)
