from app.services.language import (
    split_inline_numbered_items,
    strip_empty_numbered_items,
    strip_meta_source_phrases,
)


def test_strip_trailing_empty_numbered_line():
    text = "١٠. نص كامل.\n١١. بند آخر.\n**١٢.**"
    assert strip_empty_numbered_items(text) == "١٠. نص كامل.\n١١. بند آخر."


def test_strip_western_number_marker():
    text = "10. Done.\n11. Also done.\n12."
    assert strip_empty_numbered_items(text) == "10. Done.\n11. Also done."


def test_strip_kb_meta_phrases():
    text = "أهلاً بك. أنا مساعد يجيب عن سيرة جهاد بناءً على قاعدة المعرفة المتوفرة."
    cleaned = strip_meta_source_phrases(text)
    assert "قاعدة المعرفة" not in cleaned
    assert "أهلاً بك" in cleaned


def test_split_inline_numbered_items():
    text = "١. المسمى الوظيفي: مهندس ٢. الخبرات العامة: صيانة ٣. المهارات"
    result = split_inline_numbered_items(text)
    assert result.startswith("١. المسمى")
    assert "\n٢. الخبرات" in result
    assert "\n٣. المهارات" in result


def test_keeps_numbered_lines_with_content():
    text = "١. مهارة أولى\n٢. مهارة ثانية"
    assert strip_empty_numbered_items(text) == text
