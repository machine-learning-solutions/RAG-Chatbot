from app.services.language import strip_empty_numbered_items


def test_strip_trailing_empty_numbered_line():
    text = "١٠. نص كامل.\n١١. بند آخر.\n**١٢.**"
    assert strip_empty_numbered_items(text) == "١٠. نص كامل.\n١١. بند آخر."


def test_strip_western_number_marker():
    text = "10. Done.\n11. Also done.\n12."
    assert strip_empty_numbered_items(text) == "10. Done.\n11. Also done."


def test_keeps_numbered_lines_with_content():
    text = "١. مهارة أولى\n٢. مهارة ثانية"
    assert strip_empty_numbered_items(text) == text
