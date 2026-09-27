from rag_common.cleaning import clean_text, strip_references_section


def test_clean_text_collapses_whitespace():
    assert clean_text("hello   world\t\tfoo") == "hello world foo"


def test_clean_text_collapses_blank_lines():
    assert clean_text("a\n\n\n\n\nb") == "a\n\nb"


def test_clean_text_handles_none_and_empty():
    assert clean_text(None) == ""
    assert clean_text("") == ""


def test_clean_text_strips_control_chars():
    assert clean_text("hello\x00world") == "helloworld"


def test_clean_text_strips_leading_trailing_whitespace():
    assert clean_text("  hello world  ") == "hello world"


def test_strip_references_section_truncates_at_heading():
    text = "Intro text here.\n\nReferences\n\n[1] Someone. Some paper. arXiv:1234.5678, 2024."
    assert strip_references_section(text) == "Intro text here."


def test_strip_references_section_case_insensitive_and_bibliography():
    text = "Body.\n\nBIBLIOGRAPHY\n\n[1] X."
    assert strip_references_section(text) == "Body."


def test_strip_references_section_no_heading_returns_unchanged():
    text = "Just a paper with no references heading at all."
    assert strip_references_section(text) == text


def test_strip_references_section_uses_last_match():
    text = "References\n\nSection heading mentioned early, not real bibliography.\n\nBody.\n\nReferences\n\n[1] X."
    expected = "References\n\nSection heading mentioned early, not real bibliography.\n\nBody."
    assert strip_references_section(text) == expected
