from rag_common.cleaning import clean_text


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
