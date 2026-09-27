from rag_common import extraction


def test_extract_text_non_pdf_decodes_utf8():
    assert extraction.extract_text(b"hello world", "notes.txt") == "hello world"


def test_extract_text_non_pdf_ignores_bad_bytes():
    assert extraction.extract_text(b"hello \xff world", "notes.txt") == "hello  world"


def test_looks_garbled_detects_uni_placeholders():
    garbled = "/uni000001fb/uni0000008a/uni000001fc" * 5
    assert extraction._looks_garbled(garbled) is True


def test_looks_garbled_false_for_normal_text():
    assert extraction._looks_garbled("This is a perfectly normal sentence.") is False


def test_looks_garbled_true_for_empty():
    assert extraction._looks_garbled("") is True


def test_extract_pdf_text_uses_pypdf_when_clean(monkeypatch):
    monkeypatch.setattr(extraction, "_extract_with_pypdf", lambda content: "clean readable text")
    monkeypatch.setattr(
        extraction, "_extract_with_pdfplumber",
        lambda content: (_ for _ in ()).throw(AssertionError("should not be called")),
    )
    assert extraction.extract_pdf_text(b"fake-pdf-bytes") == "clean readable text"


def test_extract_pdf_text_falls_back_when_pypdf_garbled(monkeypatch):
    monkeypatch.setattr(extraction, "_extract_with_pypdf", lambda content: "/uni0000" * 50)
    monkeypatch.setattr(extraction, "_extract_with_pdfplumber", lambda content: "clean fallback text")
    assert extraction.extract_pdf_text(b"fake-pdf-bytes") == "clean fallback text"


def test_extract_pdf_text_falls_back_when_pypdf_raises(monkeypatch):
    def _raise(content):
        raise ValueError("corrupt pdf")

    monkeypatch.setattr(extraction, "_extract_with_pypdf", _raise)
    monkeypatch.setattr(extraction, "_extract_with_pdfplumber", lambda content: "clean fallback text")
    assert extraction.extract_pdf_text(b"fake-pdf-bytes") == "clean fallback text"


def test_extract_pdf_text_keeps_longer_when_both_garbled(monkeypatch):
    monkeypatch.setattr(extraction, "_extract_with_pypdf", lambda content: "/uni0000" * 10)
    monkeypatch.setattr(extraction, "_extract_with_pdfplumber", lambda content: "/uni0000" * 20)
    result = extraction.extract_pdf_text(b"fake-pdf-bytes")
    assert result == "/uni0000" * 20
