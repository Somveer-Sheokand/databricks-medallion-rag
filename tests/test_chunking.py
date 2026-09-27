from rag_common.chunking import chunk_fixed_size, chunk_recursive


def test_chunk_fixed_size_overlap():
    text = " ".join(f"word{i}" for i in range(10))
    chunks = chunk_fixed_size(text, chunk_size=4, overlap=1)
    assert chunks[0] == "word0 word1 word2 word3"
    assert chunks[1].startswith("word3")


def test_chunk_fixed_size_empty():
    assert chunk_fixed_size("") == []


def test_chunk_recursive_packs_short_paragraphs_together():
    text = "para one is short.\n\npara two is also short."
    chunks = chunk_recursive(text, chunk_size=100, overlap=10)
    assert len(chunks) == 1
    assert "para one" in chunks[0] and "para two" in chunks[0]


def test_chunk_recursive_splits_oversized_paragraph():
    long_para = " ".join(f"word{i}" for i in range(50))
    chunks = chunk_recursive(long_para, chunk_size=10, overlap=2)
    assert len(chunks) > 1


def test_chunk_recursive_starts_new_chunk_when_full():
    para_a = " ".join(f"a{i}" for i in range(8))
    para_b = " ".join(f"b{i}" for i in range(8))
    chunks = chunk_recursive(f"{para_a}\n\n{para_b}", chunk_size=10, overlap=2)
    assert len(chunks) == 2
    assert "a0" in chunks[0] and "b0" in chunks[1]


def test_chunk_recursive_carries_overlap_across_paragraphs():
    para_a = " ".join(f"a{i}" for i in range(8))
    para_b = " ".join(f"b{i}" for i in range(8))
    chunks = chunk_recursive(f"{para_a}\n\n{para_b}", chunk_size=10, overlap=2)
    # last 2 words of chunk 0 ("a6 a7") should seed the start of chunk 1
    assert chunks[1].startswith("a6 a7 b0")
