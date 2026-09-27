"""Two chunking strategies, picked via config/config.yaml's chunking.strategy.

Chunk size/overlap are expressed in words here as a zero-dependency proxy for
tokens. If you need chunks that land precisely under your embedding model's
token limit, swap the `len(x.split())` measure for a real tokenizer (tiktoken
or the model's own AutoTokenizer) -- the split points below don't change.
"""
from __future__ import annotations

from typing import List


def chunk_fixed_size(text: str, chunk_size: int = 512, overlap: int = 75) -> List[str]:
    """Naive baseline: slide a fixed-size window over the word stream."""
    words = text.split()
    if not words:
        return []

    step = max(chunk_size - overlap, 1)
    chunks: List[str] = []
    for start in range(0, len(words), step):
        chunk_words = words[start : start + chunk_size]
        if not chunk_words:
            break
        chunks.append(" ".join(chunk_words))
        if start + chunk_size >= len(words):
            break
    return chunks


def chunk_recursive(text: str, chunk_size: int = 512, overlap: int = 75) -> List[str]:
    """Structure-aware: pack whole paragraphs together up to chunk_size, and
    only fall back to fixed-size splitting for a paragraph too long to fit in
    one chunk on its own.

    Overlap is carried across paragraph-packing boundaries: when a chunk is
    flushed, its trailing `overlap` words seed the next chunk, the same way
    `chunk_fixed_size`'s sliding window does.
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: List[str] = []
    current: List[str] = []
    current_len = 0

    def flush_and_carry() -> None:
        nonlocal current, current_len
        if not current:
            return
        chunks.append(" ".join(current))
        current = current[-overlap:] if overlap > 0 else []
        current_len = len(current)

    for para in paragraphs:
        para_words = para.split()

        if len(para_words) > chunk_size:
            flush_and_carry()
            current = []
            current_len = 0
            chunks.extend(chunk_fixed_size(para, chunk_size, overlap))
            continue

        if current_len + len(para_words) > chunk_size:
            flush_and_carry()

        current.extend(para_words)
        current_len += len(para_words)

    if current:
        chunks.append(" ".join(current))
    return chunks
