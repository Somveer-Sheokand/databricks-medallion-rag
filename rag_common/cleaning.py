"""Text cleaning for the Silver layer.

Deliberately built from built-in string/regex operations rather than a UDF —
these are cheap, embarrassingly parallel transforms where Spark's native
functions (or a plain Python function on the driver-side rows) outperform a
UDF's per-row Python/JVM serialization cost. The chunking step next to this
one is the contrasting case where a pandas_udf earns its keep.
"""
from __future__ import annotations

import re
from typing import Optional

_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
_WHITESPACE_RE = re.compile(r"[ \t\f\v]+")
_BLANK_LINES_RE = re.compile(r"\n{3,}")


def clean_text(text: Optional[str]) -> str:
    if not text:
        return ""
    text = _CONTROL_CHARS_RE.sub("", text)
    text = _WHITESPACE_RE.sub(" ", text)
    text = _BLANK_LINES_RE.sub("\n\n", text)
    return text.strip()
