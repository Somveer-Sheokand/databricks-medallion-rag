"""Deterministic ID helpers shared by Bronze ingestion and Silver chunking.

Determinism matters here: re-running Bronze on the same file must produce the
same doc_id (needed for the incremental/anti-join pattern in
databricks/notebooks/05_incremental_bronze.py), and chunk_id must be stable
across reruns since it's the Delta Sync Index primary key.
"""
from __future__ import annotations

import hashlib


def doc_id_for_path(path: str) -> str:
    return hashlib.sha1(path.encode("utf-8")).hexdigest()[:16]


def chunk_id_for(doc_id: str, chunk_index: int) -> str:
    return f"{doc_id}-{chunk_index:05d}"
