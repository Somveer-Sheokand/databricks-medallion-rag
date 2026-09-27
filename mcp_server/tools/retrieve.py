"""The retrieval tool the agent calls: embeds a query and searches the Gold
Delta Sync vector index for the most relevant chunks.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Any, Dict, List, Optional

from rag_common.config import Config, load_config
from rag_common.embeddings import embed_texts_fmapi, embed_texts_local

from mcp_server.databricks_client import get_vector_search_client, get_workspace_client


@lru_cache(maxsize=1)
def _cached_config() -> Config:
    # The MCP server is long-running; re-reading and re-parsing config.yaml on
    # every retrieve_context call would be pure per-request overhead. Safe to
    # cache for the process lifetime -- restart the server to pick up edits.
    return load_config()


def _embed_query(query: str, cfg: Config) -> List[float]:
    if cfg.embedding.provider == "fmapi":
        return embed_texts_fmapi([query], cfg.embedding.endpoint_name, get_workspace_client())[0]
    return embed_texts_local([query], cfg.embedding.local_model_name)[0]


def search_documents(query: str, top_k: Optional[int] = None,
                       source_filter: Optional[str] = None) -> List[Dict[str, Any]]:
    """Return the top-k most relevant chunks for `query`.

    Each result: {chunk_text, source, doc_id, chunk_id, score}.
    """
    cfg = _cached_config()
    top_k = top_k if top_k is not None else cfg.retrieval.default_top_k

    query_vector = _embed_query(query, cfg)

    index_name = f"{cfg.catalog}.{cfg.schema}.{cfg.vector_search.index_name}"
    index = get_vector_search_client().get_index(cfg.vector_search.endpoint_name, index_name)

    # Equality-filter syntax for a STANDARD endpoint; storage-optimized
    # endpoints use a different (string, SQL-like) filter syntax -- see
    # databricks/notebooks/04_create_vector_index.py.
    filters = {"source_dataset": source_filter} if source_filter else None

    results = index.similarity_search(
        query_vector=query_vector,
        columns=["chunk_id", "doc_id", "chunk_text", "source_dataset"],
        num_results=top_k,
        filters=filters,
    )

    columns = [c["name"] for c in results["manifest"]["columns"]]
    hits: List[Dict[str, Any]] = []
    for row in results["result"]["data_array"]:
        record = dict(zip(columns, row))
        hits.append({
            "chunk_text": record.get("chunk_text"),
            "source": record.get("source_dataset"),
            "doc_id": record.get("doc_id"),
            "chunk_id": record.get("chunk_id"),
            "score": row[-1],  # similarity_search appends the score as the last element
        })
    return hits
