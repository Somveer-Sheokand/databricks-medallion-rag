"""Embedding helpers shared by the Gold notebook (batch) and the MCP retrieval
tool (single query at request time).

Two providers, matching config/config.yaml's embedding.provider:

- "fmapi": calls a Databricks Foundation Model API serving endpoint directly
  via the Databricks SDK. Inside a notebook this is what the built-in
  ai_query() SQL function wraps; outside Databricks (i.e. from the MCP
  server process) there is no ai_query(), so we hit the same serving endpoint
  over its REST API via WorkspaceClient instead.
- "local": a CPU sentence-transformers model, for workspaces/accounts where a
  Foundation Model API embedding endpoint isn't available. Requires the
  `local-embeddings` extra (see pyproject.toml) and, in notebooks,
  databricks/requirements-notebook.txt.

Whichever provider you pick, the SAME provider/model must be used for both
Gold embedding generation and query-time embedding, or similarity scores will
be meaningless -- that's why both paths live in this one module.
"""
from __future__ import annotations

from functools import lru_cache
from typing import List, Optional


@lru_cache(maxsize=4)
def _load_local_model(model_name: str):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


def embed_texts_local(texts: List[str], model_name: str) -> List[List[float]]:
    model = _load_local_model(model_name)
    return model.encode(list(texts), show_progress_bar=False).tolist()


def embed_texts_fmapi(texts: List[str], endpoint_name: str, workspace_client=None) -> List[List[float]]:
    if workspace_client is None:
        from databricks.sdk import WorkspaceClient

        workspace_client = WorkspaceClient()

    response = workspace_client.serving_endpoints.query(name=endpoint_name, input=list(texts))
    return [item.embedding for item in response.data]


def embed_texts(texts: List[str], provider: str, endpoint_name: str, local_model_name: str,
                 workspace_client: Optional[object] = None) -> List[List[float]]:
    if provider == "fmapi":
        return embed_texts_fmapi(texts, endpoint_name, workspace_client)
    if provider == "local":
        return embed_texts_local(texts, local_model_name)
    raise ValueError(f"Unknown embedding provider: {provider!r} (expected 'fmapi' or 'local')")
