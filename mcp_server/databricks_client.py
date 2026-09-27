"""Lazily-constructed, cached Databricks clients.

Auth is picked up the standard databricks-sdk way: DATABRICKS_HOST /
DATABRICKS_TOKEN env vars (see .env.example), or a ~/.databrickscfg profile,
or a service principal -- see the databricks-sdk docs for the full precedence
order. Nothing here needs to know which one is in play.
"""
from __future__ import annotations

from functools import lru_cache


@lru_cache(maxsize=1)
def get_workspace_client():
    from databricks.sdk import WorkspaceClient

    return WorkspaceClient()


@lru_cache(maxsize=1)
def get_vector_search_client():
    # databricks-vectorsearch is the established package this project depends
    # on (see requirements.txt). Databricks has begun rebranding this surface
    # as "AI Search"; if your environment only has the newer
    # databricks-ai-search package installed, fall back to it transparently.
    try:
        from databricks.vector_search.client import VectorSearchClient

        return VectorSearchClient()
    except ImportError:
        from databricks.ai_search.client import AISearchClient

        return AISearchClient()
