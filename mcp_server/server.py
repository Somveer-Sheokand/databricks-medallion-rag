"""MCP server exposing the RAG retrieval tool built in databricks/notebooks/.

Run with:  python -m mcp_server.server
or, once installed (`pip install -e .`):  varnam-agent-mcp
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

from mcp_server.tools.retrieve import search_documents

load_dotenv()  # picks up DATABRICKS_HOST / DATABRICKS_TOKEN from a local .env, if present

mcp = FastMCP("varnam-agent-rag")


@mcp.tool()
def retrieve_context(query: str, top_k: int = 5,
                       source_filter: Optional[str] = None) -> List[Dict[str, Any]]:
    """Retrieve the most relevant document chunks for `query` from the RAG vector index.

    Args:
        query: natural-language question or search text.
        top_k: number of chunks to return.
        source_filter: optional source_dataset value to restrict the search to.
    """
    return search_documents(query, top_k=top_k, source_filter=source_filter)


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
