# varnam-agent

A RAG pipeline built on Databricks Free Edition, following the medallion
architecture (Bronze/Silver/Gold), with retrieval exposed to an agent as an
MCP tool.

```
Bronze (raw text) -> Silver (cleaned + chunked) -> Gold (embedded)
                                                       |
                                          Vector Search Delta Sync Index
                                                       |
                                    mcp_server.tools.retrieve.search_documents
                                                       |
                                              your agent, via MCP
```

## Layout

- `config/config.yaml` — single source of truth for catalog/schema names,
  chunking strategy, embedding provider/model, vector search names. Read by
  both sides below.
- `rag_common/` — pure-Python, no PySpark/Databricks import required. Chunking,
  cleaning, hashing, embedding, config — unit-tested in `tests/`.
- `databricks/notebooks/` — the actual pipeline, run inside Databricks. See
  `databricks/README.md` for setup and run order.
- `mcp_server/` — the MCP server an agent talks to. `retrieve_context` embeds
  the query and searches the Gold vector index.
- `scripts/fetch_arxiv_sample.py` — downloads the default sample dataset
  locally, for upload to a Unity Catalog volume (see `databricks/README.md`).

## Quickstart

1. **Databricks side**: follow `databricks/README.md` end to end — Free
   Edition signup, Git folder sync, document upload, running the 4 pipeline
   notebooks in order.
2. **Local/agent side**:

   ```bash
   python -m venv .venv
   .venv/Scripts/activate          # or: source .venv/bin/activate on macOS/Linux
   pip install -e ".[dev]"
   cp .env.example .env            # fill in DATABRICKS_HOST / DATABRICKS_TOKEN
   pytest                          # rag_common + retrieval-tool unit tests, no Databricks needed
   python -m mcp_server.server     # starts the MCP server (stdio transport)
   ```

   If `config/config.yaml` has `embedding.provider: local`, also
   `pip install -e ".[local-embeddings]"` — needed by both the Gold notebook's
   fallback path and the MCP server's query-time embedding.

3. Point your agent's MCP client at `python -m mcp_server.server`; it will see
   one tool, `retrieve_context(query, top_k=5, source_filter=None)`.

## Key design choices (and why)

- **Self-managed embeddings, not Databricks-managed.** Gold computes the
  `embedding` column itself (via `ai_query()` or a local `pandas_udf`) instead
  of letting the vector index compute it from a text column — this is what
  makes the "UDFs vs built-ins" comparison in `databricks/notebooks/03_gold_embed.py`
  possible, and it's why the vector index in `04_create_vector_index.py` is
  created with `embedding_vector_column`, not `embedding_source_column`.
- **Delta Sync Index, triggered sync.** Free Edition's single Vector Search
  unit doesn't support Direct Vector Access indexes; the pipeline job triggers
  a sync as its last step rather than relying on continuous sync. Because of
  that, Gold's writer also raises `delta.deletedFileRetentionDuration` to 30
  days: a triggered (non-continuous) index's sync starts failing once more
  than the retention window has passed since the last sync, and the Delta
  default of 7 days is too short for a table you don't re-sync daily.
- **Upload-to-volume ingestion, not fetch-in-notebook.** Free Edition
  restricts outbound internet from notebooks unless you verify with LinkedIn.
  `scripts/fetch_arxiv_sample.py` runs on your machine instead, then you
  `databricks fs cp` the result up.
- **Query-time embedding calls the serving endpoint directly**
  (`rag_common/embeddings.py:embed_texts_fmapi`), not `ai_query()` — that
  function only exists inside a Spark SQL/DataFrame context, and the MCP
  server runs as an external process.
- **Dynamic partition overwrite on every Bronze/Silver/Gold write.** Each of
  `01_bronze_ingest.py`/`02_silver_chunk.py`/`03_gold_embed.py` writes with
  `.option("partitionOverwriteMode", "dynamic")`. Without it, Spark's default
  *static* overwrite mode replaces the entire table on every write — so
  ingesting a second dataset (e.g. AWS docs after arXiv) would silently delete
  the first one's `source_dataset` partition, even though the table is
  partitioned by it. Dynamic mode scopes the overwrite to only the partitions
  the current write actually produced, which is what makes multiple datasets
  coexist in one set of tables (and what the MCP tool's `source_filter`
  assumes).

## Swapping the document set

Everything is parameterized by `source_dataset` (a widget in the notebooks,
`default_source_dataset` in config.yaml) and a matching subfolder under the
Unity Catalog volume — swapping arXiv papers for AWS docs or Indian
government policy PDFs is a folder + config change, not a code change.
