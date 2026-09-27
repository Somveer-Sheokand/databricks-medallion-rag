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
- `rag_common/` — pure-Python, no PySpark/Databricks import required.
  Chunking, cleaning, extraction (PDF, with a pdfplumber fallback), hashing,
  embedding, config — unit-tested in `tests/`.
- `databricks/notebooks/` — the actual pipeline, run inside Databricks. See
  `databricks/README.md` for setup and run order.
- `mcp_server/` — the MCP server an agent talks to. `retrieve_context` embeds
  the query and searches the Gold vector index.
- `scripts/fetch_arxiv_sample.py` — downloads the default sample dataset
  locally, for upload to a Unity Catalog volume (see `databricks/README.md`).
- `scripts/eval_retrieval.py` — one distinctive question per ingested
  document, checked against the live index (needs `.env`/Databricks access,
  so it's opt-in, not part of `pytest`). Run it after touching chunking,
  cleaning, or the embedding model to catch a retrieval-quality regression:
  `python scripts/eval_retrieval.py`. Update `EVAL_CASES` if you re-fetch a
  different document set — see the script's docstring.
- `app/` — a small Flask chat UI (retrieve + generate, with cited sources),
  deployed as a **Databricks App** so it runs inside the workspace with its
  own service-principal auth — no token ever leaves Databricks. See
  "Live chat app" below.

## Live chat app

`app/app.py` is a minimal RAG chat page: it embeds the question, calls
`WorkspaceClient().vector_search_indexes.query_index` against the Gold index,
then generates a cited answer via a chat serving endpoint
(`databricks-meta-llama-3-3-70b-instruct`). Deployed as a Databricks App, so
visitors authenticate with their own Databricks login — this isn't an
anonymous public site, but nothing needs a personal access token either.

The client keeps a rolling conversation history and sends it with each
request, so follow-ups ("which one is faster?") resolve correctly — the chat
model sees prior turns. **Retrieval itself is still single-turn**: only the
latest question gets embedded and searched, not a history-aware rewrite of
it, so a follow-up whose retrieval-relevant content was only implied by
earlier turns may come back empty. Query rewriting (e.g. having the chat
model expand "which one is faster?" into a self-contained question before
embedding it) would close that gap but isn't implemented.

Deploy or redeploy after code changes:

```bash
python -m venv .venv && .venv/Scripts/activate  # if not already set up
pip install -e ".[dev]"
cp .env.example .env   # fill in DATABRICKS_HOST / DATABRICKS_TOKEN once, for this script only
python - <<'PY'
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path.cwd() / ".env")
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.workspace import ImportFormat
from databricks.sdk.service.apps import AppDeployment

w = WorkspaceClient()
me = w.current_user.me().user_name
source_path = f"/Workspace/Users/{me}/varnam-rag-chat-src"
for fname in ["app.py", "app.yaml", "requirements.txt"]:
    w.workspace.upload(f"{source_path}/{fname}", (Path("app") / fname).read_bytes(),
                        format=ImportFormat.RAW, overwrite=True)
print(w.apps.deploy_and_wait(app_name="varnam-rag-chat",
                              app_deployment=AppDeployment(source_code_path=source_path)).status)
PY
```

(First-time setup also needs `w.apps.create_and_wait(...)` with the app's
resources declared — see git history for the one-off creation script — since
`app.py` needs `CAN_QUERY` on both serving endpoints and `SELECT` on
`workspace.rag_demo.gold_chunks_index` granted to its own service principal.)

**If you ever drop and recreate the vector index** (e.g. rebuilding from
scratch), the app will start failing with `PermissionDenied: You do not have
the SELECT privilege on ...` even though nothing about the app changed —
the UC grant is tied to that specific index *object*, not its name, and a
recreated index is a new object. Re-grant it (the error message gives you
the exact principal to grant to):

```sql
GRANT SELECT ON TABLE workspace.rag_demo.gold_chunks_index TO `<service-principal-client-id>`;
```

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
- **Dynamic partition overwrite on Bronze's write.** `01_bronze_ingest.py`
  writes with `.option("partitionOverwriteMode", "dynamic")`. Without it,
  Spark's default *static* overwrite mode replaces the entire table on every
  write — so ingesting a second dataset (e.g. AWS docs after arXiv) would
  silently delete the first one's `source_dataset` partition, even though the
  table is partitioned by it. Dynamic mode scopes the overwrite to only the
  partitions the current write actually produced, which is what makes
  multiple datasets coexist in one set of tables (and what the MCP tool's
  `source_filter` assumes).
- **Silver and Gold are incremental, keyed on `doc_id`/`chunk_id`.**
  `02_silver_chunk.py` and `03_gold_embed.py` anti-join against what's already
  in their output table, process only the new rows, and `MERGE ... WHEN NOT
  MATCHED THEN INSERT` the result — a rerun neither re-chunks unchanged
  documents nor pays to re-embed unchanged chunks. This trades off one thing:
  a rerun after changing the chunking strategy or fixing a cleaning bug won't
  retroactively reprocess documents already in Silver/Gold — drop the table
  (or delete the affected rows) to force a full reprocess.

## Swapping the document set

Everything is parameterized by `source_dataset` (a widget in the notebooks,
`default_source_dataset` in config.yaml) and a matching subfolder under the
Unity Catalog volume — swapping arXiv papers for AWS docs or Indian
government policy PDFs is a folder + config change, not a code change.
