# Databricks side

## One-time setup

1. Sign up for [Databricks Free Edition](https://www.databricks.com/learn/free-edition). Optionally verify with LinkedIn to unblock outbound internet access and serverless GPU — not required for the default setup below, which avoids needing internet access from inside notebooks.
2. In the workspace, add this repo as a **Git folder** (Repos): `Workspace > Git folders > Add`. This puts the repo root on `sys.path` for every notebook, which is what lets `01_bronze_ingest.py` etc. do `import rag_common`.
3. Open Catalog Explorer and confirm your account's default catalog name. Edit `catalog:` in `config/config.yaml` (repo root) if it isn't `workspace`.
4. Under **Serving > Foundation Models**, confirm an embedding endpoint is available (the config default is `databricks-gte-large-en`, 1024-dim). If your workspace doesn't expose one, set `embedding.provider: local` in `config/config.yaml` instead (uses a CPU sentence-transformers model — slower, no workspace dependency).

## Getting documents in

Free Edition restricts outbound internet from notebooks, so download documents locally and upload them, rather than fetching them from inside a notebook:

```bash
# from the repo root, on your own machine
python scripts/fetch_arxiv_sample.py --category cs.CL --target-count 20
databricks configure   # one-time: paste your workspace URL + a personal access token
databricks fs cp -r ./local_docs/arxiv_sample dbfs:/Volumes/<catalog>/rag_demo/raw_docs/arxiv_sample
```

Swap in AWS docs or Indian government policy PDFs by uploading them to a differently-named folder under `raw_docs/` and changing `default_source_dataset` in `config/config.yaml` (or the `source_dataset` widget when running notebooks).

## Running the pipeline

In order, either interactively or via the job below:

1. `notebooks/00_setup.py` — creates catalog/schema/volume.
2. `notebooks/01_bronze_ingest.py` — raw text extraction.
3. `notebooks/02_silver_chunk.py` — cleaning + chunking.
4. `notebooks/03_gold_embed.py` — embeddings, enables Change Data Feed.
5. `notebooks/04_create_vector_index.py` — creates/syncs the Vector Search index.

Stretch: `notebooks/05_incremental_bronze.py` replaces step 2 with Auto Loader so reruns only process new files. Silver and Gold (steps 3-4) are already incremental regardless of which Bronze notebook you use — each anti-joins against what it's already processed and `MERGE`s in only the new rows (see `README.md`'s design-choices section).

## Scheduling

`jobs/rag_pipeline_job.json` chains steps 2-5 as a Databricks Job. It has no cluster spec, so it runs on serverless compute by default (Free Edition has no other option anyway). Before creating it:

- Replace every `<UPDATE_ME>` with this Git folder's workspace path (visible in the Git folder's UI, typically `/Workspace/Repos/<you>/varnam-agent` or `/Workspace/Users/<you>/varnam-agent`).

```bash
databricks jobs create --json @databricks/jobs/rag_pipeline_job.json
databricks jobs run-now --job-id <id-from-previous-command>
```

Free Edition caps you at 5 concurrent job tasks — this DAG uses 4, sequentially, so you're well under it.
