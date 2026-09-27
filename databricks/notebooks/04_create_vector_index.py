# Databricks notebook source
# MAGIC %md
# MAGIC # 04 - Vector Search index
# MAGIC Creates the Vector Search endpoint and a **self-managed-embedding Delta
# MAGIC Sync Index** over `gold_embedded_chunks` (self-managed because we already
# MAGIC computed the `embedding` column ourselves in 03, rather than letting
# MAGIC Databricks compute it from a text column).
# MAGIC
# MAGIC Free Edition gives you exactly **one Vector Search endpoint and one search
# MAGIC unit**, and does not support Direct Vector Access indexes -- this has to
# MAGIC be a Delta Sync Index, and `pipeline_type="TRIGGERED"` (rather than
# MAGIC continuous) is the safer assumption for that single-unit tier. Re-run the
# MAGIC last cell after every Gold refresh to pick up new/changed rows.

# COMMAND ----------

# MAGIC %pip install databricks-vectorsearch
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import os
import sys

try:
    import rag_common  # noqa: F401
except ImportError:
    sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..", "..")))

from rag_common.config import load_config

cfg = load_config()

gold_table = f"{cfg.catalog}.{cfg.schema}.gold_embedded_chunks"
index_name = f"{cfg.catalog}.{cfg.schema}.{cfg.vector_search.index_name}"

# COMMAND ----------

from databricks.vector_search.client import VectorSearchClient

vsc = VectorSearchClient()

if not vsc.endpoint_exists(cfg.vector_search.endpoint_name):
    vsc.create_endpoint_and_wait(name=cfg.vector_search.endpoint_name, endpoint_type="STANDARD")

# COMMAND ----------

if not vsc.index_exists(cfg.vector_search.endpoint_name, index_name):
    vsc.create_delta_sync_index_and_wait(
        endpoint_name=cfg.vector_search.endpoint_name,
        index_name=index_name,
        source_table_name=gold_table,
        pipeline_type="TRIGGERED",
        primary_key="chunk_id",
        embedding_vector_column="embedding",
        embedding_dimension=cfg.embedding.dimension,
    )
else:
    vsc.get_index(cfg.vector_search.endpoint_name, index_name).sync()

# COMMAND ----------

print(f"Index ready: {index_name} on endpoint {cfg.vector_search.endpoint_name}")
vsc.get_index(cfg.vector_search.endpoint_name, index_name).describe()
