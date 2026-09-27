# Databricks notebook source
# MAGIC %md
# MAGIC # 00 - Setup
# MAGIC Creates the catalog/schema/volume this pipeline uses and prints where to
# MAGIC upload source documents. Run this once (safe to re-run: everything here is
# MAGIC `IF NOT EXISTS`).
# MAGIC
# MAGIC Sync this whole repo into Databricks as a **Git folder** (Repos) first --
# MAGIC that's what puts `rag_common` on `sys.path` for every notebook below.

# COMMAND ----------

# MAGIC %pip install -r ../requirements-notebook.txt
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import os
import sys

try:
    import rag_common  # noqa: F401
except ImportError:
    # Fallback if this notebook is opened outside a synced Git folder.
    sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..", "..")))

from rag_common.config import load_config

cfg = load_config()

# COMMAND ----------

dbutils.widgets.text("catalog", cfg.catalog)
dbutils.widgets.text("schema", cfg.schema)
dbutils.widgets.text("source_dataset", cfg.default_source_dataset)

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
source_dataset = dbutils.widgets.get("source_dataset")

# COMMAND ----------

spark.sql(f"CREATE CATALOG IF NOT EXISTS {catalog}")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.{schema}")
spark.sql(f"CREATE VOLUME IF NOT EXISTS {catalog}.{schema}.raw_docs")

volume_path = f"/Volumes/{catalog}/{schema}/raw_docs/{source_dataset}"
print(f"Upload source files to: {volume_path}")
print("From your local machine (after `databricks configure`):")
print(f"  databricks fs cp -r ./local_docs/{source_dataset} dbfs:{volume_path}")
