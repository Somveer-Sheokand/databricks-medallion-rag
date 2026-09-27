# Databricks notebook source
# MAGIC %md
# MAGIC # 05 - Stretch: incremental Bronze ingestion
# MAGIC Replaces the batch `binaryFile` read in `01_bronze_ingest.py` with Auto
# MAGIC Loader, so re-running only processes files that weren't already ingested.
# MAGIC Free Edition allows only one active Lakeflow pipeline per type, so this
# MAGIC stays a plain Structured Streaming job (`trigger(availableNow=True)`)
# MAGIC rather than a full DLT/Lakeflow pipeline.
# MAGIC
# MAGIC Once you switch to this notebook, stop running `01_bronze_ingest.py`
# MAGIC (it overwrites the table; this one appends to it) -- pick one ingestion
# MAGIC path for `bronze_raw_docs`.
# MAGIC
# MAGIC Silver (`02_silver_chunk.py`) and Gold (`03_gold_embed.py`) can be made
# MAGIC incremental the same way: since `chunk_id`/`doc_id` are deterministic
# MAGIC hashes, swap their `overwrite` writes for a `MERGE ... WHEN NOT MATCHED
# MAGIC THEN INSERT` keyed on those ids, so reruns only add rows for genuinely new
# MAGIC documents.

# COMMAND ----------

# MAGIC %pip install pypdf>=4.0.0
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import os
import sys

try:
    import rag_common  # noqa: F401
except ImportError:
    sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..", "..")))

from rag_common.config import load_config
from rag_common.hashing import doc_id_for_path

cfg = load_config()

dbutils.widgets.text("source_dataset", cfg.default_source_dataset)
source_dataset = dbutils.widgets.get("source_dataset")

volume_path = f"/Volumes/{cfg.catalog}/{cfg.schema}/raw_docs/{source_dataset}"
bronze_table = f"{cfg.catalog}.{cfg.schema}.bronze_raw_docs"
checkpoint_path = f"/Volumes/{cfg.catalog}/{cfg.schema}/_checkpoints/bronze/{source_dataset}"
schema_path = f"/Volumes/{cfg.catalog}/{cfg.schema}/_schemas/bronze/{source_dataset}"

# COMMAND ----------

from io import BytesIO

import pandas as pd
from pypdf import PdfReader
from pyspark.sql import functions as F
from pyspark.sql.types import StringType


def _extract_text(content: bytes, path: str) -> str:
    if path.lower().endswith(".pdf"):
        try:
            reader = PdfReader(BytesIO(content))
            return "\n".join(page.extract_text() or "" for page in reader.pages)
        except Exception:
            return ""
    return content.decode("utf-8", errors="ignore")


@F.pandas_udf(StringType())
def extract_text_udf(content: pd.Series, path: pd.Series) -> pd.Series:
    return pd.Series([_extract_text(c, p) for c, p in zip(content, path)])


doc_id_udf = F.udf(doc_id_for_path, StringType())

# COMMAND ----------

stream_df = (
    spark.readStream.format("cloudFiles")
    .option("cloudFiles.format", "binaryFile")
    .option("cloudFiles.schemaLocation", schema_path)
    .option("recursiveFileLookup", "true")
    .load(volume_path)
)

bronze_stream_df = (
    stream_df.withColumn("doc_id", doc_id_udf(F.col("path")))
    .withColumn("raw_text", extract_text_udf(F.col("content"), F.col("path")))
    .withColumn("source_dataset", F.lit(source_dataset))
    .withColumn("ingested_at", F.current_timestamp())
    .select("doc_id", "source_dataset", "path", "raw_text", "ingested_at")
    .filter(F.length("raw_text") > 0)
)

# COMMAND ----------

(
    bronze_stream_df.writeStream.option("checkpointLocation", checkpoint_path)
    .trigger(availableNow=True)
    .toTable(bronze_table)
)
