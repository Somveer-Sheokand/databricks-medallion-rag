# Databricks notebook source
# MAGIC %md
# MAGIC # 01 - Bronze: raw ingestion
# MAGIC Reads uploaded files from the Unity Catalog volume, extracts raw text, and
# MAGIC writes `bronze_raw_docs`. Batch/overwrite version -- see
# MAGIC `05_incremental_bronze.py` for the Auto Loader stretch variant; don't run
# MAGIC both against the same table.
# MAGIC
# MAGIC Demonstrates a **scalar UDF** (`doc_id_for_path`, one Python call per row)
# MAGIC next to a **pandas_udf** (`extract_text_udf`, batched) on the same
# MAGIC DataFrame, for a direct before/after comparison in the Spark UI.

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

raw_df = (
    spark.read.format("binaryFile")
    .option("recursiveFileLookup", "true")
    .load(volume_path)
)

bronze_df = (
    raw_df.withColumn("doc_id", doc_id_udf(F.col("path")))
    .withColumn("raw_text", extract_text_udf(F.col("content"), F.col("path")))
    .withColumn("source_dataset", F.lit(source_dataset))
    .withColumn("ingested_at", F.current_timestamp())
    .select("doc_id", "source_dataset", "path", "raw_text", "ingested_at")
    .filter(F.length("raw_text") > 0)
)

# COMMAND ----------

(
    bronze_df.write.mode("overwrite")
    # Dynamic (not Spark's default static) partition overwrite: this write only
    # ever contains rows for `source_dataset`, so a static overwrite would
    # silently delete every OTHER dataset's partition from the table. This is
    # what makes "swap in a new document set" additive rather than destructive.
    .option("partitionOverwriteMode", "dynamic")
    .option("overwriteSchema", "true")
    .partitionBy("source_dataset")
    .saveAsTable(bronze_table)
)

display(spark.table(bronze_table).limit(10))
