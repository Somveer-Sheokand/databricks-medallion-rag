# Databricks notebook source
# MAGIC %md
# MAGIC # 02 - Silver: clean + chunk
# MAGIC Cleans Bronze text with plain built-in string/regex functions (including
# MAGIC dropping each paper's bibliography -- see `rag_common.cleaning.
# MAGIC strip_references_section`, which keeps retrieval from surfacing
# MAGIC citation-list chunks instead of actual content), then chunks it with
# MAGIC whichever strategy `chunking.strategy` in config.yaml selects ("fixed" or
# MAGIC "recursive" -- see `rag_common/chunking.py`), and writes `silver_chunks`.
# MAGIC Chunking runs as a `pandas_udf`: it does real per-row work (splitting,
# MAGIC packing), unlike the cleaning step, so batching pays off.
# MAGIC
# MAGIC **Incremental**: only chunks documents whose `doc_id` isn't already in
# MAGIC `silver_chunks` (an anti-join against Bronze), then `MERGE ... WHEN NOT
# MAGIC MATCHED THEN INSERT`s the result -- reruns only add rows for genuinely
# MAGIC new documents, without re-chunking (or re-embedding, in Gold) everything
# MAGIC that's already there.

# COMMAND ----------

import os
import sys

try:
    import rag_common  # noqa: F401
except ImportError:
    sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..", "..")))

from rag_common.chunking import chunk_fixed_size, chunk_recursive
from rag_common.cleaning import clean_text, strip_references_section
from rag_common.config import load_config
from rag_common.hashing import chunk_id_for

cfg = load_config()
bronze_table = f"{cfg.catalog}.{cfg.schema}.bronze_raw_docs"
silver_table = f"{cfg.catalog}.{cfg.schema}.silver_chunks"

# COMMAND ----------

import pandas as pd
from delta.tables import DeltaTable
from pyspark.sql import functions as F
from pyspark.sql.types import ArrayType, StringType, StructField, StructType

bronze_df = spark.table(bronze_table)

silver_exists = spark.catalog.tableExists(silver_table)
if silver_exists:
    already_processed = spark.table(silver_table).select("doc_id").distinct()
    bronze_df = bronze_df.join(already_processed, on="doc_id", how="left_anti")


def _clean_and_strip_refs(text: str) -> str:
    return strip_references_section(clean_text(text))


clean_text_udf = F.udf(_clean_and_strip_refs, StringType())
cleaned_df = bronze_df.withColumn("clean_text", clean_text_udf(F.col("raw_text")))

# COMMAND ----------

chunk_schema = ArrayType(
    StructType(
        [
            StructField("chunk_index", StringType()),
            StructField("chunk_text", StringType()),
        ]
    )
)

CHUNK_STRATEGY = cfg.chunking.strategy
CHUNK_SIZE = cfg.chunking.chunk_size
CHUNK_OVERLAP = cfg.chunking.overlap


@F.pandas_udf(chunk_schema)
def chunk_udf(text: pd.Series) -> pd.Series:
    fn = chunk_fixed_size if CHUNK_STRATEGY == "fixed" else chunk_recursive

    def _chunks(t: str):
        return [
            {"chunk_index": str(i), "chunk_text": c}
            for i, c in enumerate(fn(t, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP))
        ]

    return text.apply(_chunks)


# COMMAND ----------

chunk_id_udf = F.udf(lambda doc_id, idx: chunk_id_for(doc_id, int(idx)), StringType())

chunked_df = (
    cleaned_df.withColumn("chunks", chunk_udf(F.col("clean_text")))
    .withColumn("chunk", F.explode("chunks"))
    .select(
        chunk_id_udf("doc_id", F.col("chunk.chunk_index")).alias("chunk_id"),
        "doc_id",
        "source_dataset",
        F.col("chunk.chunk_index").cast("int").alias("chunk_index"),
        F.col("chunk.chunk_text").alias("chunk_text"),
        F.length(F.col("chunk.chunk_text")).alias("char_len"),
        F.current_timestamp().alias("cleaned_at"),
    )
    .filter(F.col("char_len") > 0)
)

# COMMAND ----------

if silver_exists:
    # chunked_df only contains chunks for doc_ids that were absent above, so
    # every row is guaranteed "not matched" -- this is a pure insert, never an
    # update. A plain `append` would do the same thing here, but MERGE is what
    # protects against ever double-inserting if this cell is re-run before the
    # anti-join's read is refreshed.
    (
        DeltaTable.forName(spark, silver_table).alias("t")
        .merge(chunked_df.alias("s"), "t.chunk_id = s.chunk_id")
        .whenNotMatchedInsertAll()
        .execute()
    )
else:
    (
        chunked_df.write.mode("overwrite")
        .partitionBy("source_dataset")
        .saveAsTable(silver_table)
    )

display(spark.table(silver_table).limit(10))
