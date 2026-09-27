# Databricks notebook source
# MAGIC %md
# MAGIC # 02 - Silver: clean + chunk
# MAGIC Cleans Bronze text with plain built-in string/regex functions, then chunks
# MAGIC it with whichever strategy `chunking.strategy` in config.yaml selects
# MAGIC ("fixed" or "recursive" -- see rag_common/chunking.py), and writes
# MAGIC `silver_chunks`. Chunking runs as a `pandas_udf`: it does real per-row
# MAGIC work (splitting, packing), unlike the cleaning step, so batching pays off.

# COMMAND ----------

import os
import sys

try:
    import rag_common  # noqa: F401
except ImportError:
    sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..", "..")))

from rag_common.chunking import chunk_fixed_size, chunk_recursive
from rag_common.cleaning import clean_text
from rag_common.config import load_config
from rag_common.hashing import chunk_id_for

cfg = load_config()
bronze_table = f"{cfg.catalog}.{cfg.schema}.bronze_raw_docs"
silver_table = f"{cfg.catalog}.{cfg.schema}.silver_chunks"

# COMMAND ----------

import pandas as pd
from pyspark.sql import functions as F
from pyspark.sql.types import ArrayType, StringType, StructField, StructType

bronze_df = spark.table(bronze_table)

clean_text_udf = F.udf(clean_text, StringType())
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

(
    chunked_df.write.mode("overwrite")
    # See 01_bronze_ingest.py: dynamic partition overwrite so this write only
    # ever touches the source_dataset partitions it actually produced.
    .option("partitionOverwriteMode", "dynamic")
    .option("overwriteSchema", "true")
    .partitionBy("source_dataset")
    .saveAsTable(silver_table)
)

display(spark.table(silver_table).limit(10))
