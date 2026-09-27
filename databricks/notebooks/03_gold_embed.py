# Databricks notebook source
# MAGIC %md
# MAGIC # 03 - Gold: embeddings
# MAGIC Embeds Silver chunks using either the Databricks Foundation Model API
# MAGIC (built-in `ai_query()`, no UDF) or a local CPU model (`pandas_udf`
# MAGIC fallback) -- set `embedding.provider` in config/config.yaml. This is the
# MAGIC "UDF vs built-in" comparison: same output column, two very different
# MAGIC execution paths worth benchmarking against each other.
# MAGIC
# MAGIC Also enables Change Data Feed on the output table, which the Delta Sync
# MAGIC Index in `04_create_vector_index.py` requires.
# MAGIC
# MAGIC **Incremental**: only embeds chunks whose `chunk_id` isn't already in
# MAGIC `gold_embedded_chunks` (an anti-join against Silver), then `MERGE ...
# MAGIC WHEN NOT MATCHED THEN INSERT`s the result -- avoids paying for
# MAGIC re-embedding chunks that haven't changed on every rerun.

# COMMAND ----------

import os
import sys

try:
    import rag_common  # noqa: F401
except ImportError:
    sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..", "..")))

from rag_common.config import load_config

cfg = load_config()

silver_table = f"{cfg.catalog}.{cfg.schema}.silver_chunks"
gold_table = f"{cfg.catalog}.{cfg.schema}.gold_embedded_chunks"

# COMMAND ----------

from delta.tables import DeltaTable
from pyspark.sql import functions as F

silver_df = spark.table(silver_table)

gold_exists = spark.catalog.tableExists(gold_table)
if gold_exists:
    already_embedded = spark.table(gold_table).select("chunk_id").distinct()
    silver_df = silver_df.join(already_embedded, on="chunk_id", how="left_anti")

if cfg.embedding.provider == "fmapi":
    # Built-in: ai_query() calls the Databricks-hosted embedding endpoint
    # directly from the DataFrame expression -- no UDF, no driver round-trips.
    # Assumes the endpoint's serving task type is llm/v1/embeddings, in which
    # case ai_query infers an array-of-double return automatically; pass
    # `returnType => 'ARRAY<DOUBLE>'` explicitly if your endpoint needs it.
    embedded_df = silver_df.withColumn(
        "embedding",
        F.expr(f"ai_query('{cfg.embedding.endpoint_name}', chunk_text)").cast("array<float>"),
    )
else:
    # Fallback: local CPU model, batched in a pandas_udf.
    import pandas as pd
    from pyspark.sql.types import ArrayType, FloatType

    from rag_common.embeddings import embed_texts_local

    LOCAL_MODEL_NAME = cfg.embedding.local_model_name

    @F.pandas_udf(ArrayType(FloatType()))
    def embed_udf(texts: pd.Series) -> pd.Series:
        vectors = embed_texts_local(texts.tolist(), model_name=LOCAL_MODEL_NAME)
        return pd.Series(vectors)

    embedded_df = silver_df.withColumn("embedding", embed_udf(F.col("chunk_text")))

# COMMAND ----------

if gold_exists:
    # embedded_df only contains chunks that were absent above, so every row
    # is guaranteed "not matched" -- see 02_silver_chunk.py for why MERGE
    # rather than a plain append.
    (
        DeltaTable.forName(spark, gold_table).alias("t")
        .merge(embedded_df.alias("s"), "t.chunk_id = s.chunk_id")
        .whenNotMatchedInsertAll()
        .execute()
    )
else:
    (
        embedded_df.write.mode("overwrite")
        .partitionBy("source_dataset")
        .saveAsTable(gold_table)
    )

spark.sql(
    f"ALTER TABLE {gold_table} SET TBLPROPERTIES ("
    "delta.enableChangeDataFeed = true, "
    # The Delta Sync Index's sync fails once it's been longer than this
    # since the last sync -- the default 7 days is too short for a
    # TRIGGERED (not continuous) index you don't re-sync daily.
    "delta.deletedFileRetentionDuration = 'interval 30 days')"
)

display(spark.table(gold_table).limit(10))
