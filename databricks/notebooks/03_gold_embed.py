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

from pyspark.sql import functions as F

silver_df = spark.table(silver_table)

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

(
    embedded_df.write.mode("overwrite")
    # See 01_bronze_ingest.py: dynamic partition overwrite so this write only
    # ever touches the source_dataset partitions it actually produced.
    .option("partitionOverwriteMode", "dynamic")
    .option("overwriteSchema", "true")
    .partitionBy("source_dataset")
    .saveAsTable(gold_table)
)

spark.sql(f"ALTER TABLE {gold_table} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)")

display(spark.table(gold_table).limit(10))
