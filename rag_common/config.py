"""Loads config/config.yaml into typed config objects.

Imported both by the Databricks notebooks (databricks/notebooks/) and by the
local MCP server (mcp_server/) so the two sides can never drift out of sync on
table names, chunking parameters, or which embedding model is in use.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml

_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "config.yaml"


@dataclass
class ChunkingConfig:
    strategy: str
    chunk_size: int
    overlap: int


@dataclass
class EmbeddingConfig:
    provider: str
    endpoint_name: str
    local_model_name: str
    dimension: int


@dataclass
class VectorSearchConfig:
    endpoint_name: str
    index_name: str


@dataclass
class RetrievalConfig:
    default_top_k: int


@dataclass
class Config:
    catalog: str
    schema: str
    default_source_dataset: str
    chunking: ChunkingConfig
    embedding: EmbeddingConfig
    vector_search: VectorSearchConfig
    retrieval: RetrievalConfig


def load_config(path: Optional[Path] = None) -> Config:
    config_path = path or _CONFIG_PATH
    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    # Env var overrides let the same file work unmodified across Databricks
    # notebook widgets and local runs (e.g. a quick catalog switch without editing yaml).
    catalog = os.environ.get("VARNAM_CATALOG", raw["catalog"])
    schema = os.environ.get("VARNAM_SCHEMA", raw["schema"])

    return Config(
        catalog=catalog,
        schema=schema,
        default_source_dataset=raw["default_source_dataset"],
        chunking=ChunkingConfig(**raw["chunking"]),
        embedding=EmbeddingConfig(**raw["embedding"]),
        vector_search=VectorSearchConfig(**raw["vector_search"]),
        retrieval=RetrievalConfig(**raw["retrieval"]),
    )
