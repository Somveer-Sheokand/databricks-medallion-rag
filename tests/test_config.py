from rag_common.config import load_config


def test_load_config_reads_yaml():
    cfg = load_config()
    assert cfg.catalog
    assert cfg.schema
    assert cfg.chunking.strategy in {"fixed", "recursive"}
    assert cfg.embedding.provider in {"fmapi", "local"}
    assert cfg.vector_search.endpoint_name
    assert cfg.retrieval.default_top_k > 0


def test_load_config_env_override(monkeypatch):
    monkeypatch.setenv("VARNAM_CATALOG", "some_other_catalog")
    cfg = load_config()
    assert cfg.catalog == "some_other_catalog"
