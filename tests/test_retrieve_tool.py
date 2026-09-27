from unittest.mock import MagicMock

from mcp_server.tools import retrieve


def test_search_documents_maps_results(monkeypatch):
    fake_index = MagicMock()
    fake_index.similarity_search.return_value = {
        "manifest": {
            "columns": [
                {"name": "chunk_id"},
                {"name": "doc_id"},
                {"name": "chunk_text"},
                {"name": "source_dataset"},
                {"name": "score"},
            ]
        },
        "result": {"data_array": [["c1", "d1", "hello world", "arxiv_sample", 0.87]]},
    }
    fake_vs_client = MagicMock()
    fake_vs_client.get_index.return_value = fake_index

    monkeypatch.setattr(retrieve, "get_vector_search_client", lambda: fake_vs_client)
    monkeypatch.setattr(retrieve, "_embed_query", lambda query, cfg: [0.1, 0.2, 0.3])

    hits = retrieve.search_documents("what is this about?", top_k=1)

    assert hits == [
        {
            "chunk_text": "hello world",
            "source": "arxiv_sample",
            "doc_id": "d1",
            "chunk_id": "c1",
            "score": 0.87,
        }
    ]
    fake_index.similarity_search.assert_called_once()
    fake_vs_client.get_index.assert_called_once()


def test_search_documents_applies_source_filter(monkeypatch):
    fake_index = MagicMock()
    fake_index.similarity_search.return_value = {
        "manifest": {"columns": [{"name": "chunk_id"}]},
        "result": {"data_array": []},
    }
    fake_vs_client = MagicMock()
    fake_vs_client.get_index.return_value = fake_index

    monkeypatch.setattr(retrieve, "get_vector_search_client", lambda: fake_vs_client)
    monkeypatch.setattr(retrieve, "_embed_query", lambda query, cfg: [0.1])

    retrieve.search_documents("query", source_filter="aws_docs")

    _, kwargs = fake_index.similarity_search.call_args
    assert kwargs["filters"] == {"source_dataset": "aws_docs"}
