"""Tests for switching Chroma between embedded and HTTP clients."""

from unittest.mock import Mock

from src.libs.vector_store.chroma_store import ChromaStore


def test_chroma_store_uses_http_client_in_server_mode(monkeypatch):
    collection = Mock()
    collection.count.return_value = 0
    client = Mock()
    client.get_or_create_collection.return_value = collection
    http_client = Mock(return_value=client)
    monkeypatch.setattr(
        "src.libs.vector_store.chroma_store.chromadb.HttpClient", http_client
    )

    store = ChromaStore(
        mode="server",
        host="chroma.internal",
        port=9000,
        ssl=True,
        collection_name="paper_profiles_v1",
    )

    call = http_client.call_args
    assert call.kwargs["host"] == "chroma.internal"
    assert call.kwargs["port"] == 9000
    assert call.kwargs["ssl"] is True
    client.get_or_create_collection.assert_called_once_with(
        name="paper_profiles_v1", metadata={"hnsw:space": "cosine"}
    )
    store.close()


def test_chroma_store_rejects_unknown_mode():
    try:
        ChromaStore(mode="shared-folder")
    except ValueError as error:
        assert "local" in str(error)
        assert "server" in str(error)
    else:
        raise AssertionError("Expected an unsupported Chroma mode to be rejected")
