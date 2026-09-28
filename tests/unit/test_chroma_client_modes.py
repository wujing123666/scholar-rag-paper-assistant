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


def test_chroma_get_by_ids_batches_large_requests_and_preserves_order():
    collection = Mock()

    def get_batch(*, ids, include):
        assert include == ["metadatas", "documents"]
        returned = [record_id for record_id in reversed(ids) if record_id != "id-0600"]
        return {
            "ids": returned,
            "documents": [f"text:{record_id}" for record_id in returned],
            "metadatas": [{"source": record_id} for record_id in returned],
        }

    collection.get.side_effect = get_batch
    store = object.__new__(ChromaStore)
    store.client = object()
    store.collection = collection
    ids = [f"id-{index:04d}" for index in range(1203)]

    records = store.get_by_ids(ids)

    assert collection.get.call_count == 3
    assert max(len(call.kwargs["ids"]) for call in collection.get.call_args_list) == 500
    assert records[0]["id"] == "id-0000"
    assert records[600] == {}
    assert records[-1]["id"] == "id-1202"


def test_chroma_upsert_respects_client_batch_limit():
    collection = Mock()
    client = Mock()
    client.get_max_batch_size.return_value = 2
    store = object.__new__(ChromaStore)
    store.client = client
    store.collection = collection

    store.upsert(
        [
            {
                "id": f"id-{index}",
                "vector": [float(index), 0.0],
                "document": f"text-{index}",
                "metadata": {"index": index},
            }
            for index in range(5)
        ]
    )

    assert collection.upsert.call_count == 3
    assert [
        len(call.kwargs["ids"]) for call in collection.upsert.call_args_list
    ] == [2, 2, 1]


def test_chroma_rebuild_keeps_live_collection_until_commit(tmp_path):
    with ChromaStore(
        persist_directory=tmp_path / "chroma",
        collection_name="dimension_swap",
    ) as store:
        store.upsert(
            [
                {
                    "id": "old",
                    "vector": [1.0, 0.0],
                    "document": "old document",
                    "metadata": {"version": "old"},
                }
            ]
        )
        store.begin_rebuild()
        store.upsert_rebuild(
            [
                {
                    "id": "new",
                    "vector": [1.0, 0.0, 0.0],
                    "document": "new document",
                    "metadata": {"version": "new"},
                }
            ]
        )

        assert store.list_ids() == ["old"]
        store.commit_rebuild()

        assert store.list_ids() == ["new"]
        assert store.get_by_ids(["new"])[0]["metadata"]["version"] == "new"
