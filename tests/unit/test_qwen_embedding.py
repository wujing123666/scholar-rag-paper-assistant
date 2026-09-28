"""Tests for the production Qwen embedding provider."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.libs.embedding.qwen_embedding import QwenEmbedding


class _EmbeddingsEndpoint:
    def __init__(self, dimension: int = 3) -> None:
        self.dimension = dimension
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        data = [
            SimpleNamespace(index=index, embedding=[float(index)] * self.dimension)
            for index, _text in enumerate(kwargs["input"])
        ]
        return SimpleNamespace(data=data)


def _settings(*, api_key: str | None = "test-key", dimensions: int = 3):
    return SimpleNamespace(
        embedding=SimpleNamespace(
            provider="qwen",
            model="text-embedding-v4",
            dimensions=dimensions,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            api_key=api_key,
        )
    )


def test_qwen_batches_documents_and_preserves_order():
    endpoint = _EmbeddingsEndpoint()
    client = SimpleNamespace(embeddings=endpoint)
    embedding = QwenEmbedding(_settings(), batch_size=2, client=client)

    vectors = embedding.embed(["a", "b", "c"])

    assert vectors == [[0.0] * 3, [1.0] * 3, [0.0] * 3]
    assert [call["input"] for call in endpoint.calls] == [["a", "b"], ["c"]]
    assert all(call["dimensions"] == 3 for call in endpoint.calls)
    assert embedding.index_identity == "qwen:text-embedding-v4:d3:document-v1"


def test_qwen_parallel_batches_preserve_input_batch_order():
    endpoint = _EmbeddingsEndpoint()
    embedding = QwenEmbedding(
        _settings(),
        batch_size=2,
        max_workers=3,
        client=SimpleNamespace(embeddings=endpoint),
    )

    vectors = embedding.embed(["a", "b", "c", "d", "e"])

    assert vectors == [
        [0.0] * 3,
        [1.0] * 3,
        [0.0] * 3,
        [1.0] * 3,
        [0.0] * 3,
    ]
    assert {tuple(call["input"]) for call in endpoint.calls} == {
        ("a", "b"),
        ("c", "d"),
        ("e",),
    }


def test_qwen_retries_transient_batching_backend_internal_error():
    endpoint = _EmbeddingsEndpoint()
    original_create = endpoint.create
    attempts = 0

    class TransientBackendError(RuntimeError):
        status_code = 400

    def flaky_create(**kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise TransientBackendError(
                "InternalError: Receive batching backend response failed!"
            )
        return original_create(**kwargs)

    endpoint.create = flaky_create
    embedding = QwenEmbedding(
        _settings(),
        backend_max_attempts=2,
        backend_retry_base_seconds=0,
        client=SimpleNamespace(embeddings=endpoint),
    )

    assert embedding.embed(["paper chunk"]) == [[0.0] * 3]
    assert attempts == 2


def test_qwen_adds_retrieval_instruction_only_to_queries():
    endpoint = _EmbeddingsEndpoint()
    embedding = QwenEmbedding(
        _settings(), client=SimpleNamespace(embeddings=endpoint)
    )

    embedding.embed(["开放问题"], is_query=True)

    sent = endpoint.calls[0]["input"][0]
    assert sent.startswith("Instruct: Given a web search query")
    assert sent.endswith("Query: 开放问题")


def test_qwen_rejects_missing_private_api_key(monkeypatch):
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)

    with pytest.raises(ValueError, match="API key is missing"):
        QwenEmbedding(_settings(api_key="YOUR_API_KEY_HERE"))


@pytest.mark.parametrize("configured_key", [None, "", "YOUR_API_KEY_HERE"])
def test_qwen_uses_environment_key_when_config_is_missing_or_placeholder(
    configured_key, monkeypatch
):
    monkeypatch.setenv("DASHSCOPE_API_KEY", "environment-key")
    captured = {}

    class _Client:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr("src.libs.embedding.qwen_embedding.OpenAI", _Client)

    QwenEmbedding(_settings(api_key=configured_key))

    assert captured["api_key"] == "environment-key"


def test_qwen_rejects_unexpected_vector_dimension():
    endpoint = _EmbeddingsEndpoint(dimension=2)
    embedding = QwenEmbedding(
        _settings(dimensions=3), client=SimpleNamespace(embeddings=endpoint)
    )

    with pytest.raises(RuntimeError, match="unexpected dimension"):
        embedding.embed(["论文"])
