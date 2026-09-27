"""Qwen text embedding provider through DashScope's OpenAI-compatible API."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

from openai import OpenAI

from src.libs.embedding.base_embedding import BaseEmbedding

if TYPE_CHECKING:
    from src.core.settings import Settings


class QwenEmbedding(BaseEmbedding):
    """Generate document and query vectors with Qwen text-embedding-v4."""

    DEFAULT_MODEL = "text-embedding-v4"
    DEFAULT_DIMENSION = 1024
    DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    QUERY_INSTRUCTION = (
        "Instruct: Given a web search query, retrieve relevant passages that answer "
        "the query\nQuery: "
    )

    def __init__(
        self,
        settings: Settings,
        *,
        batch_size: int = 10,
        timeout: float = 60.0,
        max_retries: int = 2,
        client: Any | None = None,
    ) -> None:
        config = settings.embedding
        self.model = str(config.model or self.DEFAULT_MODEL)
        self.dimension = int(config.dimensions or self.DEFAULT_DIMENSION)
        self.base_url = str(config.base_url or self.DEFAULT_BASE_URL).rstrip("/")
        self.batch_size = int(batch_size)
        if self.batch_size < 1:
            raise ValueError("Qwen embedding batch_size must be at least one")
        configured_key = str(config.api_key or "").strip()
        if not configured_key or configured_key.startswith("YOUR_"):
            configured_key = str(os.getenv("DASHSCOPE_API_KEY") or "").strip()
        if not configured_key or configured_key.startswith("YOUR_"):
            raise ValueError(
                "Qwen embedding API key is missing. Set embedding.api_key in a "
                "private settings file or DASHSCOPE_API_KEY."
            )
        self.index_identity = f"qwen:{self.model}:d{self.dimension}:document-v1"
        self._client = client or OpenAI(
            api_key=configured_key,
            base_url=self.base_url,
            timeout=timeout,
            max_retries=max_retries,
        )

    def embed(
        self,
        texts: list[str],
        trace: Any | None = None,
        **kwargs: Any,
    ) -> list[list[float]]:
        del trace
        self.validate_texts(texts)
        is_query = bool(kwargs.pop("is_query", False))
        if kwargs:
            raise TypeError(f"Unsupported Qwen embedding options: {sorted(kwargs)}")
        prepared = (
            [self.QUERY_INSTRUCTION + text for text in texts]
            if is_query
            else list(texts)
        )
        vectors: list[list[float]] = []
        for start in range(0, len(prepared), self.batch_size):
            batch = prepared[start : start + self.batch_size]
            response = self._client.embeddings.create(
                model=self.model,
                input=batch,
                dimensions=self.dimension,
            )
            ordered = sorted(response.data, key=lambda item: item.index)
            batch_vectors = [list(item.embedding) for item in ordered]
            if len(batch_vectors) != len(batch):
                raise RuntimeError("Qwen returned the wrong number of embedding vectors")
            if any(len(vector) != self.dimension for vector in batch_vectors):
                raise RuntimeError(
                    "Qwen returned an embedding vector with an unexpected dimension"
                )
            vectors.extend(batch_vectors)
        return vectors

    def get_dimension(self) -> int:
        return self.dimension
