"""Local ONNX embedding provider backed by FastEmbed."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.libs.embedding.base_embedding import BaseEmbedding

if TYPE_CHECKING:
    from collections.abc import Iterable


class FastEmbedEmbedding(BaseEmbedding):
    """Generate embeddings locally without an API key.

    The model is loaded lazily so importing the application does not download
    model files or allocate an ONNX session. ``query_embed`` is used for search
    queries so model-specific query handling remains inside FastEmbed.
    """

    DEFAULT_MODEL = "BAAI/bge-small-zh-v1.5"
    MODEL_DIMENSIONS = {DEFAULT_MODEL: 512}

    def __init__(
        self,
        settings: Any | None = None,
        *,
        model: str | None = None,
        cache_dir: str | Path | None = None,
        threads: int | None = None,
        **kwargs: Any,
    ) -> None:
        configured_model = None
        if settings is not None:
            configured_model = getattr(getattr(settings, "embedding", None), "model", None)
        self.model = model or configured_model or self.DEFAULT_MODEL
        self.cache_dir = str(cache_dir) if cache_dir is not None else None
        self.threads = threads
        self._model: Any | None = None
        self._model_kwargs = kwargs

    def _get_model(self) -> Any:
        if self._model is None:
            try:
                from fastembed import TextEmbedding
            except ImportError as error:
                raise RuntimeError(
                    "FastEmbed is not installed. Install the local embedding extra with "
                    "'pip install -e .[local]'."
                ) from error
            self._model = TextEmbedding(
                model_name=self.model,
                cache_dir=self.cache_dir,
                threads=self.threads,
                **self._model_kwargs,
            )
        return self._model

    def embed(
        self,
        texts: list[str],
        trace: Any | None = None,
        **kwargs: Any,
    ) -> list[list[float]]:
        """Embed passages, or queries when ``is_query=True`` is supplied."""
        del trace
        self.validate_texts(texts)
        is_query = bool(kwargs.pop("is_query", False))
        model = self._get_model()
        vectors: Iterable[Any]
        if is_query:
            vectors = model.query_embed(texts, **kwargs)
        else:
            vectors = model.embed(texts, **kwargs)
        return [vector.tolist() for vector in vectors]

    def get_dimension(self) -> int:
        try:
            return self.MODEL_DIMENSIONS[self.model]
        except KeyError as error:
            raise NotImplementedError(
                f"Embedding dimension is not registered for model {self.model!r}"
            ) from error
