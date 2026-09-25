"""Optional local Cross-Encoder reranking for paper evidence chunks."""

from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path
from typing import Protocol

from src.paper_assistant.chunk_retriever import ChunkSearchResult, expand_chunk_query


class CrossEncoderModel(Protocol):
    def rerank(
        self, query: str, documents: list[str], batch_size: int = 64
    ) -> object: ...


class FastEmbedChunkReranker:
    """Blend local ONNX Cross-Encoder scores with first-stage chunk scores."""

    def __init__(
        self,
        model_name: str,
        *,
        cache_dir: str | Path | None = None,
        weight: float = 0.35,
        batch_size: int = 8,
        model: CrossEncoderModel | None = None,
    ) -> None:
        if not 0 <= weight <= 1:
            raise ValueError("reranker weight must be between zero and one")
        if batch_size < 1:
            raise ValueError("reranker batch_size must be at least one")
        self.model_name = model_name
        self.weight = weight
        self.batch_size = batch_size
        if model is None:
            from fastembed.rerank.cross_encoder import TextCrossEncoder

            model = TextCrossEncoder(
                model_name=model_name,
                cache_dir=str(cache_dir) if cache_dir else None,
            )
        self.model = model

    @staticmethod
    def _document(result: ChunkSearchResult) -> str:
        fields = []
        if result.section:
            fields.append(f"Section: {result.section}")
        fields.append(result.text)
        return "\n".join(fields)

    @staticmethod
    def _normalize(scores: list[float]) -> list[float]:
        if not scores:
            return []
        if any(not math.isfinite(score) for score in scores):
            raise ValueError("Cross-Encoder returned a non-finite score")
        low, high = min(scores), max(scores)
        if math.isclose(low, high):
            return [0.5] * len(scores)
        return [(score - low) / (high - low) for score in scores]

    def rerank(
        self,
        query: str,
        results: list[ChunkSearchResult],
        *,
        top_k: int,
    ) -> list[ChunkSearchResult]:
        if top_k < 1:
            raise ValueError("top_k must be at least one")
        if not results:
            return []
        raw_scores = [
            float(score)
            for score in self.model.rerank(
                expand_chunk_query(query),
                [self._document(result) for result in results],
                batch_size=self.batch_size,
            )
        ]
        if len(raw_scores) != len(results):
            raise ValueError("Cross-Encoder returned an invalid score count")
        normalized = self._normalize(raw_scores)
        reranked = [
            replace(
                result,
                score=(1 - self.weight) * result.score
                + self.weight * normalized_score,
                rerank_score=raw_score,
            )
            for result, raw_score, normalized_score in zip(
                results, raw_scores, normalized
            )
        ]
        return sorted(reranked, key=lambda item: (-item.score, item.chunk_id))[:top_k]


def rerank_with_fallback(
    reranker: FastEmbedChunkReranker,
    query: str,
    candidates: list[ChunkSearchResult],
    *,
    top_k: int,
) -> tuple[list[ChunkSearchResult], str | None]:
    """Rerank one comparable candidate pool and preserve retrieval on failure."""
    try:
        return reranker.rerank(query, candidates, top_k=top_k), None
    except Exception as error:
        reason = f"{type(error).__name__}: {error}"
        return candidates[:top_k], reason
