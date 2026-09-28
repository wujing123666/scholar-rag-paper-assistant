"""Paper-level Cross-Encoder reranking and open-set relevance scoring."""

from __future__ import annotations

import math
import threading
from dataclasses import replace
from pathlib import Path
from typing import Protocol

from src.paper_assistant.dense_retriever import profile_text
from src.paper_assistant.retriever import PaperSearchResult


class CrossEncoderModel(Protocol):
    def rerank(
        self, query: str, documents: list[str], batch_size: int = 64
    ) -> object: ...


class LazySynchronizedCrossEncoder:
    """Load one FastEmbed model lazily and serialize shared inference calls."""

    def __init__(
        self,
        model_name: str,
        *,
        cache_dir: str | Path | None = None,
    ) -> None:
        self.model_name = model_name
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self._model: CrossEncoderModel | None = None
        self._lock = threading.Lock()

    def rerank(
        self, query: str, documents: list[str], batch_size: int = 64
    ) -> object:
        with self._lock:
            if self._model is None:
                from fastembed.rerank.cross_encoder import TextCrossEncoder

                self._model = TextCrossEncoder(
                    model_name=self.model_name,
                    cache_dir=str(self.cache_dir) if self.cache_dir else None,
                )
            return self._model.rerank(
                query,
                documents,
                batch_size=batch_size,
            )


class FastEmbedPaperReranker:
    """Score first-stage paper candidates with an absolute relevance logit.

    Chunk reranking normalizes scores within one result list because it only
    needs an ordering.  An open-set gate must retain the raw model logit: a
    normalized winner would always look strong even when every paper is
    irrelevant.  Model loading and inference are serialized because one
    service instance is shared by concurrent Streamlit requests.
    """

    def __init__(
        self,
        model_name: str,
        *,
        cache_dir: str | Path | None = None,
        batch_size: int = 8,
        model: CrossEncoderModel | None = None,
    ) -> None:
        if batch_size < 1:
            raise ValueError("paper reranker batch_size must be at least one")
        self.model_name = model_name
        self.batch_size = batch_size
        self.model = model or LazySynchronizedCrossEncoder(
            model_name,
            cache_dir=cache_dir,
        )

    def rerank(
        self,
        query: str,
        candidates: list[PaperSearchResult],
        *,
        top_k: int,
    ) -> list[PaperSearchResult]:
        query = query.strip()
        if not query:
            raise ValueError("query cannot be empty")
        if top_k < 1:
            raise ValueError("top_k must be at least one")
        if not candidates:
            return []
        documents = [profile_text(item.paper) for item in candidates]
        raw_scores = [
            float(score)
            for score in self.model.rerank(
                query,
                documents,
                batch_size=self.batch_size,
            )
        ]
        if len(raw_scores) != len(candidates):
            raise ValueError("Paper Cross-Encoder returned an invalid score count")
        if any(not math.isfinite(score) for score in raw_scores):
            raise ValueError("Paper Cross-Encoder returned a non-finite score")
        scored = [
            replace(candidate, score=score)
            for candidate, score in zip(candidates, raw_scores, strict=True)
        ]
        return sorted(
            scored,
            key=lambda item: (-item.score, item.paper.paper_id),
        )[:top_k]
