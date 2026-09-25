"""Paper-level dense retrieval for fuzzy semantic descriptions."""

from __future__ import annotations

import numpy as np

from src.libs.embedding.base_embedding import BaseEmbedding
from src.paper_assistant.catalog import PaperCatalog, PaperProfile
from src.paper_assistant.retriever import PaperSearchResult


def profile_text(profile: PaperProfile) -> str:
    """Build one readable embedding document for one logical paper."""
    fields = (
        ("英文标题", profile.canonical_title),
        ("中文标题", profile.title_zh),
        ("主题标签", "；".join(profile.tags)),
        ("方法", "；".join(profile.method_summaries)),
        ("数据集", "；".join(profile.datasets)),
        ("记忆线索", "；".join(profile.memory_cues)),
        ("作者", "；".join(profile.authors)),
        ("年份", profile.year),
        ("期刊或会议", profile.venue),
    )
    return "\n".join(f"{label}：{value}" for label, value in fields if value)


def _normalized(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("Embedding provider returned a zero-length vector")
    return matrix / norms


class PaperDenseRetriever:
    """Rank complete paper profiles using cosine similarity."""

    name = "paper_dense"

    def __init__(self, catalog: PaperCatalog, embedding: BaseEmbedding) -> None:
        self.catalog = catalog
        self.embedding = embedding
        self._profiles = catalog.profiles
        vectors = embedding.embed([profile_text(profile) for profile in self._profiles])
        matrix = np.asarray(vectors, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[0] != len(self._profiles):
            raise ValueError("Embedding provider returned an invalid document matrix")
        self._document_vectors = _normalized(matrix)

    def search(self, query: str, top_k: int = 3) -> list[PaperSearchResult]:
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")
        if top_k < 1:
            raise ValueError("top_k must be at least one")

        vector = np.asarray(
            self.embedding.embed([query], is_query=True), dtype=np.float32
        )
        if vector.ndim != 2 or vector.shape[0] != 1:
            raise ValueError("Embedding provider returned an invalid query vector")
        query_vector = _normalized(vector)[0]
        scores = self._document_vectors @ query_vector
        ranked_indices = sorted(
            range(len(self._profiles)),
            key=lambda index: (-float(scores[index]), self._profiles[index].paper_id),
        )
        return [
            PaperSearchResult(
                paper=self._profiles[index],
                score=float(scores[index]),
                matched_terms=(),
            )
            for index in ranked_indices[: min(top_k, len(ranked_indices))]
        ]
