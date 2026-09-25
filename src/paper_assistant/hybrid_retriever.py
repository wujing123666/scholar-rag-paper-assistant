"""Rank fusion for sparse and dense paper retrieval."""

from __future__ import annotations

from src.paper_assistant.catalog import PaperCatalog
from src.paper_assistant.dense_retriever import PaperDenseRetriever
from src.paper_assistant.retriever import PaperBM25Retriever, PaperSearchResult


class PaperHybridRetriever:
    """Fuse BM25 and dense ranks using reciprocal rank fusion (RRF)."""

    name = "paper_hybrid_rrf"

    def __init__(
        self,
        catalog: PaperCatalog,
        sparse: PaperBM25Retriever,
        dense: PaperDenseRetriever,
        *,
        rrf_k: int = 60,
    ) -> None:
        if sparse.catalog is not catalog or dense.catalog is not catalog:
            raise ValueError("Hybrid retrievers must share the same paper catalog")
        if rrf_k < 1:
            raise ValueError("rrf_k must be at least one")
        self.catalog = catalog
        self.sparse = sparse
        self.dense = dense
        self.rrf_k = rrf_k

    def search(self, query: str, top_k: int = 3) -> list[PaperSearchResult]:
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")
        if top_k < 1:
            raise ValueError("top_k must be at least one")

        source_results = (
            self.sparse.search(query, top_k=len(self.catalog)),
            self.dense.search(query, top_k=len(self.catalog)),
        )
        scores: dict[str, float] = {}
        matches: dict[str, set[str]] = {}
        for results in source_results:
            for rank, result in enumerate(results, start=1):
                paper_id = result.paper.paper_id
                scores[paper_id] = scores.get(paper_id, 0.0) + 1 / (self.rrf_k + rank)
                matches.setdefault(paper_id, set()).update(result.matched_terms)

        ranked_ids = sorted(scores, key=lambda paper_id: (-scores[paper_id], paper_id))
        return [
            PaperSearchResult(
                paper=self.catalog.get(paper_id),
                score=scores[paper_id],
                matched_terms=tuple(sorted(matches.get(paper_id, ()))),
            )
            for paper_id in ranked_ids[: min(top_k, len(ranked_ids))]
        ]
