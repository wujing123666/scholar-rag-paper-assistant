"""Confidence gates for open-set paper retrieval."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from src.paper_assistant.catalog import PaperCatalog
from src.paper_assistant.retriever import PaperSearchResult


class ScoredPaperRetriever(Protocol):
    """Retriever contract required by the confidence gate."""

    name: str
    catalog: PaperCatalog

    def search(self, query: str, top_k: int = 3) -> list[PaperSearchResult]: ...


# Development defaults calibrated on 27 generated known-paper queries and
# 9 system-authored near-domain unknown queries. These are starting points,
# not universal probabilities or final blind-test thresholds.
DEFAULT_MIN_SCORES = {
    "paper_bm25": 8.0,
    "paper_dense": 0.78,
    "paper_hybrid_rrf": 2 / 61,
}


@dataclass(frozen=True)
class PaperRetrievalDecision:
    """Accepted results plus the evidence used to accept or reject them."""

    results: tuple[PaperSearchResult, ...]
    candidates: tuple[PaperSearchResult, ...]
    rejected: bool
    reason: str
    top_score: float | None
    min_score: float
    effective_retriever: str
    fallback_reason: str | None = None


def default_min_score(retriever_name: str) -> float:
    """Return the development threshold for one retrieval strategy."""
    try:
        return DEFAULT_MIN_SCORES[retriever_name]
    except KeyError as error:
        raise ValueError(
            f"No default rejection threshold for retriever {retriever_name!r}"
        ) from error


def decide_retrieval(
    retriever: ScoredPaperRetriever,
    query: str,
    *,
    top_k: int = 3,
    min_score: float | None = None,
) -> PaperRetrievalDecision:
    """Apply a retriever-specific score threshold to ranked candidates."""
    search_with_diagnostics = getattr(retriever, "search_with_diagnostics", None)
    if callable(search_with_diagnostics):
        search_result = search_with_diagnostics(query, top_k=top_k)
        candidates = tuple(search_result.results)
        effective_retriever = str(search_result.effective_retriever)
        fallback_reason = search_result.fallback_reason
    else:
        candidates = tuple(retriever.search(query, top_k=top_k))
        effective_retriever = retriever.name
        fallback_reason = None
    threshold = (
        default_min_score(effective_retriever) if min_score is None else min_score
    )
    if threshold < 0:
        raise ValueError("min_score must be zero or greater")

    if not candidates:
        return PaperRetrievalDecision(
            results=(),
            candidates=(),
            rejected=True,
            reason="no_candidates",
            top_score=None,
            min_score=threshold,
            effective_retriever=effective_retriever,
            fallback_reason=fallback_reason,
        )

    top_score = candidates[0].score
    if top_score < threshold:
        return PaperRetrievalDecision(
            results=(),
            candidates=candidates,
            rejected=True,
            reason="below_threshold",
            top_score=top_score,
            min_score=threshold,
            effective_retriever=effective_retriever,
            fallback_reason=fallback_reason,
        )
    return PaperRetrievalDecision(
        results=candidates,
        candidates=candidates,
        rejected=False,
        reason="accepted",
        top_score=top_score,
        min_score=threshold,
        effective_retriever=effective_retriever,
        fallback_reason=fallback_reason,
    )
