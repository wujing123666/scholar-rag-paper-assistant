"""Optional local Cross-Encoder reranking for paper evidence chunks."""

from __future__ import annotations

import math
import re
from dataclasses import replace
from pathlib import Path
from typing import Protocol

from src.paper_assistant.chunk_retriever import (
    ChunkSearchResult,
    expand_chunk_query,
    is_formula_detail,
    is_method_detail,
)


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
                + self.weight * normalized_score
                + _method_detail_bonus(result),
                rerank_score=raw_score,
            )
            for result, raw_score, normalized_score in zip(
                results, raw_scores, normalized
            )
        ]
        ranked = sorted(reranked, key=lambda item: (-item.score, item.chunk_id))
        return select_diverse_chunks(ranked, top_k=top_k)


_METHOD_SECTION = re.compile(
    r"\b(method(?:ology)?|approach|algorithm|framework|model|方法|算法|模型)\b",
    re.IGNORECASE,
)


def _method_detail_bonus(result: ChunkSearchResult) -> float:
    if is_formula_detail(result):
        return 0.06
    if is_method_detail(result):
        return 0.04
    return 0.02 if _METHOD_SECTION.search(result.section) else 0.0


def select_diverse_chunks(
    results: list[ChunkSearchResult], *, top_k: int, max_per_page: int = 1
) -> list[ChunkSearchResult]:
    """Prefer distinct paper pages, then fill any remaining result slots."""
    if top_k < 1:
        raise ValueError("top_k must be at least one")
    if max_per_page < 1:
        raise ValueError("max_per_page must be at least one")
    selected: list[ChunkSearchResult] = []
    deferred: list[ChunkSearchResult] = []
    page_counts: dict[tuple[str, int], int] = {}
    focused = sorted(
        (
            result
            for result in results
            if result.focus_rank > 0 and result.routing_rank in (0, 1)
        ),
        key=lambda result: result.focus_rank,
    )[:top_k]
    focused_ids = {result.chunk_id for result in focused}
    for result in focused:
        selected.append(result)
        page = (result.paper.paper_id, result.page_number)
        page_counts[page] = page_counts.get(page, 0) + 1
    for result in results:
        if len(selected) == top_k:
            break
        if result.chunk_id in focused_ids:
            continue
        page = (result.paper.paper_id, result.page_number)
        if page_counts.get(page, 0) >= max_per_page:
            deferred.append(result)
            continue
        selected.append(result)
        page_counts[page] = page_counts.get(page, 0) + 1
        if len(selected) == top_k:
            break
    selected.extend(deferred[: top_k - len(selected)])
    original_order = {result.chunk_id: index for index, result in enumerate(results)}
    return sorted(selected, key=lambda result: original_order[result.chunk_id])


def select_rerank_candidates(
    results: list[ChunkSearchResult], *, top_k: int
) -> list[ChunkSearchResult]:
    """Keep top candidates while retaining focused evidence from the first paper."""
    if top_k < 1:
        raise ValueError("top_k must be at least one")
    selected = list(results[:top_k])
    selected_ids = {result.chunk_id for result in selected}
    reserved_ids: set[str] = set()
    focused = sorted(
        (
            result
            for result in results
            if result.focus_rank > 0 and result.routing_rank in (0, 1)
        ),
        key=lambda result: result.focus_rank,
    )[:top_k]
    for result in focused:
        reserved_ids.add(result.chunk_id)
        if result.chunk_id in selected_ids:
            continue
        replace_index = next(
            (
                index
                for index in range(len(selected) - 1, -1, -1)
                if selected[index].chunk_id not in reserved_ids
            ),
            None,
        )
        if replace_index is None:
            break
        selected_ids.remove(selected[replace_index].chunk_id)
        selected[replace_index] = result
        selected_ids.add(result.chunk_id)
    original_order = {result.chunk_id: index for index, result in enumerate(results)}
    return sorted(selected, key=lambda result: original_order[result.chunk_id])


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
        return select_diverse_chunks(candidates, top_k=top_k), reason
