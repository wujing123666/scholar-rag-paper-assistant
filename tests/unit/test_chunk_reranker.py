"""Tests for optional local paper chunk reranking."""

from __future__ import annotations

from dataclasses import replace

import pytest

from src.paper_assistant.catalog import PaperProfile
from src.paper_assistant.chunk_reranker import (
    FastEmbedChunkReranker,
    rerank_with_fallback,
    select_diverse_chunks,
    select_rerank_candidates,
)
from src.paper_assistant.chunk_retriever import ChunkSearchResult


def _profile() -> PaperProfile:
    return PaperProfile(
        paper_id="paper",
        pdf_files=("paper.pdf",),
        canonical_title="Paper",
        title_zh="",
        authors=(),
        year="2026",
        venue="Test",
        languages=("en",),
        tags=(),
        method_summaries=(),
        datasets=(),
        memory_cues=(),
        duplicate_groups=(),
    )


def _result(
    chunk_id: str,
    text: str,
    score: float,
    *,
    page_number: int = 1,
    section: str = "METHOD",
) -> ChunkSearchResult:
    return ChunkSearchResult(
        chunk_id=chunk_id,
        paper=_profile(),
        pdf_file="paper.pdf",
        page_number=page_number,
        chunk_index=1,
        section=section,
        text=text,
        score=score,
        dense_score=score,
    )


class FakeCrossEncoder:
    def __init__(self, scores=None, error=None):
        self.scores = scores or []
        self.error = error
        self.calls = []

    def rerank(self, query, documents, batch_size=64):
        self.calls.append((query, documents, batch_size))
        if self.error:
            raise self.error
        return iter(self.scores)


def test_cross_encoder_blend_moves_relevant_chunk_to_top():
    model = FakeCrossEncoder([0.1, 3.0])
    reranker = FastEmbedChunkReranker("fake", weight=0.8, model=model)
    candidates = [
        _result("generic", "generic overview", 0.9),
        _result("method", "adaptive coefficient method", 0.7),
    ]

    results = reranker.rerank("自适应系数", candidates, top_k=2)

    assert results[0].chunk_id == "method"
    assert results[0].rerank_score == 3.0
    assert "adaptive coefficient" in model.calls[0][0]
    assert model.calls[0][1][1].startswith("Section: METHOD")


def test_cross_encoder_validates_score_count_and_values():
    candidates = [_result("one", "text", 0.8)]

    with pytest.raises(ValueError, match="score count"):
        FastEmbedChunkReranker("fake", model=FakeCrossEncoder([])).rerank(
            "query", candidates, top_k=1
        )
    with pytest.raises(ValueError, match="non-finite"):
        FastEmbedChunkReranker("fake", model=FakeCrossEncoder([float("nan")])).rerank(
            "query", candidates, top_k=1
        )


def test_cross_encoder_equal_scores_keep_base_order():
    reranker = FastEmbedChunkReranker(
        "fake", weight=0.8, model=FakeCrossEncoder([1.0, 1.0])
    )
    candidates = [
        _result("first", "first", 0.9),
        _result("second", "second", 0.7),
    ]

    results = reranker.rerank("query", candidates, top_k=2)

    assert [result.chunk_id for result in results] == ["first", "second"]


def test_safe_rerank_falls_back_to_original_ranking_on_runtime_error():
    original = [_result("first", "first", 0.9), _result("second", "second", 0.8)]
    reranker = FastEmbedChunkReranker(
        "fake", model=FakeCrossEncoder(error=RuntimeError("inference failed"))
    )

    results, reason = rerank_with_fallback(
        reranker, "query", original, top_k=1
    )

    assert results == original[:1]
    assert reason == "RuntimeError: inference failed"


def test_diverse_selection_prefers_distinct_pages_before_duplicates():
    candidates = [
        _result("p1-a", "first", 0.9, page_number=1),
        _result("p1-b", "duplicate page", 0.8, page_number=1),
        _result("p2", "second page", 0.7, page_number=2),
    ]

    results = select_diverse_chunks(candidates, top_k=2)

    assert [result.chunk_id for result in results] == ["p1-a", "p2"]


def test_diverse_selection_keeps_focused_subquery_evidence():
    candidates = [
        _result("high", "highest", 0.95, page_number=1),
        _result("middle", "middle", 0.85, page_number=2),
        replace(
            _result("focused", "formula", 0.55, page_number=7), focus_rank=1
        ),
    ]

    results = select_diverse_chunks(candidates, top_k=2)

    assert [result.chunk_id for result in results] == ["high", "focused"]


def test_rerank_pool_keeps_focused_evidence_below_initial_cutoff():
    candidates = [
        _result(f"high-{index}", "high", 1 - index / 100, page_number=index)
        for index in range(1, 5)
    ]
    candidates.append(
        replace(
            _result("focused", "formula", 0.4, page_number=7),
            focus_rank=1,
            routing_rank=1,
        )
    )

    results = select_rerank_candidates(candidates, top_k=3)

    assert [result.chunk_id for result in results] == ["high-1", "high-2", "focused"]


def test_method_section_receives_a_small_reranking_bonus():
    reranker = FastEmbedChunkReranker(
        "fake", weight=0.0, model=FakeCrossEncoder([1.0, 1.0])
    )
    candidates = [
        _result("intro", "same", 0.7, section="1 Introduction"),
        _result("method", "same", 0.68, section="4 Methodology"),
    ]

    results = reranker.rerank("query", candidates, top_k=2)

    assert results[0].chunk_id == "method"


@pytest.mark.parametrize("weight", [-0.1, 1.1])
def test_cross_encoder_rejects_invalid_weight(weight):
    with pytest.raises(ValueError, match="weight"):
        FastEmbedChunkReranker("fake", weight=weight, model=FakeCrossEncoder())
