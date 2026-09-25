"""Tests for optional local paper chunk reranking."""

from __future__ import annotations

import pytest

from src.paper_assistant.catalog import PaperProfile
from src.paper_assistant.chunk_reranker import (
    FastEmbedChunkReranker,
    rerank_with_fallback,
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


def _result(chunk_id: str, text: str, score: float) -> ChunkSearchResult:
    return ChunkSearchResult(
        chunk_id=chunk_id,
        paper=_profile(),
        pdf_file="paper.pdf",
        page_number=1,
        chunk_index=1,
        section="METHOD",
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


@pytest.mark.parametrize("weight", [-0.1, 1.1])
def test_cross_encoder_rejects_invalid_weight(weight):
    with pytest.raises(ValueError, match="weight"):
        FastEmbedChunkReranker("fake", weight=weight, model=FakeCrossEncoder())
