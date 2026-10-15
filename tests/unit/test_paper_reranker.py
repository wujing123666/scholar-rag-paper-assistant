"""Tests for paper-level Cross-Encoder relevance scoring."""

from __future__ import annotations

import math

import pytest

from src.paper_assistant.catalog import PaperProfile
from src.paper_assistant.paper_reranker import FastEmbedPaperReranker
from src.paper_assistant.retriever import PaperSearchResult


def _paper(paper_id: str, method: str) -> PaperProfile:
    return PaperProfile(
        paper_id=paper_id,
        pdf_files=(f"{paper_id}.pdf",),
        canonical_title=f"Paper {paper_id}",
        title_zh="",
        authors=(),
        year="2026",
        venue="",
        languages=("en",),
        tags=("IoT",),
        method_summaries=(method,),
        datasets=(),
        memory_cues=(),
        duplicate_groups=(),
    )


class _Model:
    def __init__(self, scores):
        self.scores = scores
        self.calls = []

    def rerank(self, query, documents, batch_size=64):
        self.calls.append((query, documents, batch_size))
        return self.scores


def test_paper_reranker_keeps_absolute_logits_and_reorders_candidates():
    model = _Model([-3.0, 4.5])
    reranker = FastEmbedPaperReranker("fake", model=model, batch_size=2)
    candidates = [
        PaperSearchResult(_paper("wrong", "image compression"), 0.9, ()),
        PaperSearchResult(_paper("right", "intrusion detection"), 0.4, ()),
    ]

    result = reranker.rerank("IoT intrusion detection", candidates, top_k=2)

    assert [item.paper.paper_id for item in result] == ["right", "wrong"]
    assert [item.score for item in result] == [4.5, -3.0]
    assert model.calls[0][0] == "IoT intrusion detection"
    assert "intrusion detection" in model.calls[0][1][1]
    assert model.calls[0][2] == 2


@pytest.mark.parametrize("scores", [[1.0], [math.nan, 2.0]])
def test_paper_reranker_rejects_invalid_scores(scores):
    candidates = [
        PaperSearchResult(_paper("a", "method a"), 1.0, ()),
        PaperSearchResult(_paper("b", "method b"), 0.5, ()),
    ]
    reranker = FastEmbedPaperReranker("fake", model=_Model(scores))

    with pytest.raises(ValueError, match="Cross-Encoder"):
        reranker.rerank("query", candidates, top_k=2)


def test_shared_reranker_serializes_lazy_model_inference():
    import threading
    from concurrent.futures import ThreadPoolExecutor

    from src.paper_assistant.paper_reranker import LazySynchronizedCrossEncoder

    class LazyModel:
        def __init__(self):
            self.barrier = threading.Barrier(2)
            self.state_lock = threading.Lock()
            self.active = 0
            self.peak = 0

        def rerank(self, query, documents, batch_size=64):
            with self.state_lock:
                self.active += 1
                self.peak = max(self.peak, self.active)
            try:
                # A broken wrapper allows both lazy inference bodies through.
                # Correct serialization makes the first wait expire, then lets
                # the next call proceed after the first iterator is exhausted.
                try:
                    self.barrier.wait(timeout=0.2)
                except threading.BrokenBarrierError:
                    pass
                yield 0.7 if query == "first" else 0.2
            finally:
                with self.state_lock:
                    self.active -= 1

    model = LazyModel()
    reranker = LazySynchronizedCrossEncoder("fake-no-network")
    reranker._model = model
    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(lambda: list(reranker.rerank("first", ["doc"])))
        second = executor.submit(lambda: list(reranker.rerank("second", ["doc"])))
        assert first.result(timeout=5) == [0.7]
        assert second.result(timeout=5) == [0.2]
    assert model.peak == 1
    assert model.active == 0


def test_lazy_inference_failure_releases_shared_reranker_lock():
    from src.paper_assistant.paper_reranker import LazySynchronizedCrossEncoder

    class LazyModel:
        def rerank(self, query, documents, batch_size=64):
            if query == "fail":
                raise RuntimeError("simulated lazy inference failure")
            yield 0.5

    reranker = LazySynchronizedCrossEncoder("fake-no-network")
    reranker._model = LazyModel()
    with pytest.raises(RuntimeError, match="lazy inference failure"):
        list(reranker.rerank("fail", ["doc"]))
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=1) as executor:
        result = executor.submit(lambda: list(reranker.rerank("ok", ["doc"])))
        assert result.result(timeout=5) == [0.5]
