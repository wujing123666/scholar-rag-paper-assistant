"""Tests for page-level evidence retrieval evaluation."""

from __future__ import annotations

import json

import pytest

from src.paper_assistant.catalog import PaperProfile
from src.paper_assistant.chunk_evaluation import (
    evaluate_chunk_retriever,
    load_chunk_evaluation_cases,
)
from src.paper_assistant.chunk_retriever import ChunkSearchResult


def _profile(paper_id: str) -> PaperProfile:
    return PaperProfile(
        paper_id=paper_id,
        pdf_files=(f"{paper_id}.pdf",),
        canonical_title=paper_id,
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


def _result(paper_id: str, page: int, text: str, score: float) -> ChunkSearchResult:
    return ChunkSearchResult(
        chunk_id=f"{paper_id}:{page}",
        paper=_profile(paper_id),
        pdf_file=f"{paper_id}.pdf",
        page_number=page,
        chunk_index=page,
        section="METHOD",
        text=text,
        score=score,
        dense_score=score,
    )


class FakeChunkRetriever:
    def __init__(self, by_paper):
        self.by_paper = by_paper
        self.calls = []

    def search(self, query, *, top_k=5, paper_ids=()):
        self.calls.append((query, top_k, paper_ids))
        results = []
        for paper_id in paper_ids:
            results.extend(self.by_paper.get(paper_id, []))
        return sorted(results, key=lambda result: -result.score)[:top_k]


def _case():
    return {
        "id": "q1",
        "query": "怎样融合两路预测？",
        "expected_paper_id": "target",
        "relevant_pages": [6, 9],
        "evidence_terms": ["Kalman", "uncertainty"],
        "min_evidence_terms": 2,
    }


def test_load_chunk_cases_validates_and_normalizes_pages(tmp_path):
    path = tmp_path / "chunks.jsonl"
    path.write_text(
        json.dumps({**_case(), "relevant_pages": [9, 6, 6]}, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )

    cases = load_chunk_evaluation_cases(path)

    assert cases[0]["relevant_pages"] == [6, 9]
    assert cases[0]["min_evidence_terms"] == 2


@pytest.mark.parametrize(
    "invalid_field,value",
    [
        ("relevant_pages", [0]),
        ("evidence_terms", [""]),
        ("min_evidence_terms", 3),
    ],
)
def test_load_chunk_cases_rejects_invalid_ground_truth(
    tmp_path, invalid_field, value
):
    path = tmp_path / "chunks.jsonl"
    path.write_text(
        json.dumps({**_case(), invalid_field: value}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Chunk evaluation line 1"):
        load_chunk_evaluation_cases(path)


def test_load_chunk_cases_rejects_duplicate_ids(tmp_path):
    path = tmp_path / "chunks.jsonl"
    line = json.dumps(_case(), ensure_ascii=False)
    path.write_text(f"{line}\n{line}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate id q1"):
        load_chunk_evaluation_cases(path)


def test_oracle_evaluation_separates_page_and_evidence_ranks():
    retriever = FakeChunkRetriever(
        {
            "target": [
                _result("target", 6, "generic method overview", 0.9),
                _result(
                    "target",
                    9,
                    "Kalman fusion uses uncertainty estimates.",
                    0.8,
                ),
            ]
        }
    )

    report = evaluate_chunk_retriever(retriever, [_case()], top_k=5)

    assert report["mode"] == "oracle_paper"
    assert report["page"]["recall_at_1"] == 1.0
    assert report["evidence"]["recall_at_1"] == 0.0
    assert report["evidence"]["recall_at_3"] == 1.0
    assert report["evidence"]["mrr"] == 0.5
    assert retriever.calls == [("怎样融合两路预测？", 5, ("target",))]


def test_evidence_matching_ignores_pdf_line_break_whitespace():
    retriever = FakeChunkRetriever(
        {
            "target": [
                _result(
                    "target",
                    6,
                    "Kalman fusion uses uncer\n tainty estimates.",
                    0.9,
                )
            ]
        }
    )

    report = evaluate_chunk_retriever(retriever, [_case()])

    assert report["evidence"]["recall_at_1"] == 1.0


def test_two_stage_evaluation_reports_routing_miss_without_chunk_search():
    retriever = FakeChunkRetriever({"other": [_result("other", 1, "text", 0.7)]})

    report = evaluate_chunk_retriever(
        retriever,
        [_case()],
        candidate_resolver=lambda _query: ("other",),
    )

    assert report["mode"] == "two_stage"
    assert report["paper_routing"]["recall"] == 0.0
    assert report["page"]["recall_at_5"] == 0.0
    assert report["cases"][0]["candidate_paper_ids"] == ("other",)


def test_two_stage_evaluation_fuses_candidates_and_finds_target_evidence():
    retriever = FakeChunkRetriever(
        {
            "target": [
                _result(
                    "target",
                    9,
                    "Kalman fusion uses uncertainty estimates.",
                    0.8,
                )
            ],
            "other": [_result("other", 2, "generic prediction fusion", 0.9)],
        }
    )

    report = evaluate_chunk_retriever(
        retriever,
        [_case()],
        candidate_resolver=lambda _query: ("target", "other"),
    )

    assert report["paper_routing"]["recall"] == 1.0
    assert report["page"]["recall_at_1"] == 1.0
    assert report["evidence"]["recall_at_1"] == 1.0
    assert len(retriever.calls) == 2


def test_two_stage_reranks_all_paper_candidates_in_one_comparable_pool():
    retriever = FakeChunkRetriever(
        {
            "target": [
                _result(
                    "target",
                    9,
                    "Kalman fusion uses uncertainty estimates.",
                    0.8,
                )
            ],
            "other": [_result("other", 2, "generic prediction fusion", 0.9)],
        }
    )

    class RecordingReranker:
        def __init__(self):
            self.calls = []

        def rerank(self, query, results, *, top_k):
            self.calls.append((query, tuple(result.chunk_id for result in results)))
            return list(reversed(results))[:top_k]

    reranker = RecordingReranker()
    report = evaluate_chunk_retriever(
        retriever,
        [_case()],
        candidate_resolver=lambda _query: ("target", "other"),
        reranker=reranker,
        rerank_candidates=5,
    )

    assert len(reranker.calls) == 1
    assert set(reranker.calls[0][1]) == {"target:9", "other:2"}
    assert report["reranker_fallbacks"] == {}


def test_default_reranking_keeps_the_original_global_top_five_candidates():
    retriever = FakeChunkRetriever(
        {
            "target": [
                _result("target", page, f"target evidence {page}", score)
                for page, score in ((1, 0.95), (2, 0.85), (3, 0.75))
            ],
            "other": [
                _result("other", page, f"other evidence {page}", score)
                for page, score in ((1, 0.9), (2, 0.8), (3, 0.1))
            ],
        }
    )

    class RecordingReranker:
        def __init__(self):
            self.candidate_ids = ()

        def rerank(self, query, results, *, top_k):
            self.candidate_ids = tuple(result.chunk_id for result in results)
            return list(reversed(results))[:top_k]

    reranker = RecordingReranker()
    report = evaluate_chunk_retriever(
        retriever,
        [_case()],
        candidate_resolver=lambda _query: ("target", "other"),
        reranker=reranker,
    )

    assert len(reranker.candidate_ids) == 5
    assert "other:3" not in reranker.candidate_ids
    assert {item["chunk_id"] for item in report["cases"][0]["returned_chunks"]} == set(
        reranker.candidate_ids
    )


def test_chunk_evaluation_rejects_non_positive_top_k():
    with pytest.raises(ValueError, match="top_k"):
        evaluate_chunk_retriever(FakeChunkRetriever({}), [_case()], top_k=0)
