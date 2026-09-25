"""Tests for page-aware paper chunk indexing and retrieval."""

from __future__ import annotations

from pathlib import Path

import pymupdf as fitz
import pytest

from src.paper_assistant.catalog import PaperCatalog, PaperProfile
from src.paper_assistant.chunk_retriever import (
    ChunkSearchResult,
    PaperChunkRetriever,
    apply_paper_routing_prior,
    expand_chunk_query,
    extract_paper_chunks,
)


class FakeEmbedding:
    model = "fake-bge"

    def __init__(self) -> None:
        self.calls: list[tuple[list[str], bool]] = []

    def get_dimension(self) -> int:
        return 3

    def embed(self, texts, **kwargs):
        self.calls.append((list(texts), bool(kwargs.get("is_query", False))))
        return [[0.1, 0.2, 0.3] for _ in texts]


class FakeStore:
    def __init__(self) -> None:
        self.records: dict[str, dict] = {}
        self.last_filters = None

    def list_ids(self):
        return list(self.records)

    def get_by_ids(self, ids):
        return [self.records.get(record_id, {}) for record_id in ids]

    def upsert(self, records):
        for record in records:
            self.records[record["id"]] = {
                "id": record["id"],
                "text": record["document"],
                "metadata": record["metadata"],
                "score": 0.9,
            }

    def delete(self, ids):
        for record_id in ids:
            self.records.pop(record_id, None)

    def query(self, vector, top_k=10, filters=None):
        del vector
        self.last_filters = filters
        values = list(self.records.values())
        if filters and "paper_id" in filters:
            wanted = filters["paper_id"]
            if isinstance(wanted, dict):
                allowed = set(wanted["$in"])
            else:
                allowed = {wanted}
            values = [item for item in values if item["metadata"]["paper_id"] in allowed]
        return values[:top_k]


def _profile(paper_id: str = "paper_a", pdf_file: str = "paper.pdf") -> PaperProfile:
    return PaperProfile(
        paper_id=paper_id,
        pdf_files=(pdf_file,),
        canonical_title="Cold-Start Completion",
        title_zh="冷启动数据补全",
        authors=("A. Author",),
        year="2026",
        venue="TMC",
        languages=("en",),
        tags=("cold start",),
        method_summaries=("CORAL and Kalman fusion",),
        datasets=("U-Air",),
        memory_cues=("one target timeslot",),
        duplicate_groups=(),
    )


def _write_pdf(path: Path) -> None:
    document = fitz.open()
    page1 = document.new_page()
    page1.insert_text((72, 72), "I. INTRODUCTION", fontsize=14)
    page1.insert_textbox(
        fitz.Rect(72, 100, 520, 700),
        "Cold-start sparse crowdsensing has only one target-domain timeslot. " * 12,
        fontsize=10,
    )
    page2 = document.new_page()
    page2.insert_text((72, 72), "III. KALMAN FUSION", fontsize=14)
    page2.insert_textbox(
        fitz.Rect(72, 100, 520, 700),
        "The Kalman filter dynamically fuses dedicated and generalized predictions "
        "using uncertainty estimates. "
        * 12,
        fontsize=10,
    )
    document.save(path)
    document.close()


def test_extract_chunks_preserves_page_section_and_stable_ids(tmp_path):
    pdf = tmp_path / "paper.pdf"
    _write_pdf(pdf)

    first = extract_paper_chunks(_profile(), pdf, chunk_size=300, chunk_overlap=40)
    second = extract_paper_chunks(_profile(), pdf, chunk_size=300, chunk_overlap=40)

    assert [chunk.chunk_id for chunk in first] == [chunk.chunk_id for chunk in second]
    assert {chunk.page_number for chunk in first} == {1, 2}
    assert any("INTRODUCTION" in chunk.section for chunk in first)
    assert any("KALMAN FUSION" in chunk.section for chunk in first)
    assert all(chunk.pdf_file == "paper.pdf" for chunk in first)


def test_extract_chunks_does_not_treat_chart_axes_as_sections(tmp_path):
    pdf = tmp_path / "paper.pdf"
    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 72), "III. METHOD", fontsize=14)
    page.insert_text((72, 110), "20 20 0.4 0.4", fontsize=10)
    page.insert_textbox(
        fitz.Rect(72, 140, 520, 500),
        "The CORAL alignment creates pseudo historical target observations. " * 10,
        fontsize=10,
    )
    document.save(pdf)
    document.close()

    chunks = extract_paper_chunks(_profile(), pdf, chunk_size=300, chunk_overlap=40)

    assert chunks
    assert all(chunk.section == "III. METHOD" for chunk in chunks)


def test_expand_chunk_query_adds_transparent_english_aliases():
    expanded = expand_chunk_query("冷启动时用卡尔曼滤波融合专用模型和泛化模型")

    assert "cold-start" in expanded
    assert "Kalman filter" in expanded
    assert "dedicated model" in expanded
    assert "generalized model" in expanded


def test_expand_chunk_query_covers_cross_language_evaluation_terms():
    expanded = expand_chunk_query(
        "历史数据最相似片段、贝叶斯推断、主动学习、时域和频域，通过交叉注意力融合"
    )

    assert "historical data" in expanded
    assert "Frobenius norm" in expanded
    assert "Bayesian inference" in expanded
    assert "active learning" in expanded
    assert "temporal domain" in expanded
    assert "frequency domain" in expanded
    assert "cross-attention" in expanded


def test_paper_routing_prior_prevents_generic_lower_ranked_chunks_from_winning():
    first = _profile("first", "first.pdf")
    second = _profile("second", "second.pdf")
    results = [
        ChunkSearchResult("second:1", second, "second.pdf", 1, 0, "", "x", 0.84, 0.84),
        ChunkSearchResult("first:1", first, "first.pdf", 2, 0, "", "y", 0.82, 0.82),
    ]

    fused = apply_paper_routing_prior(results, ("first", "second"))

    assert fused[0].paper.paper_id == "first"
    assert fused[0].routing_rank == 1
    assert fused[0].score > fused[0].dense_score


def test_chunk_retriever_indexes_once_and_reuses_unchanged_chunks(tmp_path):
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    _write_pdf(inbox / "paper.pdf")
    catalog = PaperCatalog([_profile()])
    embedding = FakeEmbedding()
    store = FakeStore()

    first = PaperChunkRetriever(
        catalog, inbox, embedding, store, chunk_size=300, chunk_overlap=40
    )
    embedded = first.index_sync.embedded
    second = PaperChunkRetriever(
        catalog, inbox, embedding, store, chunk_size=300, chunk_overlap=40
    )

    assert embedded > 0
    assert first.index_sync.reused == 0
    assert second.index_sync.embedded == 0
    assert second.index_sync.reused == embedded
    assert len(embedding.calls) == 1


def test_chunk_search_filters_candidates_and_returns_traceable_text(tmp_path):
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    _write_pdf(inbox / "paper.pdf")
    catalog = PaperCatalog([_profile()])
    embedding = FakeEmbedding()
    store = FakeStore()
    retriever = PaperChunkRetriever(
        catalog, inbox, embedding, store, chunk_size=300, chunk_overlap=40
    )

    results = retriever.search(
        "How are the two predictions fused?", top_k=2, paper_ids=("paper_a",)
    )

    assert results
    assert store.last_filters == {"paper_id": "paper_a"}
    assert results[0].paper.paper_id == "paper_a"
    assert results[0].pdf_file == "paper.pdf"
    assert results[0].page_number >= 1
    assert results[0].text
    assert embedding.calls[-1][1] is True


def test_chunk_hybrid_search_uses_english_aliases_to_find_exact_evidence(tmp_path):
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    _write_pdf(inbox / "paper.pdf")
    retriever = PaperChunkRetriever(
        PaperCatalog([_profile()]),
        inbox,
        FakeEmbedding(),
        FakeStore(),
        chunk_size=300,
        chunk_overlap=40,
    )

    results = retriever.search(
        "卡尔曼滤波如何融合专用模型和泛化模型",
        top_k=3,
        paper_ids=("paper_a",),
    )

    assert results[0].page_number == 2
    assert "Kalman filter" in results[0].text
    assert results[0].sparse_score > 0


def test_repeated_candidate_search_reuses_the_same_query_embedding(tmp_path):
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    _write_pdf(inbox / "paper.pdf")
    embedding = FakeEmbedding()
    retriever = PaperChunkRetriever(
        PaperCatalog([_profile()]),
        inbox,
        embedding,
        FakeStore(),
        chunk_size=300,
        chunk_overlap=40,
    )

    retriever.search("卡尔曼滤波", paper_ids=("paper_a",))
    retriever.search("卡尔曼滤波", paper_ids=("paper_a",))

    query_calls = [call for call in embedding.calls if call[1] is True]
    assert len(query_calls) == 1


def test_chunk_retriever_rejects_unknown_paper_filter(tmp_path):
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    _write_pdf(inbox / "paper.pdf")
    retriever = PaperChunkRetriever(
        PaperCatalog([_profile()]),
        inbox,
        FakeEmbedding(),
        FakeStore(),
        chunk_size=300,
        chunk_overlap=40,
    )

    with pytest.raises(ValueError, match="Unknown paper_id"):
        retriever.search("query", paper_ids=("missing",))


def test_chunk_retriever_fails_before_writes_when_registered_pdf_is_missing(tmp_path):
    store = FakeStore()

    with pytest.raises(FileNotFoundError, match="paper_a"):
        PaperChunkRetriever(
            PaperCatalog([_profile()]), tmp_path, FakeEmbedding(), store
        )

    assert store.records == {}
