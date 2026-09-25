"""Tests for the paper-level ScholarRAG baseline."""

from __future__ import annotations

import csv
import sys
import types

import numpy as np

from src.libs.embedding.base_embedding import BaseEmbedding
from src.libs.embedding.fastembed_embedding import FastEmbedEmbedding
from src.libs.vector_store.chroma_store import ChromaStore
from src.paper_assistant.catalog import PaperCatalog
from src.paper_assistant.dense_retriever import PaperDenseRetriever, profile_text
from src.paper_assistant.evaluation import evaluate_retriever, load_evaluation_cases
from src.paper_assistant.hybrid_retriever import PaperHybridRetriever
from src.paper_assistant.rejection import decide_retrieval
from src.paper_assistant.retriever import PaperBM25Retriever

CATALOG_FIELDS = [
    "paper_id",
    "pdf_file",
    "canonical_title",
    "title_zh",
    "authors",
    "year",
    "venue",
    "language",
    "tags",
    "method_summary",
    "datasets",
    "memory_cues",
    "duplicate_group",
]


def _write_catalog(tmp_path):
    path = tmp_path / "catalog.csv"
    rows = [
        {
            "paper_id": "diffusion",
            "pdf_file": "draft.pdf",
            "canonical_title": "Topology Diffusion Imputation",
            "title_zh": "拓扑扩散插补",
            "authors": "A;B",
            "year": "2025",
            "venue": "TestConf",
            "language": "zh",
            "tags": "条件扩散;图拓扑",
            "method_summary": "把五十步教师扩散蒸馏为五步学生采样器",
            "datasets": "Los-loop;PEMS-BAY",
            "memory_cues": "百分之三观测率和切比雪夫图滤波",
            "duplicate_group": "diffusion-revisions",
        },
        {
            "paper_id": "diffusion",
            "pdf_file": "revision.pdf",
            "canonical_title": "Topology Diffusion Imputation",
            "title_zh": "拓扑扩散插补",
            "authors": "A;B",
            "year": "2025",
            "venue": "TestConf",
            "language": "zh",
            "tags": "条件扩散;图拓扑",
            "method_summary": "五步学生版本",
            "datasets": "Los-loop;PEMS-BAY",
            "memory_cues": "学生噪声预测修订稿",
            "duplicate_group": "diffusion-revisions",
        },
        {
            "paper_id": "recruitment",
            "pdf_file": "recruitment.pdf",
            "canonical_title": "Online User Recruitment",
            "title_zh": "在线用户招募",
            "authors": "C",
            "year": "2024",
            "venue": "IoT Journal",
            "language": "zh",
            "tags": "强化学习;预算分割",
            "method_summary": "两个强化学习模型分别选择用户和保留预算",
            "datasets": "Geolife",
            "memory_cues": "根据移动轨迹动态招募用户",
            "duplicate_group": "",
        },
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CATALOG_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return path


def test_catalog_collapses_multiple_pdf_versions(tmp_path):
    catalog = PaperCatalog.from_csv(_write_catalog(tmp_path))

    assert len(catalog) == 2
    assert catalog.get("diffusion").pdf_files == ("draft.pdf", "revision.pdf")
    assert len(catalog.get("diffusion").method_summaries) == 2


def test_retriever_uses_distinctive_memory_cues(tmp_path):
    retriever = PaperBM25Retriever(PaperCatalog.from_csv(_write_catalog(tmp_path)))

    results = retriever.search("那篇用两个强化学习模型决定招谁和保留预算的论文")

    assert results[0].paper.paper_id == "recruitment"
    assert {"强化", "学习"}.issubset(results[0].matched_terms)


def test_evaluation_reports_rank_metrics(tmp_path):
    retriever = PaperBM25Retriever(PaperCatalog.from_csv(_write_catalog(tmp_path)))
    cases = [
        {
            "id": "q1",
            "description": "Los-loop上的五步扩散学生采样器",
            "expected_paper_id": "diffusion",
            "difficulty": "hard",
            "split": "dev",
        },
        {
            "id": "q2",
            "description": "动态预算和在线用户招募",
            "expected_paper_id": "recruitment",
            "difficulty": "easy",
            "split": "dev",
        },
    ]

    report = evaluate_retriever(retriever, cases, split="dev")

    assert report["overall"] == {
        "queries": 2,
        "recall_at_1": 1.0,
        "recall_at_3": 1.0,
        "mrr": 1.0,
    }


def test_rejection_gate_hides_candidates_below_threshold(tmp_path):
    retriever = PaperBM25Retriever(PaperCatalog.from_csv(_write_catalog(tmp_path)))
    raw = retriever.search("强化学习控制机器人", top_k=2)

    assert raw
    decision = decide_retrieval(
        retriever,
        "强化学习控制机器人",
        top_k=2,
        min_score=raw[0].score + 0.001,
    )

    assert decision.rejected is True
    assert decision.reason == "below_threshold"
    assert decision.results == ()
    assert decision.candidates[0].paper.paper_id == "recruitment"
    assert decision.top_score == raw[0].score


def test_rejection_gate_handles_score_boundary_and_no_candidates(tmp_path):
    retriever = PaperBM25Retriever(PaperCatalog.from_csv(_write_catalog(tmp_path)))
    raw = retriever.search("强化学习用户招募", top_k=1)

    accepted = decide_retrieval(
        retriever,
        "强化学习用户招募",
        top_k=1,
        min_score=raw[0].score,
    )
    empty = decide_retrieval(
        retriever,
        "量子密码协议",
        top_k=1,
        min_score=0.0,
    )

    assert accepted.rejected is False
    assert accepted.reason == "accepted"
    assert empty.rejected is True
    assert empty.reason == "no_candidates"
    assert empty.top_score is None


def test_rejection_gate_rejects_negative_threshold(tmp_path):
    retriever = PaperBM25Retriever(PaperCatalog.from_csv(_write_catalog(tmp_path)))

    try:
        decide_retrieval(retriever, "强化学习", min_score=-0.1)
    except ValueError as error:
        assert "zero or greater" in str(error)
    else:
        raise AssertionError("Expected a negative threshold to be rejected")


def test_evaluation_reports_unknown_paper_rejection(tmp_path):
    retriever = PaperBM25Retriever(PaperCatalog.from_csv(_write_catalog(tmp_path)))
    cases = [
        {
            "id": "known",
            "description": "强化学习用户招募",
            "expected_paper_id": "recruitment",
        },
        {
            "id": "unknown",
            "description": "强化学习控制机器人",
            "expected_paper_id": None,
        },
    ]
    raw_unknown_score = retriever.search(cases[1]["description"])[0].score
    known_score = retriever.search(cases[0]["description"])[0].score
    threshold = (raw_unknown_score + known_score) / 2

    report = evaluate_retriever(retriever, cases, min_score=threshold)

    assert report["overall"]["recall_at_1"] == 1.0
    assert report["rejection"] == {
        "queries": 1,
        "correctly_rejected": 1,
        "false_accepts": 0,
        "accuracy": 1.0,
    }
    assert report["open_set"] == {"queries": 2, "correct": 2, "accuracy": 1.0}
    assert report["cases"][1]["rejected"] is True


def test_evaluation_loader_allows_explicit_unknown_cases(tmp_path):
    path = tmp_path / "unknown.jsonl"
    path.write_text(
        '{"id":"u1","description":"不存在的论文","expected_paper_id":null}\n',
        encoding="utf-8",
    )

    assert load_evaluation_cases(path)[0]["expected_paper_id"] is None


class _KeywordEmbedding(BaseEmbedding):
    """Small deterministic test double; no model download is required."""

    def __init__(self):
        self.calls = []

    def embed(self, texts, trace=None, **kwargs):
        del trace
        self.calls.append((tuple(texts), kwargs.get("is_query", False)))
        vectors = []
        for text in texts:
            vectors.append(
                [
                    float("强化学习" in text or "招募" in text),
                    float("扩散" in text or "插补" in text),
                ]
            )
        return vectors

    def get_dimension(self):
        return 2


def test_dense_retriever_ranks_one_vector_per_paper(tmp_path):
    catalog = PaperCatalog.from_csv(_write_catalog(tmp_path))
    embedding = _KeywordEmbedding()

    with ChromaStore(
        persist_directory=tmp_path / "chroma", collection_name="paper_profiles_test"
    ) as store:
        retriever = PaperDenseRetriever(catalog, embedding, store)
        results = retriever.search("帮我找使用强化学习的论文", top_k=2)

        assert results[0].paper.paper_id == "recruitment"
        assert results[0].matched_terms == ()
        assert len(embedding.calls[0][0]) == 2
        assert embedding.calls[0][1] is False
        assert embedding.calls[1][1] is True
        assert evaluate_retriever(
            retriever,
            [
                {
                    "id": "q1",
                    "description": "强化学习招募",
                    "expected_paper_id": "recruitment",
                }
            ],
        )["retriever"] == "paper_dense"


def test_dense_retriever_reuses_persisted_profile_vectors(tmp_path):
    catalog = PaperCatalog.from_csv(_write_catalog(tmp_path))
    first_embedding = _KeywordEmbedding()

    with ChromaStore(
        persist_directory=tmp_path / "chroma", collection_name="paper_profiles_test"
    ) as store:
        first = PaperDenseRetriever(catalog, first_embedding, store)
        second_embedding = _KeywordEmbedding()
        second = PaperDenseRetriever(catalog, second_embedding, store)

        assert first.index_sync.embedded == 2
        assert second.index_sync.embedded == 0
        assert second.index_sync.reused == 2
        assert second_embedding.calls == []
        assert set(store.list_ids()) == {"paper:diffusion", "paper:recruitment"}

        store.upsert(
            [
                {
                    "id": "paper:removed",
                    "vector": [1.0, 0.0],
                    "document": "removed paper",
                    "metadata": {
                        "paper_id": "removed",
                        "profile_hash": "old",
                        "embedding_model": "_KeywordEmbedding",
                        "embedding_dimension": 2,
                    },
                }
            ]
        )
        third = PaperDenseRetriever(catalog, _KeywordEmbedding(), store)
        assert third.index_sync.deleted == 1
        assert "paper:removed" not in store.list_ids()


def test_dense_model_change_keeps_old_index_when_reembedding_fails(tmp_path):
    catalog = PaperCatalog.from_csv(_write_catalog(tmp_path))
    original_embedding = _KeywordEmbedding()
    original_embedding.model = "test-model-v1"

    class BrokenReplacement(_KeywordEmbedding):
        model = "test-model-v2"

        def embed(self, texts, trace=None, **kwargs):
            raise RuntimeError("replacement model failed")

    with ChromaStore(
        persist_directory=tmp_path / "chroma", collection_name="paper_profiles_test"
    ) as store:
        PaperDenseRetriever(catalog, original_embedding, store)
        original_ids = set(store.list_ids())

        try:
            PaperDenseRetriever(catalog, BrokenReplacement(), store)
        except RuntimeError as error:
            assert "replacement model failed" in str(error)
        else:
            raise AssertionError("Expected replacement embedding to fail")

        assert set(store.list_ids()) == original_ids
        records = store.get_by_ids(sorted(original_ids))
        assert {
            record["metadata"]["embedding_model"] for record in records
        } == {"test-model-v1"}


def test_profile_text_keeps_human_readable_field_labels(tmp_path):
    profile = PaperCatalog.from_csv(_write_catalog(tmp_path)).get("diffusion")

    text = profile_text(profile)

    assert "中文标题：拓扑扩散插补" in text
    assert "数据集：Los-loop；PEMS-BAY" in text
    assert "记忆线索：百分之三观测率和切比雪夫图滤波" in text


def test_fastembed_provider_uses_separate_document_and_query_methods(monkeypatch):
    calls = []

    class FakeTextEmbedding:
        def __init__(self, **kwargs):
            calls.append(("init", kwargs))

        def embed(self, texts, **kwargs):
            calls.append(("documents", tuple(texts), kwargs))
            return iter([np.asarray([1.0, 0.0]) for _ in texts])

        def query_embed(self, texts, **kwargs):
            calls.append(("queries", tuple(texts), kwargs))
            return iter([np.asarray([0.0, 1.0]) for _ in texts])

    monkeypatch.setitem(
        sys.modules, "fastembed", types.SimpleNamespace(TextEmbedding=FakeTextEmbedding)
    )
    embedding = FastEmbedEmbedding(model="BAAI/bge-small-zh-v1.5", threads=2)

    assert embedding.embed(["论文档案"]) == [[1.0, 0.0]]
    assert embedding.embed(["模糊查询"], is_query=True) == [[0.0, 1.0]]
    assert calls == [
        (
            "init",
            {
                "model_name": "BAAI/bge-small-zh-v1.5",
                "cache_dir": None,
                "threads": 2,
            },
        ),
        ("documents", ("论文档案",), {}),
        ("queries", ("模糊查询",), {}),
    ]


def test_hybrid_retriever_combines_sparse_and_dense_ranks(tmp_path):
    catalog = PaperCatalog.from_csv(_write_catalog(tmp_path))
    sparse = PaperBM25Retriever(catalog)
    with ChromaStore(
        persist_directory=tmp_path / "chroma", collection_name="paper_profiles_test"
    ) as store:
        dense = PaperDenseRetriever(catalog, _KeywordEmbedding(), store)
        retriever = PaperHybridRetriever(catalog, sparse, dense)
        results = retriever.search("强化学习招募", top_k=2)

        assert results[0].paper.paper_id == "recruitment"
        assert results[0].score == 2 / 61
        assert evaluate_retriever(
            retriever,
            [
                {
                    "id": "q1",
                    "description": "强化学习招募",
                    "expected_paper_id": "recruitment",
                }
            ],
        )["retriever"] == "paper_hybrid_rrf"


def test_hybrid_retriever_requires_shared_catalog(tmp_path):
    catalog = PaperCatalog.from_csv(_write_catalog(tmp_path))
    other_catalog = PaperCatalog.from_csv(_write_catalog(tmp_path))

    with ChromaStore(
        persist_directory=tmp_path / "chroma", collection_name="paper_profiles_test"
    ) as store:
        try:
            PaperHybridRetriever(
                catalog,
                PaperBM25Retriever(other_catalog),
                PaperDenseRetriever(catalog, _KeywordEmbedding(), store),
            )
        except ValueError as error:
            assert "same paper catalog" in str(error)
        else:
            raise AssertionError("Expected mismatched catalogs to be rejected")
