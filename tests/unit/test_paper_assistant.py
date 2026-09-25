"""Tests for the paper-level ScholarRAG baseline."""

from __future__ import annotations

import csv

from src.paper_assistant.catalog import PaperCatalog
from src.paper_assistant.evaluation import evaluate_retriever
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
