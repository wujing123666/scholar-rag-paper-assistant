"""Unit tests for the paper-level MCP tool."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from src.libs.embedding.base_embedding import BaseEmbedding
from src.mcp_server.protocol_handler import ProtocolHandler
from src.mcp_server.tools.find_paper import (
    TOOL_INPUT_SCHEMA,
    TOOL_NAME,
    FindPaperTool,
    register_tool,
)

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


@pytest.fixture
def catalog_path(tmp_path):
    path = tmp_path / "papers.csv"
    rows = [
        {
            "paper_id": "tcdi",
            "pdf_file": "tcdi-draft.pdf",
            "canonical_title": "Topology-aware Conditional Diffusion Imputation",
            "title_zh": "拓扑感知条件扩散插补",
            "authors": "A;B",
            "year": "2025",
            "venue": "TestConf",
            "language": "zh",
            "tags": "扩散模型;切比雪夫",
            "method_summary": "使用切比雪夫图滤波和条件扩散补全数据",
            "datasets": "PEMS-BAY",
            "memory_cues": "学生五步采样",
            "duplicate_group": "tcdi-versions",
        },
        {
            "paper_id": "tcdi",
            "pdf_file": "tcdi-revision.pdf",
            "canonical_title": "Topology-aware Conditional Diffusion Imputation",
            "title_zh": "拓扑感知条件扩散插补",
            "authors": "A;B",
            "year": "2025",
            "venue": "TestConf",
            "language": "zh",
            "tags": "扩散模型;切比雪夫",
            "method_summary": "五步学生噪声预测版本",
            "datasets": "PEMS-BAY",
            "memory_cues": "修订版",
            "duplicate_group": "tcdi-versions",
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
            "tags": "强化学习;用户招募",
            "method_summary": "使用强化学习动态选择参与用户",
            "datasets": "Geolife",
            "memory_cues": "预算分配",
            "duplicate_group": "",
        },
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CATALOG_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return path


def test_find_papers_returns_ranked_paper_level_candidates(catalog_path):
    tool = FindPaperTool(catalog_path, min_scores={"bm25": 0.0})

    payload = tool.find_papers("扩散模型加切比雪夫", top_k=3)

    assert payload["retriever"] == "bm25"
    assert payload["rejected"] is False
    assert payload["result_count"] == 1
    assert payload["results"][0]["paper_id"] == "tcdi"
    assert payload["results"][0]["pdf_files"] == [
        "tcdi-draft.pdf",
        "tcdi-revision.pdf",
    ]
    assert tool._embedding is None


@pytest.mark.asyncio
async def test_execute_returns_text_and_structured_content(catalog_path):
    result = await FindPaperTool(
        catalog_path, min_scores={"bm25": 0.0}
    ).execute(
        "使用强化学习进行用户招募", top_k=1
    )

    assert result.isError is False
    assert result.structuredContent["results"][0]["paper_id"] == "recruitment"
    assert "在线用户招募" in result.content[0].text
    assert "recruitment.pdf" in result.content[0].text


@pytest.mark.asyncio
async def test_execute_rejects_a_weak_match_to_an_unknown_paper(catalog_path):
    result = await FindPaperTool(catalog_path).execute(
        "使用强化学习控制机器人的论文", top_k=3
    )

    assert result.isError is False
    assert result.structuredContent["rejected"] is True
    assert result.structuredContent["rejection_reason"] == "below_threshold"
    assert result.structuredContent["results"] == []
    assert "没有找到足够可靠的匹配" in result.content[0].text


def test_rejection_threshold_can_be_configured(catalog_path):
    tool = FindPaperTool(catalog_path, min_scores={"bm25": 0.0})

    payload = tool.find_papers("使用强化学习控制机器人的论文")

    assert payload["rejected"] is False
    assert payload["results"][0]["paper_id"] == "recruitment"


def test_hybrid_initialization_failure_falls_back_to_bm25(catalog_path):
    tool = FindPaperTool(catalog_path, min_scores={"bm25": 0.0})
    original = tool._get_retriever

    def get_retriever(name):
        if name == "hybrid":
            raise ConnectionError("chroma unavailable")
        return original(name)

    tool._get_retriever = get_retriever
    payload = tool.find_papers("强化学习用户招募", retriever="hybrid")

    assert payload["retriever"] == "hybrid"
    assert payload["effective_retriever"] == "paper_bm25"
    assert payload["retriever_fallback"] == (
        "dense_initialization_unavailable:ConnectionError"
    )
    assert payload["results"][0]["paper_id"] == "recruitment"


@pytest.mark.parametrize(
    ("query", "top_k", "retriever", "message"),
    [
        ("", 3, "bm25", "query cannot be empty"),
        ("论文", 0, "bm25", "between 1 and 10"),
        ("论文", 3, "unknown", "one of"),
    ],
)
def test_find_papers_validates_arguments(
    catalog_path, query, top_k, retriever, message
):
    with pytest.raises(ValueError, match=message):
        FindPaperTool(catalog_path).find_papers(
            query, top_k=top_k, retriever=retriever
        )


def test_schema_and_registration():
    handler = ProtocolHandler(server_name="test", server_version="0")

    register_tool(handler)

    assert TOOL_NAME == "find_paper"
    assert TOOL_INPUT_SCHEMA["required"] == ["query"]
    assert handler.tools["find_paper"].input_schema["properties"]["retriever"][
        "enum"
    ] == ["bm25", "dense", "hybrid"]


def test_catalog_edit_invalidates_cached_retriever(catalog_path):
    tool = FindPaperTool(catalog_path, min_scores={"bm25": 0.0})
    first_retriever = tool._get_retriever("bm25")
    assert tool.find_papers("图神经网络")["result_count"] == 0

    with catalog_path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CATALOG_FIELDS)
        writer.writerow(
            {
                "paper_id": "gnn",
                "pdf_file": "gnn.pdf",
                "canonical_title": "Graph Neural Networks",
                "title_zh": "图神经网络综述",
                "authors": "D",
                "year": "2026",
                "venue": "Journal",
                "language": "zh",
                "tags": "图神经网络",
                "method_summary": "总结图神经网络方法",
                "datasets": "",
                "memory_cues": "图结构学习",
                "duplicate_group": "",
            }
        )

    payload = tool.find_papers("图神经网络")

    assert payload["results"][0]["paper_id"] == "gnn"
    assert tool._get_retriever("bm25") is not first_retriever


@pytest.mark.asyncio
async def test_missing_catalog_error_does_not_expose_absolute_path(tmp_path):
    missing = tmp_path / "private" / "missing.csv"

    result = await FindPaperTool(missing).execute("扩散模型")

    assert result.isError is True
    assert "论文目录不可用" in result.content[0].text
    assert str(Path(missing).resolve()) not in result.content[0].text


@pytest.mark.asyncio
async def test_retriever_error_does_not_expose_internal_details(
    catalog_path, tmp_path
):
    class BrokenEmbedding(BaseEmbedding):
        def embed(self, texts, trace=None, **kwargs):
            raise RuntimeError("secret model path C:/private/model.onnx")

        def get_dimension(self):
            return 2

    result = await FindPaperTool(
        catalog_path,
        embedding=BrokenEmbedding(),
        chroma_path=tmp_path / "chroma",
    ).execute("扩散模型", retriever="dense")

    assert result.isError is True
    assert "检索器暂时不可用" in result.content[0].text
    assert "C:/private" not in result.content[0].text
