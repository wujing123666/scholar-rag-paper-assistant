"""Unit tests for the paper-level MCP tool."""

from __future__ import annotations

import csv

import pytest

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
    tool = FindPaperTool(catalog_path)

    payload = tool.find_papers("扩散模型加切比雪夫", top_k=3)

    assert payload["retriever"] == "bm25"
    assert payload["result_count"] == 1
    assert payload["results"][0]["paper_id"] == "tcdi"
    assert payload["results"][0]["pdf_files"] == [
        "tcdi-draft.pdf",
        "tcdi-revision.pdf",
    ]
    assert tool._embedding is None


@pytest.mark.asyncio
async def test_execute_returns_text_and_structured_content(catalog_path):
    result = await FindPaperTool(catalog_path).execute(
        "使用强化学习进行用户招募", top_k=1
    )

    assert result.isError is False
    assert result.structuredContent["results"][0]["paper_id"] == "recruitment"
    assert "在线用户招募" in result.content[0].text
    assert "recruitment.pdf" in result.content[0].text


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
