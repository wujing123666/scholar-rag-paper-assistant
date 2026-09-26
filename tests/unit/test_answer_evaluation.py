"""Tests for grounded-answer evaluation helpers."""

from __future__ import annotations

import json

import pytest

from src.libs.llm.base_llm import ChatResponse
from src.paper_assistant.answer_evaluation import (
    build_answer_case_result,
    judge_required_facts,
    load_answer_evaluation_cases,
    summarize_answer_results,
)
from src.paper_assistant.grounded_answer import (
    GroundedAnswer,
    GroundedCitation,
    GroundedClaim,
)


class FakeLLM:
    def __init__(self, payload):
        self.payload = payload

    def chat(self, messages, **kwargs):
        return ChatResponse(
            content=json.dumps(self.payload),
            model="judge",
            usage={"total_tokens": 7},
        )


def _case():
    return {
        "id": "q1",
        "question": "如何做？",
        "expected_paper_id": "paper-a",
        "relevant_pages": [2, 3],
        "reference_answer": "参考答案",
        "required_facts": ["事实一", "事实二"],
    }


def _answer():
    citation = GroundedCitation(
        "C1", "paper-a", "论文", "paper.pdf", 2, "Method", "chunk-1", 0.9, "证据"
    )
    return GroundedAnswer(
        "answered",
        "答案 [C1]",
        (GroundedClaim("答案", ("C1",)),),
        (citation,),
        None,
        "generator",
        {"total_tokens": 11},
    )


def test_load_answer_cases_validates_jsonl(tmp_path):
    path = tmp_path / "cases.jsonl"
    path.write_text(json.dumps(_case(), ensure_ascii=False) + "\n", encoding="utf-8")

    cases = load_answer_evaluation_cases(path)

    assert cases[0]["id"] == "q1"
    assert cases[0]["relevant_pages"] == [2, 3]


def test_fact_judge_accepts_only_known_fact_ids():
    judgement = judge_required_facts(
        FakeLLM({"coverage": {"F1": True, "F2": False}}), "答案", ["一", "二"]
    )
    failed = judge_required_facts(
        FakeLLM({"coverage": {"F9": True}}), "答案", ["一"]
    )

    assert judgement.covered_fact_ids == ("F1",)
    assert judgement.usage == {"total_tokens": 7}
    assert failed.error == "fact_judge_failed"


def test_case_and_summary_metrics_are_computed():
    judgement = judge_required_facts(
        FakeLLM({"coverage": {"F1": True, "F2": False}}), "答案", ["一", "二"]
    )
    result = build_answer_case_result(
        _case(), ("paper-a", "paper-b"), _answer(), judgement, latency_seconds=1.5
    )
    summary = summarize_answer_results([result])

    assert result["labeled_page_precision"] == 1.0
    assert result["labeled_page_recall"] == 0.5
    assert result["fact_coverage"] == 0.5
    assert summary["top1_paper_accuracy"] == 1.0
    assert summary["required_fact_coverage_micro"] == 0.5
    assert summary["total_tokens"] == 18


def test_loader_rejects_duplicate_ids(tmp_path):
    path = tmp_path / "cases.jsonl"
    line = json.dumps(_case(), ensure_ascii=False)
    path.write_text(f"{line}\n{line}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate id"):
        load_answer_evaluation_cases(path)
