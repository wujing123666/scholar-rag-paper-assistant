"""Tests for replaying judges over frozen grounded answers."""

from __future__ import annotations

import json

import pytest

from src.libs.llm.base_llm import ChatResponse
from src.paper_assistant.judge_comparison import (
    JudgeSpec,
    compare_frozen_judge_specs,
    compare_frozen_judges,
    load_audit_labels,
    load_frozen_report,
    score_audit_labels,
    select_audit_claims,
    write_audit_markdown,
)


class ModelAwareLLM:
    def chat(self, messages, **kwargs):
        model = kwargs["model"]
        support = {"K1": True, "K2": model == "judge-a"}
        return ChatResponse(
            content=json.dumps({"support": support}),
            model=model,
            usage={"total_tokens": 10},
        )


class FixedJudgeLLM:
    def __init__(self, supported):
        self.supported = supported

    def chat(self, messages, **kwargs):
        return ChatResponse(
            content=json.dumps({"support": self.supported}),
            model=kwargs["model"],
            usage={"total_tokens": 7},
        )


def _report():
    return {
        "config": {"frozen": True},
        "results": [
            {
                "id": "q1",
                "question": "方法是什么？",
                "status": "answered",
                "claims": [
                    {"text": "共同支持", "citations": ["C1"]},
                    {"text": "存在分歧", "citations": ["C2"]},
                ],
                "evidence": [
                    {
                        "citation_id": "C1",
                        "paper_id": "p1",
                        "page_number": 2,
                        "section": "Method",
                        "text": "证据一",
                    },
                    {
                        "citation_id": "C2",
                        "paper_id": "p1",
                        "page_number": 3,
                        "section": "Method",
                        "text": "证据二",
                    },
                ],
            }
        ],
    }


def test_compare_frozen_judges_finds_claim_level_disagreement():
    comparison = compare_frozen_judges(
        ModelAwareLLM(), _report(), ["judge-a", "judge-b"]
    )

    assert comparison["summary"]["claims"] == 2
    assert comparison["summary"]["agreement_claims"] == 1
    assert comparison["summary"]["disagreement_claims"] == 1
    assert comparison["summary"]["agreement_rate"] == 0.5
    assert comparison["claims"][1]["decisions"] == {
        "judge-a": True,
        "judge-b": False,
    }


def test_compare_frozen_judges_supports_independent_provider_clients():
    comparison = compare_frozen_judge_specs(
        _report(),
        [
            JudgeSpec(
                label="deepseek-pro",
                llm=FixedJudgeLLM({"K1": True, "K2": True}),
                model="deepseek-v4-pro",
                disable_thinking=True,
            ),
            JudgeSpec(
                label="qwen-plus",
                llm=FixedJudgeLLM({"K1": True, "K2": False}),
                model="qwen-plus",
            ),
        ],
    )

    assert comparison["judge_models"] == ["deepseek-pro", "qwen-plus"]
    assert comparison["claims"][1]["decisions"] == {
        "deepseek-pro": True,
        "qwen-plus": False,
    }
    assert comparison["summary"]["models"][1]["requested_model"] == "qwen-plus"


def test_audit_selection_includes_disagreements_and_supported_sample(tmp_path):
    comparison = compare_frozen_judges(
        ModelAwareLLM(), _report(), ["judge-a", "judge-b"]
    )
    selected = select_audit_claims(comparison, agreement_sample=1, seed=7)
    destination = tmp_path / "audit.md"
    write_audit_markdown(destination, selected)
    content = destination.read_text(encoding="utf-8")

    assert {row["review_id"] for row in selected} == {"q1:K1", "q1:K2"}
    assert "审查结论：`待填写`" in content
    assert "证据二" in content


def test_frozen_report_requires_full_evidence(tmp_path):
    path = tmp_path / "old-report.json"
    report = _report()
    report["results"][0]["evidence"] = []
    path.write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")

    with pytest.raises(ValueError, match="does not contain full cited evidence"):
        load_frozen_report(path)


def test_comparison_requires_two_distinct_models():
    with pytest.raises(ValueError, match="At least two distinct"):
        compare_frozen_judges(ModelAwareLLM(), _report(), ["judge-a", "judge-a"])


def test_audit_labels_score_false_rejections_and_keep_provenance(tmp_path):
    comparison = compare_frozen_judges(
        ModelAwareLLM(), _report(), ["judge-a", "judge-b"]
    )
    path = tmp_path / "labels.json"
    path.write_text(
        json.dumps(
            {
                "reviewer": "codex",
                "review_type": "agent_evidence_audit",
                "limitations": "Not a human gold standard.",
                "labels": [
                    {
                        "review_id": "q1:K1",
                        "label": "supported",
                        "reason": "Directly stated.",
                    },
                    {
                        "review_id": "q1:K2",
                        "label": "supported",
                        "reason": "Directly stated.",
                    },
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    score = score_audit_labels(comparison, load_audit_labels(path))

    assert score["review_type"] == "agent_evidence_audit"
    assert score["summary"]["audited_claim_support_rate"] == 1.0
    assert score["summary"]["models"][0]["accuracy"] == 1.0
    assert score["summary"]["models"][1]["false_rejects"] == 1
