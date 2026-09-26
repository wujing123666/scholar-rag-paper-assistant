"""Tests for grounded-answer evaluation helpers."""

from __future__ import annotations

import json

import pytest

from src.libs.llm.base_llm import ChatResponse
from src.paper_assistant.answer_evaluation import (
    ClaimSupportJudgement,
    FactJudgement,
    build_answer_case_result,
    judge_claim_support,
    judge_required_facts,
    load_answer_evaluation_cases,
    summarize_answer_results,
)
from src.paper_assistant.catalog import PaperProfile
from src.paper_assistant.chunk_retriever import ChunkSearchResult
from src.paper_assistant.grounded_answer import (
    GroundedAnswer,
    GroundedCitation,
    GroundedClaim,
)


class FakeLLM:
    def __init__(self, payload):
        self.payload = payload
        self.last_kwargs = None

    def chat(self, messages, **kwargs):
        self.last_kwargs = kwargs
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


def _evidence_result():
    paper = PaperProfile(
        paper_id="paper-a",
        pdf_files=("paper.pdf",),
        canonical_title="Paper",
        title_zh="论文",
        authors=("A",),
        year="2026",
        venue="Venue",
        languages=("zh",),
        tags=("topic",),
        method_summaries=("contribution",),
        datasets=("dataset",),
        memory_cues=("cue",),
        duplicate_groups=(),
    )
    return ChunkSearchResult(
        "chunk-1", paper, "paper.pdf", 2, 0, "Method", "支持答案的完整证据", 0.9, 0.9
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


def test_claim_judge_checks_every_claim_against_full_cited_chunk():
    llm = FakeLLM({"support": {"K1": True}})
    judgement = judge_claim_support(
        llm,
        _answer(),
        [_evidence_result()],
        model="deepseek-v4-pro",
        disable_thinking=True,
    )

    assert judgement.supported_claim_ids == ("K1",)
    assert judgement.usage == {"total_tokens": 7}
    assert llm.last_kwargs == {
        "temperature": 0.0,
        "model": "deepseek-v4-pro",
        "thinking": {"type": "disabled"},
    }


def test_fact_judge_can_override_the_generation_model():
    llm = FakeLLM({"coverage": {"F1": True}})

    judgement = judge_required_facts(
        llm,
        "答案",
        ["事实"],
        model="deepseek-v4-pro",
        disable_thinking=True,
    )

    assert judgement.covered_fact_ids == ("F1",)
    assert llm.last_kwargs == {
        "temperature": 0.0,
        "model": "deepseek-v4-pro",
        "thinking": {"type": "disabled"},
    }


def test_claim_judge_rejects_missing_chunk_and_invalid_ids():
    missing = judge_claim_support(
        FakeLLM({"support": {"K1": True}}), _answer(), []
    )
    invalid = judge_claim_support(
        FakeLLM({"support": {"K9": True}}), _answer(), [_evidence_result()]
    )

    assert missing.error == "claim_judge_missing_chunk"
    assert invalid.error == "claim_judge_failed"


def test_case_and_summary_metrics_are_computed():
    judgement = judge_required_facts(
        FakeLLM({"coverage": {"F1": True, "F2": False}}), "答案", ["一", "二"]
    )
    result = build_answer_case_result(
        _case(),
        ("paper-a", "paper-b"),
        _answer(),
        judgement,
        ClaimSupportJudgement(("K1",), "claim-judge", {"total_tokens": 5}),
        latency_seconds=1.5,
    )
    summary = summarize_answer_results([result])

    assert result["labeled_page_precision"] == 1.0
    assert result["labeled_page_recall"] == 0.5
    assert result["fact_coverage"] == 0.5
    assert result["claim_support_rate"] == 1.0
    assert summary["top1_paper_accuracy"] == 1.0
    assert summary["required_fact_coverage_micro"] == 0.5
    assert summary["claims"] == 1
    assert summary["supported_claims"] == 1
    assert summary["unsupported_claims"] == 0
    assert summary["claim_support_rate_micro"] == 1.0
    assert summary["fully_supported_answer_rate"] == 1.0
    assert summary["fully_supported_case_rate"] == 1.0
    assert summary["fact_judge_failures"] == 0
    assert summary["fact_judge_skipped"] == 0
    assert summary["claim_judge_failures"] == 0
    assert summary["claim_judge_skipped"] == 0
    assert summary["total_tokens"] == 23


def test_unanswered_case_marks_judges_skipped_instead_of_failed():
    answer = GroundedAnswer(
        "insufficient_evidence", "证据不足", (), (), "no_candidate_papers", None, None
    )
    result = build_answer_case_result(
        _case(),
        (),
        answer,
        FactJudgement((), None, None, "answer_not_generated"),
        ClaimSupportJudgement((), None, None, "answer_not_generated"),
        latency_seconds=0.1,
    )

    summary = summarize_answer_results([result])

    assert summary["fact_judge_failures"] == 0
    assert summary["fact_judge_skipped"] == 1
    assert summary["claim_judge_failures"] == 0
    assert summary["claim_judge_skipped"] == 1


def test_loader_rejects_duplicate_ids(tmp_path):
    path = tmp_path / "cases.jsonl"
    line = json.dumps(_case(), ensure_ascii=False)
    path.write_text(f"{line}\n{line}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate id"):
        load_answer_evaluation_cases(path)
