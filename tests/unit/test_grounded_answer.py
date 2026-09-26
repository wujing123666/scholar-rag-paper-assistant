"""Tests for evidence-bound paper answer generation."""

from __future__ import annotations

import json

import pytest

from src.libs.llm.base_llm import ChatResponse
from src.paper_assistant.catalog import PaperProfile
from src.paper_assistant.chunk_retriever import ChunkSearchResult
from src.paper_assistant.grounded_answer import REFUSAL_TEXT, answer_from_evidence


def _result(chunk_id: str = "paper:4:0") -> ChunkSearchResult:
    paper = PaperProfile(
        paper_id="paper",
        pdf_files=("paper.pdf",),
        canonical_title="Evidence Paper",
        title_zh="证据论文",
        authors=("Author",),
        year="2026",
        venue="Test",
        languages=("en",),
        tags=(),
        method_summaries=(),
        datasets=(),
        memory_cues=(),
        duplicate_groups=(),
    )
    return ChunkSearchResult(
        chunk_id=chunk_id,
        paper=paper,
        pdf_file="paper.pdf",
        page_number=4,
        chunk_index=0,
        section="METHOD",
        text="The method fuses two predictions according to their uncertainty.",
        score=0.82,
        dense_score=0.8,
    )


class FakeLLM:
    def __init__(self, payload=None, *, content=None, error=None):
        self.payload = payload
        self.content = content
        self.error = error
        self.calls = []

    def chat(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        if self.error:
            raise self.error
        content = self.content or json.dumps(self.payload, ensure_ascii=False)
        return ChatResponse(
            content=content,
            model="fake-model",
            usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        )


def test_valid_claims_are_assembled_with_verified_source_metadata():
    llm = FakeLLM(
        {
            "status": "answered",
            "claims": [{"text": "该方法按不确定性融合两路预测。", "citations": ["C1"]}],
        }
    )

    answer = answer_from_evidence(llm, "如何融合预测？", [_result()])

    assert answer.status == "answered"
    assert answer.answer == "该方法按不确定性融合两路预测。 [C1]"
    assert answer.citations[0].paper_id == "paper"
    assert answer.citations[0].page_number == 4
    assert answer.citations[0].section == "METHOD"
    assert answer.citations[0].chunk_id == "paper:4:0"
    messages, kwargs = llm.calls[0]
    assert "EVIDENCE_JSON" in messages[1].content
    assert "全部子问题" in messages[0].content
    assert "避免用多条 claim 重复同一概述" in messages[0].content
    assert '"citation_id": "C1"' in messages[1].content
    assert kwargs["temperature"] == 0.0


@pytest.mark.parametrize(
    "payload",
    [
        {"status": "answered", "claims": [{"text": "无引用结论", "citations": []}]},
        {"status": "answered", "claims": [{"text": "伪造引用", "citations": ["C9"]}]},
        {"status": "answered", "claims": []},
        {"status": "unknown", "claims": []},
    ],
)
def test_invalid_or_uncited_claims_are_refused(payload):
    answer = answer_from_evidence(FakeLLM(payload), "问题", [_result()])

    assert answer.status == "insufficient_evidence"
    assert answer.answer == REFUSAL_TEXT
    assert answer.reason == "invalid_model_output"
    assert answer.citations == ()


def test_markdown_wrapped_json_is_accepted_and_model_markers_are_rebuilt():
    content = """```json
{"status":"answered","claims":[{"text":"结论 [C1]","citations":["C1","C1"]}]}
```"""

    answer = answer_from_evidence(FakeLLM(content=content), "问题", [_result()])

    assert answer.answer == "结论 [C1]"
    assert answer.claims[0].citations == ("C1",)


def test_model_can_report_insufficient_evidence():
    answer = answer_from_evidence(
        FakeLLM({"status": "insufficient_evidence", "claims": []}),
        "问题",
        [_result()],
    )

    assert answer.status == "insufficient_evidence"
    assert answer.reason == "model_reported_insufficient_evidence"
    assert answer.model == "fake-model"


def test_too_few_chunks_refuses_without_calling_the_model():
    llm = FakeLLM({"status": "answered", "claims": []})

    answer = answer_from_evidence(llm, "问题", [], min_evidence_chunks=1)

    assert answer.reason == "not_enough_retrieved_chunks"
    assert llm.calls == []


def test_llm_failure_returns_safe_refusal_without_internal_error_text():
    llm = FakeLLM(error=RuntimeError("secret endpoint and token"))

    answer = answer_from_evidence(llm, "问题", [_result()])

    assert answer.reason == "llm_generation_failed"
    assert "secret" not in json.dumps(answer.to_dict(), ensure_ascii=False)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"query": ""}, "query"),
        ({"max_context_chars": 999}, "max_context_chars"),
        ({"min_evidence_chunks": 0}, "min_evidence_chunks"),
    ],
)
def test_answer_validates_inputs(kwargs, message):
    arguments = {"llm": FakeLLM(), "query": "问题", "results": [_result()], **kwargs}
    with pytest.raises(ValueError, match=message):
        answer_from_evidence(**arguments)
