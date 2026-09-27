"""Tests for conservative runtime Claim-Evidence filtering."""

from __future__ import annotations

import json
import sqlite3

from src.libs.llm.base_llm import ChatResponse
from src.paper_assistant.catalog import PaperProfile
from src.paper_assistant.chunk_retriever import ChunkSearchResult
from src.paper_assistant.claim_judgement_cache import (
    ClaimJudgementCache,
    open_claim_judgement_cache,
)
from src.paper_assistant.claim_safety import assess_claim_risk, verify_and_filter_claims
from src.paper_assistant.grounded_answer import (
    GroundedAnswer,
    GroundedCitation,
    GroundedClaim,
)


def _paper():
    return PaperProfile(
        paper_id="paper",
        pdf_files=("paper.pdf",),
        canonical_title="Paper",
        title_zh="论文",
        authors=("A",),
        year="2026",
        venue="Venue",
        languages=("zh",),
        tags=(),
        method_summaries=(),
        datasets=(),
        memory_cues=(),
        duplicate_groups=(),
    )


def _result(chunk_id: str, page: int, text: str):
    return ChunkSearchResult(
        chunk_id,
        _paper(),
        "paper.pdf",
        page,
        page,
        "Method",
        text,
        0.9,
        0.9,
    )


def _citation(citation_id: str, result: ChunkSearchResult):
    return GroundedCitation(
        citation_id,
        "paper",
        "论文",
        "paper.pdf",
        result.page_number,
        "Method",
        result.chunk_id,
        0.9,
        result.text,
    )


def _answer(claims, results):
    citations = tuple(
        _citation(f"C{index}", result) for index, result in enumerate(results, start=1)
    )
    return GroundedAnswer(
        "answered",
        "候选答案",
        tuple(claims),
        citations,
        None,
        "generator",
        {"total_tokens": 20},
    )


class SequenceLLM:
    def __init__(self, payloads):
        self.payloads = list(payloads)
        self.calls = []

    def chat(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        payload = self.payloads.pop(0)
        return ChatResponse(
            content=json.dumps(payload),
            model=kwargs.get("model"),
            usage={"total_tokens": 7},
        )


def test_consensus_keeps_only_claims_supported_by_every_judge():
    evidence = [_result("chunk-1", 2, "支持第一条"), _result("chunk-2", 3, "定义")]
    answer = _answer(
        [
            GroundedClaim("第一条", ("C1",)),
            GroundedClaim("关系推断", ("C2",)),
        ],
        evidence,
    )
    llm = SequenceLLM(
        [
            {"support": {"K1": True, "K2": False}},
            {"support": {"K1": True, "K2": True}},
        ]
    )

    result = verify_and_filter_claims(
        llm, answer, evidence, judge_models=["flash", "pro"]
    )

    assert [claim.text for claim in result.answer.claims] == ["第一条"]
    assert [citation.citation_id for citation in result.answer.citations] == ["C1"]
    assert result.report["accepted_claims"] == 1
    assert result.report["removed_claims"][0]["claim_id"] == "K2"


def test_disputed_claim_can_pass_after_targeted_evidence_retry():
    original = _result("chunk-1", 2, "只定义F、G和D")
    supplement = _result("chunk-2", 8, "F明确融合G和D的输出")
    answer = _answer([GroundedClaim("F融合G和D的输出", ("C1",))], [original])
    llm = SequenceLLM(
        [
            {"support": {"K1": False}},
            {"support": {"K1": True}},
            {"support": {"K1": True}},
            {"support": {"K1": True}},
        ]
    )

    result = verify_and_filter_claims(
        llm,
        answer,
        [original],
        judge_models=["flash", "pro"],
        retrieve_more=lambda claim, papers, top_k: [supplement],
    )

    assert result.answer.status == "answered"
    assert result.answer.claims[0].citations == ("C1", "C2")
    assert result.answer.citations[1].chunk_id == "chunk-2"
    assert {item.chunk_id for item in result.evidence_results} == {
        "chunk-1",
        "chunk-2",
    }
    assert result.report["retrieval_retries"][0]["accepted"] is True
    assert result.report["judge_tokens"] == 28


def test_all_unsupported_claims_produce_safe_refusal_after_retry():
    original = _result("chunk-1", 2, "只定义符号")
    supplement = _result("chunk-2", 3, "仍未说明关系")
    answer = _answer([GroundedClaim("F融合G和D的输出", ("C1",))], [original])
    llm = SequenceLLM(
        [
            {"support": {"K1": False}},
            {"support": {"K1": True}},
            {"support": {"K1": False}},
            {"support": {"K1": True}},
        ]
    )

    result = verify_and_filter_claims(
        llm,
        answer,
        [original],
        judge_models=["flash", "pro"],
        retrieve_more=lambda claim, papers, top_k: [supplement],
    )

    assert result.answer.status == "insufficient_evidence"
    assert result.answer.reason == "claim_verification_removed_all"
    assert result.answer.claims == ()
    assert result.report["removed_claims"][0]["reason"].startswith("not_supported")


def test_rule_gate_rejects_citation_metadata_mismatch_without_model_call():
    evidence = _result("chunk-1", 2, "证据")
    bad_citation = GroundedCitation(
        "C1", "other-paper", "论文", "paper.pdf", 2, "Method", "chunk-1", 0.9, "证据"
    )
    answer = GroundedAnswer(
        "answered",
        "答案",
        (GroundedClaim("答案", ("C1",)),),
        (bad_citation,),
        None,
        "generator",
        None,
    )
    llm = SequenceLLM([])

    result = verify_and_filter_claims(
        llm, answer, [evidence], judge_models=["flash", "pro"]
    )

    assert result.answer.status == "insufficient_evidence"
    assert result.report["removed_claims"][0]["reason"] == "citation_metadata_mismatch"
    assert llm.calls == []


def test_judge_failure_strict_mode_refuses_without_retrieval_retry():
    evidence = _result("chunk-1", 2, "直接证据")
    answer = _answer([GroundedClaim("结论", ("C1",))], [evidence])

    class BrokenJudge:
        def chat(self, messages, **kwargs):
            raise RuntimeError("judge unavailable")

    retrieval_calls = []
    result = verify_and_filter_claims(
        BrokenJudge(),
        answer,
        [evidence],
        judge_models=["judge"],
        retrieve_more=lambda *args: retrieval_calls.append(args) or [],
    )

    assert result.answer.status == "insufficient_evidence"
    assert result.answer.reason == "claim_judge_unavailable"
    assert result.report["status"] == "verification_unavailable"
    assert retrieval_calls == []


def test_judge_failure_evidence_only_mode_returns_citations_without_claims():
    evidence = _result("chunk-1", 2, "直接证据")
    answer = _answer([GroundedClaim("结论", ("C1",))], [evidence])

    class BrokenJudge:
        def chat(self, messages, **kwargs):
            raise RuntimeError("judge unavailable")

    result = verify_and_filter_claims(
        BrokenJudge(),
        answer,
        [evidence],
        judge_models=["judge"],
        failure_policy="evidence_only",
    )

    assert result.answer.status == "verification_unavailable"
    assert result.answer.reason == "claim_judge_unavailable"
    assert result.answer.claims == ()
    assert result.answer.citations[0].chunk_id == "chunk-1"
    assert result.report["failure_policy"] == "evidence_only"


def test_risk_based_second_judge_only_receives_high_risk_claims():
    evidence = [_result("chunk-1", 2, "定义"), _result("chunk-2", 3, "融合关系")]
    answer = _answer(
        [
            GroundedClaim("方法A是一种插补框架", ("C1",)),
            GroundedClaim("模块A融合模块B的输出", ("C2",)),
        ],
        evidence,
    )
    llm = SequenceLLM(
        [
            {"support": {"K1": True, "K2": True}},
            {"support": {"K1": True}},
        ]
    )

    result = verify_and_filter_claims(
        llm,
        answer,
        evidence,
        judge_models=["first", "second"],
        second_judge_policy="risk_based",
    )

    assert result.answer.status == "answered"
    assert len(result.answer.claims) == 2
    assert result.report["high_risk_claims"] == 1
    assert result.report["second_judge_claims"] == 1
    second_prompt = llm.calls[1][0][1].content
    assert "模块A融合模块B的输出" in second_prompt
    assert "方法A是一种插补框架" not in second_prompt


def test_risk_based_second_judge_can_remove_only_the_high_risk_claim():
    evidence = [_result("chunk-1", 2, "定义"), _result("chunk-2", 3, "符号表")]
    answer = _answer(
        [
            GroundedClaim("方法A是一种插补框架", ("C1",)),
            GroundedClaim("模块A融合模块B的输出", ("C2",)),
        ],
        evidence,
    )
    llm = SequenceLLM(
        [
            {"support": {"K1": True, "K2": True}},
            {"support": {"K1": False}},
        ]
    )

    result = verify_and_filter_claims(
        llm,
        answer,
        evidence,
        judge_models=["first", "second"],
        second_judge_policy="risk_based",
    )

    assert [claim.text for claim in result.answer.claims] == [
        "方法A是一种插补框架"
    ]
    assert result.report["accepted_claims"] == 1


def test_claim_judgement_cache_reuses_verdict_without_storing_paper_text(tmp_path):
    evidence = [_result("chunk-1", 2, "一段不应写入缓存数据库的论文原文")]
    answer = _answer([GroundedClaim("这是一个定义", ("C1",))], evidence)
    cache_path = tmp_path / "claim-cache.sqlite3"
    cache = ClaimJudgementCache(cache_path)
    first_llm = SequenceLLM([{"support": {"K1": True}}])

    first = verify_and_filter_claims(
        first_llm,
        answer,
        evidence,
        judge_models=["judge"],
        cache=cache,
    )
    second_llm = SequenceLLM([])
    second = verify_and_filter_claims(
        second_llm,
        answer,
        evidence,
        judge_models=["judge"],
        cache=ClaimJudgementCache(cache_path),
    )

    assert first.report["cache_misses"] == 1
    assert first.report["cache_writes"] == 1
    assert second.report["cache_hits"] == 1
    assert second.report["cache_misses"] == 0
    assert second.report["judge_tokens"] == 0
    assert second_llm.calls == []
    assert "一段不应写入缓存数据库".encode() not in cache_path.read_bytes()


def test_claim_judgement_cache_invalidates_when_evidence_text_changes(tmp_path):
    first_evidence = [_result("chunk-1", 2, "旧证据")]
    changed_evidence = [_result("chunk-1", 2, "已修改的新证据")]
    answer = _answer([GroundedClaim("这是一个定义", ("C1",))], first_evidence)
    cache = ClaimJudgementCache(tmp_path / "claim-cache.sqlite3")
    verify_and_filter_claims(
        SequenceLLM([{"support": {"K1": True}}]),
        answer,
        first_evidence,
        judge_models=["judge"],
        cache=cache,
    )
    second_llm = SequenceLLM([{"support": {"K1": False}}])

    result = verify_and_filter_claims(
        second_llm,
        answer,
        changed_evidence,
        judge_models=["judge"],
        cache=cache,
    )

    assert result.report["cache_hits"] == 0
    assert result.report["cache_misses"] == 1
    assert result.answer.status == "insufficient_evidence"
    assert len(second_llm.calls) == 1


def test_claim_judgement_cache_reuses_unsupported_verdict(tmp_path):
    evidence = [_result("chunk-1", 2, "证据不足以支持结论")]
    answer = _answer([GroundedClaim("这是一个定义", ("C1",))], evidence)
    cache = ClaimJudgementCache(tmp_path / "claim-cache.sqlite3")

    first = verify_and_filter_claims(
        SequenceLLM([{"support": {"K1": False}}]),
        answer,
        evidence,
        judge_models=["judge"],
        cache=cache,
    )
    second_llm = SequenceLLM([])
    second = verify_and_filter_claims(
        second_llm,
        answer,
        evidence,
        judge_models=["judge"],
        cache=cache,
    )

    assert first.answer.status == "insufficient_evidence"
    assert second.answer.status == "insufficient_evidence"
    assert second.report["cache_hits"] == 1
    assert second.report["judge_tokens"] == 0
    assert second_llm.calls == []


def test_failed_judge_call_is_not_cached(tmp_path):
    evidence = [_result("chunk-1", 2, "证据")]
    answer = _answer([GroundedClaim("这是一个定义", ("C1",))], evidence)
    cache = ClaimJudgementCache(tmp_path / "claim-cache.sqlite3")

    class BrokenJudge:
        def chat(self, messages, **kwargs):
            raise RuntimeError("temporarily unavailable")

    failed = verify_and_filter_claims(
        BrokenJudge(), answer, evidence, judge_models=["judge"], cache=cache
    )
    working_llm = SequenceLLM([{"support": {"K1": True}}])
    recovered = verify_and_filter_claims(
        working_llm, answer, evidence, judge_models=["judge"], cache=cache
    )

    assert failed.report["cache_writes"] == 0
    assert recovered.report["cache_misses"] == 1
    assert recovered.answer.status == "answered"
    assert len(working_llm.calls) == 1


def test_cache_read_failure_falls_back_to_live_judge():
    evidence = [_result("chunk-1", 2, "证据")]
    answer = _answer([GroundedClaim("这是一个定义", ("C1",))], evidence)

    class BrokenCache:
        def build_key(self, *args, **kwargs):
            return "key"

        def get(self, cache_key):
            raise sqlite3.OperationalError("database unavailable")

    llm = SequenceLLM([{"support": {"K1": True}}])
    result = verify_and_filter_claims(
        llm,
        answer,
        evidence,
        judge_models=["judge"],
        cache=BrokenCache(),
    )

    assert result.answer.status == "answered"
    assert result.report["cache_errors"] == 1
    assert len(llm.calls) == 1


def test_cache_initialization_failure_returns_a_sanitized_fallback(tmp_path):
    cache, fallback = open_claim_judgement_cache(tmp_path)

    assert cache is None
    assert fallback == "cache_unavailable:OperationalError"
    assert str(tmp_path) not in fallback


def test_cache_key_separates_judge_thinking_modes(tmp_path):
    evidence = _result("chunk-1", 2, "证据")
    citation = _citation("C1", evidence)
    claim = GroundedClaim("这是一个定义", ("C1",))
    cache = ClaimJudgementCache(tmp_path / "claim-cache.sqlite3")
    arguments = {
        "claim": claim,
        "citations_by_id": {"C1": citation},
        "evidence_by_chunk": {evidence.chunk_id: evidence},
        "requested_model": "judge",
    }

    default_key = cache.build_key(**arguments)
    disabled_key = cache.build_key(**arguments, disable_thinking=True)
    other_service_key = cache.build_key(
        **arguments, service_namespace="other-provider|https://example.test/v1"
    )

    assert default_key != disabled_key
    assert default_key != other_service_key


def test_claim_risk_rules_are_deterministic():
    low = assess_claim_risk(GroundedClaim("TCDI是一种数据插补方法", ("C1",)))
    high = assess_claim_risk(
        GroundedClaim("模型将误差降低18%，并融合两个模块的输出", ("C1", "C2"))
    )

    assert low.high_risk is False
    assert high.high_risk is True
    assert {"multiple_citations", "quantitative", "comparative", "component_relation"} <= set(
        high.reasons
    )
