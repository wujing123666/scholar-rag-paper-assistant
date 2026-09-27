"""Conservative runtime filtering for claims that lack sufficient citations."""

from __future__ import annotations

import re
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from src.paper_assistant.answer_evaluation import judge_claim_support
from src.paper_assistant.chunk_retriever import ChunkSearchResult
from src.paper_assistant.claim_judgement_cache import ClaimJudgementCache
from src.paper_assistant.grounded_answer import (
    REFUSAL_TEXT,
    ChatModel,
    GroundedAnswer,
    GroundedCitation,
    GroundedClaim,
)

SupplementRetriever = Callable[[str, tuple[str, ...], int], list[ChunkSearchResult]]
EVIDENCE_ONLY_TEXT = "严格核验暂时不可用，以下仅返回检索到的论文证据。"


@dataclass(frozen=True)
class ClaimSafetyResult:
    answer: GroundedAnswer
    report: dict[str, Any]
    evidence_results: tuple[ChunkSearchResult, ...]


@dataclass(frozen=True)
class ClaimRisk:
    high_risk: bool
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class _JudgeBatchResult:
    supported_claim_ids: frozenset[str]
    actual_model: str | None
    usage_tokens: int
    error: str | None
    cache_hits: int
    cache_misses: int
    cache_writes: int
    cache_errors: int


_QUANTITATIVE = re.compile(r"\d|%|％|百分之|倍|百分点")
_COMPARATIVE = re.compile(
    r"优于|高于|低于|多于|少于|提升|提高|降低|减少|显著|更(?:高|低|快|慢|好|差)"
    r"|outperform|higher|lower|increase|decrease|significant",
    re.IGNORECASE,
)
_CAUSAL = re.compile(
    r"导致|因此|因而|使得|从而|原因|归因于|caus(?:e|es|ed|al)|therefore|result in",
    re.IGNORECASE,
)
_RELATION = re.compile(
    r"输入|输出|融合|初始化|依赖|驱动|传递|先.+再|由.+组成|input|output|fuse|initialize|depend",
    re.IGNORECASE,
)


def assess_claim_risk(claim: GroundedClaim) -> ClaimRisk:
    """Flag claims whose semantics make a second judge more valuable."""
    reasons = []
    if len(set(claim.citations)) > 1:
        reasons.append("multiple_citations")
    if _QUANTITATIVE.search(claim.text):
        reasons.append("quantitative")
    if _COMPARATIVE.search(claim.text):
        reasons.append("comparative")
    if _CAUSAL.search(claim.text):
        reasons.append("causal")
    if _RELATION.search(claim.text):
        reasons.append("component_relation")
    if len(claim.text) > 100:
        reasons.append("long_claim")
    return ClaimRisk(bool(reasons), tuple(reasons))


def _citation_from_result(
    result: ChunkSearchResult, citation_id: str
) -> GroundedCitation:
    return GroundedCitation(
        citation_id=citation_id,
        paper_id=result.paper.paper_id,
        paper_title=result.paper.display_title,
        pdf_file=result.pdf_file,
        page_number=result.page_number,
        section=result.section,
        chunk_id=result.chunk_id,
        score=round(result.score, 6),
        excerpt=result.text[:300],
    )


def _answer_with_claims(
    source: GroundedAnswer,
    claims: list[GroundedClaim],
    citations_by_id: dict[str, GroundedCitation],
) -> GroundedAnswer:
    used_ids = tuple(
        dict.fromkeys(citation_id for claim in claims for citation_id in claim.citations)
    )
    return GroundedAnswer(
        status="answered",
        answer="\n".join(
            f"{claim.text} [{', '.join(claim.citations)}]" for claim in claims
        ),
        claims=tuple(claims),
        citations=tuple(citations_by_id[citation_id] for citation_id in used_ids),
        reason=None,
        model=source.model,
        usage=source.usage,
        service_diagnostics=source.service_diagnostics,
    )


def _verification_unavailable_answer(source: GroundedAnswer) -> GroundedAnswer:
    return GroundedAnswer(
        status="verification_unavailable",
        answer=EVIDENCE_ONLY_TEXT,
        claims=(),
        citations=source.citations,
        reason="claim_judge_unavailable",
        model=source.model,
        usage=source.usage,
        service_diagnostics=source.service_diagnostics,
    )


def _validate_claim_structure(
    answer: GroundedAnswer,
    evidence_by_chunk: dict[str, ChunkSearchResult],
) -> tuple[list[tuple[str, GroundedClaim]], list[dict[str, Any]]]:
    citations_by_id = {citation.citation_id: citation for citation in answer.citations}
    valid: list[tuple[str, GroundedClaim]] = []
    invalid: list[dict[str, Any]] = []
    for index, claim in enumerate(answer.claims, start=1):
        claim_id = f"K{index}"
        reason = None
        if not claim.text.strip() or not claim.citations:
            reason = "claim_or_citation_empty"
        for citation_id in claim.citations:
            citation = citations_by_id.get(citation_id)
            if citation is None:
                reason = "citation_not_found"
                break
            result = evidence_by_chunk.get(citation.chunk_id)
            if result is None:
                reason = "chunk_not_found"
                break
            if (
                citation.paper_id != result.paper.paper_id
                or citation.pdf_file != result.pdf_file
                or citation.page_number != result.page_number
            ):
                reason = "citation_metadata_mismatch"
                break
        if reason:
            invalid.append(
                {"claim_id": claim_id, "claim": claim.text, "reason": reason}
            )
        else:
            valid.append((claim_id, claim))
    return valid, invalid


def _judge_claim_pairs(
    llm: ChatModel,
    source: GroundedAnswer,
    pairs: list[tuple[str, GroundedClaim]],
    citations_by_id: dict[str, GroundedCitation],
    evidence_by_chunk: dict[str, ChunkSearchResult],
    *,
    model: str,
    cache_namespace: str,
    disable_thinking: bool,
    cache: ClaimJudgementCache | None,
) -> _JudgeBatchResult:
    cached_supported: set[str] = set()
    cached_models: list[str] = []
    misses: list[tuple[str, GroundedClaim, str | None]] = []
    cache_hits = 0
    cache_errors = 0
    for claim_id, claim in pairs:
        cache_key = None
        cached = None
        if cache is not None:
            try:
                cache_key = cache.build_key(
                    claim,
                    citations_by_id,
                    evidence_by_chunk,
                    requested_model=model,
                    service_namespace=cache_namespace,
                    disable_thinking=disable_thinking,
                )
                cached = cache.get(cache_key)
            except (OSError, ValueError, sqlite3.Error):
                cache_key = None
                cache_errors += 1
        if cached is None:
            misses.append((claim_id, claim, cache_key))
            continue
        cache_hits += 1
        if cached.supported:
            cached_supported.add(claim_id)
        if cached.actual_model:
            cached_models.append(cached.actual_model)

    if not misses:
        return _JudgeBatchResult(
            frozenset(cached_supported),
            cached_models[0] if cached_models else model,
            0,
            None,
            cache_hits,
            0,
            0,
            cache_errors,
        )

    missing_answer = _answer_with_claims(
        source,
        [claim for _, claim, _ in misses],
        citations_by_id,
    )
    judgement = judge_claim_support(
        llm,
        missing_answer,
        list(evidence_by_chunk.values()),
        model=model,
        disable_thinking=disable_thinking,
    )
    tokens = int((judgement.usage or {}).get("total_tokens", 0))
    if judgement.error:
        return _JudgeBatchResult(
            frozenset(cached_supported),
            judgement.model,
            tokens,
            judgement.error,
            cache_hits,
            len(misses),
            0,
            cache_errors,
        )

    supported = set(cached_supported)
    writes = 0
    supported_local_ids = set(judgement.supported_claim_ids)
    for index, (claim_id, _, cache_key) in enumerate(misses, start=1):
        is_supported = f"K{index}" in supported_local_ids
        if is_supported:
            supported.add(claim_id)
        if cache is not None and cache_key is not None:
            try:
                cache.put(
                    cache_key,
                    requested_model=model,
                    actual_model=judgement.model,
                    supported=is_supported,
                )
                writes += 1
            except (OSError, ValueError, sqlite3.Error):
                cache_errors += 1
    return _JudgeBatchResult(
        frozenset(supported),
        judgement.model,
        tokens,
        None,
        cache_hits,
        len(misses),
        writes,
        cache_errors,
    )


def verify_and_filter_claims(
    llm: ChatModel,
    answer: GroundedAnswer,
    evidence_results: list[ChunkSearchResult],
    *,
    judge_models: list[str],
    retrieve_more: SupplementRetriever | None = None,
    retry_k: int = 3,
    disable_thinking: bool = False,
    failure_policy: str = "strict",
    cache: ClaimJudgementCache | None = None,
    cache_namespace: str = "default",
    second_judge_policy: str = "all",
) -> ClaimSafetyResult:
    """Keep only claims supported by every judge, with one retrieval retry."""
    models = list(dict.fromkeys(model.strip() for model in judge_models if model.strip()))
    if not models:
        raise ValueError("At least one claim judge model is required")
    if retry_k < 1:
        raise ValueError("retry_k must be at least one")
    if failure_policy not in {"strict", "evidence_only"}:
        raise ValueError("failure_policy must be strict or evidence_only")
    if second_judge_policy not in {"all", "risk_based"}:
        raise ValueError("second_judge_policy must be all or risk_based")
    if answer.status != "answered":
        return ClaimSafetyResult(
            answer,
            {
                "status": "skipped",
                "reason": "answer_not_generated",
                "judge_models": models,
            },
            tuple(evidence_results),
        )

    evidence_by_chunk = {result.chunk_id: result for result in evidence_results}
    citations_by_id = {citation.citation_id: citation for citation in answer.citations}
    citation_id_by_chunk = {
        citation.chunk_id: citation.citation_id for citation in answer.citations
    }
    valid_pairs, invalid = _validate_claim_structure(answer, evidence_by_chunk)
    if not valid_pairs:
        return ClaimSafetyResult(
            GroundedAnswer(
                status="insufficient_evidence",
                answer=REFUSAL_TEXT,
                claims=(),
                citations=(),
                reason="claim_verification_removed_all",
                model=answer.model,
                usage=answer.usage,
                service_diagnostics=answer.service_diagnostics,
            ),
            {
                "status": "verified",
                "failure_policy": failure_policy,
                "judge_models": models,
                "second_judge_policy": second_judge_policy,
                "high_risk_claims": 0,
                "second_judge_claims": 0,
                "claim_risks": {},
                "original_claims": len(answer.claims),
                "accepted_claims": 0,
                "removed_claims": invalid,
                "retrieval_retries": [],
                "judge_runs": [],
                "judge_tokens": 0,
                "cache_hits": 0,
                "cache_misses": 0,
                "cache_writes": 0,
                "cache_errors": 0,
            },
            tuple(evidence_by_chunk.values()),
        )
    support_by_model: dict[str, set[str]] = {}
    judge_runs: list[dict[str, Any]] = []
    total_judge_tokens = 0
    cache_hits = 0
    cache_misses = 0
    cache_writes = 0
    cache_errors = 0
    risk_by_claim = {
        claim_id: assess_claim_risk(claim) for claim_id, claim in valid_pairs
    }
    high_risk_ids = {
        claim_id for claim_id, risk in risk_by_claim.items() if risk.high_risk
    }
    selected_by_model: dict[str, set[str]] = {}
    for model_index, model in enumerate(models):
        selected_pairs = (
            valid_pairs
            if model_index == 0 or second_judge_policy == "all"
            else [pair for pair in valid_pairs if pair[0] in high_risk_ids]
        )
        selected_by_model[model] = {claim_id for claim_id, _ in selected_pairs}
        if not selected_pairs:
            support_by_model[model] = set()
            judge_runs.append(
                {
                    "stage": "initial",
                    "requested_model": model,
                    "actual_model": None,
                    "selected_claim_ids": [],
                    "supported_claim_ids": [],
                    "error": None,
                    "skipped": "no_high_risk_claims",
                    "total_tokens": 0,
                    "cache_hits": 0,
                    "cache_misses": 0,
                    "cache_writes": 0,
                    "cache_errors": 0,
                }
            )
            continue
        judgement = _judge_claim_pairs(
            llm,
            answer,
            selected_pairs,
            citations_by_id,
            evidence_by_chunk,
            model=model,
            cache_namespace=cache_namespace,
            disable_thinking=disable_thinking,
            cache=cache,
        )
        supported_original_ids = set(judgement.supported_claim_ids)
        support_by_model[model] = (
            supported_original_ids if not judgement.error else set()
        )
        total_judge_tokens += judgement.usage_tokens
        cache_hits += judgement.cache_hits
        cache_misses += judgement.cache_misses
        cache_writes += judgement.cache_writes
        cache_errors += judgement.cache_errors
        judge_runs.append(
            {
                "stage": "initial",
                "requested_model": model,
                "actual_model": judgement.actual_model,
                "selected_claim_ids": sorted(selected_by_model[model]),
                "supported_claim_ids": sorted(supported_original_ids),
                "error": judgement.error,
                "total_tokens": judgement.usage_tokens,
                "cache_hits": judgement.cache_hits,
                "cache_misses": judgement.cache_misses,
                "cache_writes": judgement.cache_writes,
                "cache_errors": judgement.cache_errors,
            }
        )

    unavailable_models = [
        run["requested_model"] for run in judge_runs if run["error"]
    ]
    if unavailable_models:
        unavailable_answer = (
            _verification_unavailable_answer(answer)
            if failure_policy == "evidence_only"
            else GroundedAnswer(
                status="insufficient_evidence",
                answer=REFUSAL_TEXT,
                claims=(),
                citations=(),
                reason="claim_judge_unavailable",
                model=answer.model,
                usage=answer.usage,
                service_diagnostics=answer.service_diagnostics,
            )
        )
        return ClaimSafetyResult(
            unavailable_answer,
            {
                "status": "verification_unavailable",
                "failure_policy": failure_policy,
                "judge_models": models,
                "second_judge_policy": second_judge_policy,
                "high_risk_claims": len(high_risk_ids),
                "second_judge_claims": (
                    len(valid_pairs)
                    if len(models) > 1 and second_judge_policy == "all"
                    else len(high_risk_ids) if len(models) > 1 else 0
                ),
                "unavailable_models": unavailable_models,
                "original_claims": len(answer.claims),
                "accepted_claims": 0,
                "removed_claims": [],
                "retrieval_retries": [],
                "judge_runs": judge_runs,
                "judge_tokens": total_judge_tokens,
                "cache_hits": cache_hits,
                "cache_misses": cache_misses,
                "cache_writes": cache_writes,
                "cache_errors": cache_errors,
            },
            tuple(evidence_by_chunk.values()),
        )

    accepted_ids = set()
    for claim_id, _ in valid_pairs:
        required_models = [
            model for model in models if claim_id in selected_by_model[model]
        ]
        if required_models and all(
            claim_id in support_by_model[model] for model in required_models
        ):
            accepted_ids.add(claim_id)
    accepted: dict[str, GroundedClaim] = {
        claim_id: claim for claim_id, claim in valid_pairs if claim_id in accepted_ids
    }
    retry_records: list[dict[str, Any]] = []
    retry_judge_unavailable = False
    next_citation_number = max(
        (
            int(citation_id[1:])
            for citation_id in citations_by_id
            if citation_id.startswith("C") and citation_id[1:].isdigit()
        ),
        default=0,
    ) + 1

    rejected_pairs = [pair for pair in valid_pairs if pair[0] not in accepted_ids]
    for claim_id, claim in rejected_pairs:
        if retrieve_more is None:
            retry_records.append(
                {"claim_id": claim_id, "attempted": False, "accepted": False}
            )
            continue
        paper_ids = tuple(
            dict.fromkeys(citations_by_id[item].paper_id for item in claim.citations)
        )
        supplementary = retrieve_more(claim.text, paper_ids, retry_k)
        added_ids: list[str] = []
        for result in supplementary:
            if result.chunk_id in citation_id_by_chunk:
                citation_id = citation_id_by_chunk[result.chunk_id]
            else:
                citation_id = f"C{next_citation_number}"
                next_citation_number += 1
                citations_by_id[citation_id] = _citation_from_result(result, citation_id)
                citation_id_by_chunk[result.chunk_id] = citation_id
                evidence_by_chunk[result.chunk_id] = result
            if citation_id not in claim.citations and citation_id not in added_ids:
                added_ids.append(citation_id)
        if not added_ids:
            retry_records.append(
                {
                    "claim_id": claim_id,
                    "attempted": True,
                    "added_citation_ids": [],
                    "accepted": False,
                }
            )
            continue

        retried_claim = GroundedClaim(
            claim.text, tuple(dict.fromkeys((*claim.citations, *added_ids)))
        )
        retry_risk = assess_claim_risk(retried_claim)
        retry_required_models = [models[0]]
        if second_judge_policy == "all" or retry_risk.high_risk:
            retry_required_models.extend(models[1:])
        retry_supported = True
        retry_models = []
        for model in retry_required_models:
            judgement = _judge_claim_pairs(
                llm,
                answer,
                [(claim_id, retried_claim)],
                citations_by_id,
                evidence_by_chunk,
                model=model,
                cache_namespace=cache_namespace,
                disable_thinking=disable_thinking,
                cache=cache,
            )
            model_supported = (
                not judgement.error
                and claim_id in judgement.supported_claim_ids
            )
            retry_judge_unavailable = retry_judge_unavailable or bool(judgement.error)
            retry_supported = retry_supported and model_supported
            total_judge_tokens += judgement.usage_tokens
            cache_hits += judgement.cache_hits
            cache_misses += judgement.cache_misses
            cache_writes += judgement.cache_writes
            cache_errors += judgement.cache_errors
            retry_models.append(
                {
                    "requested_model": model,
                    "actual_model": judgement.actual_model,
                    "supported": model_supported,
                    "error": judgement.error,
                    "total_tokens": judgement.usage_tokens,
                    "cache_hits": judgement.cache_hits,
                    "cache_misses": judgement.cache_misses,
                    "cache_writes": judgement.cache_writes,
                    "cache_errors": judgement.cache_errors,
                }
            )
        if retry_supported:
            accepted[claim_id] = retried_claim
        retry_records.append(
            {
                "claim_id": claim_id,
                "attempted": True,
                "added_citation_ids": added_ids,
                "accepted": retry_supported,
                "risk": {
                    "high_risk": retry_risk.high_risk,
                    "reasons": list(retry_risk.reasons),
                },
                "judges": retry_models,
            }
        )

    if retry_judge_unavailable and failure_policy == "evidence_only":
        return ClaimSafetyResult(
            _verification_unavailable_answer(answer),
            {
                "status": "verification_unavailable",
                "failure_policy": failure_policy,
                "judge_models": models,
                "second_judge_policy": second_judge_policy,
                "high_risk_claims": len(high_risk_ids),
                "second_judge_claims": (
                    len(valid_pairs)
                    if len(models) > 1 and second_judge_policy == "all"
                    else len(high_risk_ids) if len(models) > 1 else 0
                ),
                "unavailable_models": sorted(
                    {
                        judge["requested_model"]
                        for retry in retry_records
                        for judge in retry.get("judges", [])
                        if judge.get("error")
                    }
                ),
                "original_claims": len(answer.claims),
                "accepted_claims": 0,
                "removed_claims": [],
                "retrieval_retries": retry_records,
                "judge_runs": judge_runs,
                "judge_tokens": total_judge_tokens,
                "cache_hits": cache_hits,
                "cache_misses": cache_misses,
                "cache_writes": cache_writes,
                "cache_errors": cache_errors,
            },
            tuple(evidence_by_chunk.values()),
        )

    kept_claims = [
        accepted[claim_id] for claim_id, _ in valid_pairs if claim_id in accepted
    ]
    removed = invalid + [
        {
            "claim_id": claim_id,
            "claim": claim.text,
            "reason": "not_supported_by_all_judges_after_retry",
        }
        for claim_id, claim in valid_pairs
        if claim_id not in accepted
    ]
    if kept_claims:
        filtered_answer = _answer_with_claims(answer, kept_claims, citations_by_id)
    else:
        filtered_answer = GroundedAnswer(
            status="insufficient_evidence",
            answer=REFUSAL_TEXT,
            claims=(),
            citations=(),
            reason="claim_verification_removed_all",
            model=answer.model,
            usage=answer.usage,
            service_diagnostics=answer.service_diagnostics,
        )
    return ClaimSafetyResult(
        filtered_answer,
        {
            "status": "verified",
            "failure_policy": failure_policy,
            "judge_models": models,
            "second_judge_policy": second_judge_policy,
            "high_risk_claims": len(high_risk_ids),
            "second_judge_claims": (
                len(valid_pairs)
                if len(models) > 1 and second_judge_policy == "all"
                else len(high_risk_ids) if len(models) > 1 else 0
            ),
            "claim_risks": {
                claim_id: {
                    "high_risk": risk.high_risk,
                    "reasons": list(risk.reasons),
                }
                for claim_id, risk in risk_by_claim.items()
            },
            "original_claims": len(answer.claims),
            "accepted_claims": len(kept_claims),
            "removed_claims": removed,
            "retrieval_retries": retry_records,
            "judge_runs": judge_runs,
            "judge_tokens": total_judge_tokens,
            "cache_hits": cache_hits,
            "cache_misses": cache_misses,
            "cache_writes": cache_writes,
            "cache_errors": cache_errors,
        },
        tuple(evidence_by_chunk.values()),
    )
