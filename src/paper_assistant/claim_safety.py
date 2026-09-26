"""Conservative runtime filtering for claims that lack sufficient citations."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from src.paper_assistant.answer_evaluation import judge_claim_support
from src.paper_assistant.chunk_retriever import ChunkSearchResult
from src.paper_assistant.grounded_answer import (
    REFUSAL_TEXT,
    ChatModel,
    GroundedAnswer,
    GroundedCitation,
    GroundedClaim,
)

SupplementRetriever = Callable[[str, tuple[str, ...], int], list[ChunkSearchResult]]


@dataclass(frozen=True)
class ClaimSafetyResult:
    answer: GroundedAnswer
    report: dict[str, Any]
    evidence_results: tuple[ChunkSearchResult, ...]


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


def verify_and_filter_claims(
    llm: ChatModel,
    answer: GroundedAnswer,
    evidence_results: list[ChunkSearchResult],
    *,
    judge_models: list[str],
    retrieve_more: SupplementRetriever | None = None,
    retry_k: int = 3,
    disable_thinking: bool = False,
) -> ClaimSafetyResult:
    """Keep only claims supported by every judge, with one retrieval retry."""
    models = list(dict.fromkeys(model.strip() for model in judge_models if model.strip()))
    if not models:
        raise ValueError("At least one claim judge model is required")
    if retry_k < 1:
        raise ValueError("retry_k must be at least one")
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
    valid_claims = [claim for _, claim in valid_pairs]
    valid_answer = _answer_with_claims(answer, valid_claims, citations_by_id)

    support_by_model: dict[str, set[str]] = {}
    judge_runs: list[dict[str, Any]] = []
    total_judge_tokens = 0
    for model in models:
        judgement = judge_claim_support(
            llm,
            valid_answer,
            evidence_results,
            model=model,
            disable_thinking=disable_thinking,
        )
        supported_original_ids = {
            valid_pairs[index - 1][0]
            for claim_id in judgement.supported_claim_ids
            if claim_id.startswith("K")
            and (index := int(claim_id[1:])) <= len(valid_pairs)
        }
        support_by_model[model] = supported_original_ids if not judgement.error else set()
        tokens = int((judgement.usage or {}).get("total_tokens", 0))
        total_judge_tokens += tokens
        judge_runs.append(
            {
                "stage": "initial",
                "requested_model": model,
                "actual_model": judgement.model,
                "supported_claim_ids": sorted(supported_original_ids),
                "error": judgement.error,
                "total_tokens": tokens,
            }
        )

    accepted_ids = set.intersection(*support_by_model.values()) if support_by_model else set()
    accepted: dict[str, GroundedClaim] = {
        claim_id: claim for claim_id, claim in valid_pairs if claim_id in accepted_ids
    }
    retry_records: list[dict[str, Any]] = []
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
        retry_answer = _answer_with_claims(answer, [retried_claim], citations_by_id)
        retry_supported = True
        retry_models = []
        retry_evidence = list(evidence_by_chunk.values())
        for model in models:
            judgement = judge_claim_support(
                llm,
                retry_answer,
                retry_evidence,
                model=model,
                disable_thinking=disable_thinking,
            )
            model_supported = not judgement.error and "K1" in judgement.supported_claim_ids
            retry_supported = retry_supported and model_supported
            tokens = int((judgement.usage or {}).get("total_tokens", 0))
            total_judge_tokens += tokens
            retry_models.append(
                {
                    "requested_model": model,
                    "actual_model": judgement.model,
                    "supported": model_supported,
                    "error": judgement.error,
                    "total_tokens": tokens,
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
                "judges": retry_models,
            }
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
        )
    return ClaimSafetyResult(
        filtered_answer,
        {
            "status": "verified",
            "judge_models": models,
            "original_claims": len(answer.claims),
            "accepted_claims": len(kept_claims),
            "removed_claims": removed,
            "retrieval_retries": retry_records,
            "judge_runs": judge_runs,
            "judge_tokens": total_judge_tokens,
        },
        tuple(evidence_by_chunk.values()),
    )
