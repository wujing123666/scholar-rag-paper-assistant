"""Deterministic evaluation for page-level paper evidence retrieval."""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

from src.paper_assistant.chunk_retriever import ChunkSearchResult


class ChunkRetriever(Protocol):
    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        paper_ids: tuple[str, ...] = (),
    ) -> list[ChunkSearchResult]: ...


CandidateResolver = Callable[[str], tuple[str, ...]]


@dataclass(frozen=True)
class ChunkEvaluationCaseResult:
    query_id: str
    expected_paper_id: str
    relevant_pages: tuple[int, ...]
    candidate_paper_ids: tuple[str, ...]
    paper_routed: bool
    page_rank: int | None
    evidence_rank: int | None
    returned_chunks: tuple[dict[str, Any], ...]


def load_chunk_evaluation_cases(path: str | Path) -> list[dict[str, Any]]:
    """Load and validate a JSONL page-level retrieval development set."""
    cases = []
    seen_ids: set[str] = set()
    for line_number, line in enumerate(
        Path(path).read_text(encoding="utf-8-sig").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            case = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid JSON on chunk evaluation line {line_number}") from error
        for required in ("id", "query", "expected_paper_id"):
            if not isinstance(case.get(required), str) or not case[required].strip():
                raise ValueError(
                    f"Chunk evaluation line {line_number} is missing {required}"
                )
        if case["id"] in seen_ids:
            raise ValueError(
                f"Chunk evaluation line {line_number} has duplicate id {case['id']}"
            )
        seen_ids.add(case["id"])
        pages = case["relevant_pages"]
        if (
            not isinstance(pages, list)
            or not pages
            or any(not isinstance(page, int) or page < 1 for page in pages)
        ):
            raise ValueError(
                f"Chunk evaluation line {line_number} has invalid relevant_pages"
            )
        evidence_terms = case.get("evidence_terms", [])
        if not isinstance(evidence_terms, list) or any(
            not isinstance(term, str) or not term.strip() for term in evidence_terms
        ):
            raise ValueError(
                f"Chunk evaluation line {line_number} has invalid evidence_terms"
            )
        minimum = case.get("min_evidence_terms", len(evidence_terms))
        if (
            not isinstance(minimum, int)
            or minimum < 0
            or minimum > len(evidence_terms)
        ):
            raise ValueError(
                f"Chunk evaluation line {line_number} has invalid min_evidence_terms"
            )
        normalized = dict(case)
        normalized["relevant_pages"] = sorted(set(pages))
        normalized["evidence_terms"] = [term.strip() for term in evidence_terms]
        normalized["min_evidence_terms"] = minimum
        cases.append(normalized)
    if not cases:
        raise ValueError("Chunk evaluation set cannot be empty")
    return cases


def _normalized_text(text: str) -> str:
    return re.sub(r"\s+", "", text).casefold()


def _matches_page(result: ChunkSearchResult, case: dict[str, Any]) -> bool:
    return (
        result.paper.paper_id == case["expected_paper_id"]
        and result.page_number in case["relevant_pages"]
    )


def _matches_evidence(result: ChunkSearchResult, case: dict[str, Any]) -> bool:
    if not _matches_page(result, case):
        return False
    terms = case["evidence_terms"]
    if not terms:
        return True
    text = _normalized_text(result.text)
    hits = sum(_normalized_text(term) in text for term in terms)
    return hits >= case["min_evidence_terms"]


def _rank(
    results: list[ChunkSearchResult],
    predicate: Callable[[ChunkSearchResult], bool],
) -> int | None:
    return next(
        (rank for rank, result in enumerate(results, start=1) if predicate(result)),
        None,
    )


def _ranking_metrics(
    results: Iterable[ChunkEvaluationCaseResult],
    rank_field: str,
) -> dict[str, float | int]:
    result_list = list(results)
    count = len(result_list)
    ranks = [getattr(result, rank_field) for result in result_list]
    if not count:
        return {
            "queries": 0,
            "recall_at_1": 0.0,
            "recall_at_3": 0.0,
            "recall_at_5": 0.0,
            "mrr": 0.0,
        }
    return {
        "queries": count,
        "recall_at_1": sum(rank == 1 for rank in ranks) / count,
        "recall_at_3": sum(rank is not None and rank <= 3 for rank in ranks) / count,
        "recall_at_5": sum(rank is not None and rank <= 5 for rank in ranks) / count,
        "mrr": sum(1 / rank for rank in ranks if rank) / count,
    }


def evaluate_chunk_retriever(
    retriever: ChunkRetriever,
    cases: Iterable[dict[str, Any]],
    *,
    top_k: int = 5,
    candidate_resolver: CandidateResolver | None = None,
) -> dict[str, Any]:
    """Evaluate oracle-paper or routed page-level evidence retrieval."""
    if top_k < 1:
        raise ValueError("top_k must be at least one")
    selected_cases = list(cases)
    results: list[ChunkEvaluationCaseResult] = []
    for case in selected_cases:
        expected_paper_id = case["expected_paper_id"]
        if candidate_resolver is None:
            candidate_ids = (expected_paper_id,)
        else:
            candidate_ids = tuple(dict.fromkeys(candidate_resolver(case["query"])))

        matches: list[ChunkSearchResult] = []
        if candidate_ids:
            if candidate_resolver is None:
                matches = retriever.search(
                    case["query"], top_k=top_k, paper_ids=candidate_ids
                )
            else:
                from src.paper_assistant.chunk_retriever import (
                    apply_paper_routing_prior,
                )

                for paper_id in candidate_ids:
                    matches.extend(
                        retriever.search(
                            case["query"], top_k=top_k, paper_ids=(paper_id,)
                        )
                    )
                matches = apply_paper_routing_prior(matches, candidate_ids)[:top_k]

        page_rank = _rank(matches, lambda result: _matches_page(result, case))
        evidence_rank = _rank(matches, lambda result: _matches_evidence(result, case))
        results.append(
            ChunkEvaluationCaseResult(
                query_id=case["id"],
                expected_paper_id=expected_paper_id,
                relevant_pages=tuple(case["relevant_pages"]),
                candidate_paper_ids=candidate_ids,
                paper_routed=expected_paper_id in candidate_ids,
                page_rank=page_rank,
                evidence_rank=evidence_rank,
                returned_chunks=tuple(
                    {
                        "rank": rank,
                        "paper_id": match.paper.paper_id,
                        "page_number": match.page_number,
                        "chunk_id": match.chunk_id,
                        "score": round(match.score, 6),
                    }
                    for rank, match in enumerate(matches, start=1)
                ),
            )
        )

    count = len(results)
    routed = sum(result.paper_routed for result in results)
    return {
        "mode": "oracle_paper" if candidate_resolver is None else "two_stage",
        "top_k": top_k,
        "paper_routing": {
            "queries": count,
            "hits": routed,
            "recall": routed / count if count else 0.0,
        },
        "page": _ranking_metrics(results, "page_rank"),
        "evidence": _ranking_metrics(results, "evidence_rank"),
        "cases": [asdict(result) for result in results],
    }
