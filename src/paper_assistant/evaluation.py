"""Evaluation utilities for fuzzy paper retrieval."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

from src.paper_assistant.catalog import PaperCatalog
from src.paper_assistant.rejection import decide_retrieval
from src.paper_assistant.retriever import PaperSearchResult


class PaperRetriever(Protocol):
    """Minimal interface shared by sparse and dense paper retrievers."""

    name: str
    catalog: PaperCatalog

    def search(self, query: str, top_k: int = 3) -> list[PaperSearchResult]: ...


@dataclass(frozen=True)
class EvaluationCaseResult:
    query_id: str
    expected_paper_id: str | None
    rank: int | None
    returned_paper_ids: tuple[str, ...]
    split: str
    difficulty: str
    rejected: bool
    top_score: float | None
    min_score: float | None


def load_evaluation_cases(path: str | Path) -> list[dict[str, Any]]:
    cases = []
    for line_number, line in enumerate(
        Path(path).read_text(encoding="utf-8-sig").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            case = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid JSON on evaluation line {line_number}") from error
        for required in ("id", "description"):
            if not case.get(required):
                raise ValueError(
                    f"Evaluation line {line_number} is missing {required}"
                )
        if "expected_paper_id" not in case:
            raise ValueError(
                f"Evaluation line {line_number} is missing expected_paper_id"
            )
        expected = case["expected_paper_id"]
        if expected is not None and (not isinstance(expected, str) or not expected):
            raise ValueError(
                f"Evaluation line {line_number} has invalid expected_paper_id"
            )
        cases.append(case)
    if not cases:
        raise ValueError("Evaluation set cannot be empty")
    return cases


def _metrics(results: Iterable[EvaluationCaseResult]) -> dict[str, float | int]:
    result_list = list(results)
    count = len(result_list)
    if count == 0:
        return {"queries": 0, "recall_at_1": 0.0, "recall_at_3": 0.0, "mrr": 0.0}
    return {
        "queries": count,
        "recall_at_1": sum(result.rank == 1 for result in result_list) / count,
        "recall_at_3": sum(
            result.rank is not None and result.rank <= 3 for result in result_list
        )
        / count,
        "mrr": sum(1 / result.rank for result in result_list if result.rank) / count,
    }


def evaluate_retriever(
    retriever: PaperRetriever,
    cases: Iterable[dict[str, Any]],
    *,
    split: str | None = None,
    min_score: float | None = None,
) -> dict[str, Any]:
    selected_cases = [case for case in cases if split is None or case.get("split") == split]
    results: list[EvaluationCaseResult] = []
    for case in selected_cases:
        decision = None
        if min_score is None:
            matches = retriever.search(
                case["description"], top_k=len(retriever.catalog)
            )
        else:
            decision = decide_retrieval(
                retriever,
                case["description"],
                top_k=len(retriever.catalog),
                min_score=min_score,
            )
            matches = list(decision.results)
        returned_ids = tuple(match.paper.paper_id for match in matches)
        expected_paper_id = case["expected_paper_id"]
        if expected_paper_id is None:
            rank = None
        else:
            try:
                rank = returned_ids.index(expected_paper_id) + 1
            except ValueError:
                rank = None
        results.append(
            EvaluationCaseResult(
                query_id=case["id"],
                expected_paper_id=expected_paper_id,
                rank=rank,
                returned_paper_ids=returned_ids[:3],
                split=case.get("split", "unspecified"),
                difficulty=case.get("difficulty", "unspecified"),
                rejected=decision.rejected if decision else not matches,
                top_score=(
                    decision.top_score
                    if decision
                    else (matches[0].score if matches else None)
                ),
                min_score=decision.min_score if decision else None,
            )
        )

    known_results = [result for result in results if result.expected_paper_id is not None]
    unknown_results = [result for result in results if result.expected_paper_id is None]
    groups: dict[str, list[EvaluationCaseResult]] = defaultdict(list)
    for result in known_results:
        groups[result.difficulty].append(result)
    correctly_rejected = sum(result.rejected for result in unknown_results)
    correct_open_set = sum(result.rank == 1 for result in known_results) + correctly_rejected
    total = len(results)
    return {
        "retriever": retriever.name,
        "split": split or "all",
        "overall": _metrics(known_results),
        "rejection": {
            "queries": len(unknown_results),
            "correctly_rejected": correctly_rejected,
            "false_accepts": len(unknown_results) - correctly_rejected,
            "accuracy": (
                correctly_rejected / len(unknown_results) if unknown_results else None
            ),
        },
        "open_set": {
            "queries": total,
            "correct": correct_open_set,
            "accuracy": correct_open_set / total if total else None,
        },
        "by_difficulty": {
            difficulty: _metrics(group) for difficulty, group in sorted(groups.items())
        },
        "cases": [asdict(result) for result in results],
    }
