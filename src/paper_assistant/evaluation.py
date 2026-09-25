"""Evaluation utilities for fuzzy paper retrieval."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from src.paper_assistant.retriever import PaperBM25Retriever


@dataclass(frozen=True)
class EvaluationCaseResult:
    query_id: str
    expected_paper_id: str
    rank: int | None
    returned_paper_ids: tuple[str, ...]
    split: str
    difficulty: str


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
        for required in ("id", "description", "expected_paper_id"):
            if not case.get(required):
                raise ValueError(
                    f"Evaluation line {line_number} is missing {required}"
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
    retriever: PaperBM25Retriever,
    cases: Iterable[dict[str, Any]],
    *,
    split: str | None = None,
) -> dict[str, Any]:
    selected_cases = [case for case in cases if split is None or case.get("split") == split]
    results: list[EvaluationCaseResult] = []
    for case in selected_cases:
        matches = retriever.search(case["description"], top_k=len(retriever.catalog))
        returned_ids = tuple(match.paper.paper_id for match in matches)
        try:
            rank = returned_ids.index(case["expected_paper_id"]) + 1
        except ValueError:
            rank = None
        results.append(
            EvaluationCaseResult(
                query_id=case["id"],
                expected_paper_id=case["expected_paper_id"],
                rank=rank,
                returned_paper_ids=returned_ids[:3],
                split=case.get("split", "unspecified"),
                difficulty=case.get("difficulty", "unspecified"),
            )
        )

    groups: dict[str, list[EvaluationCaseResult]] = defaultdict(list)
    for result in results:
        groups[result.difficulty].append(result)
    return {
        "retriever": "paper_bm25",
        "split": split or "all",
        "overall": _metrics(results),
        "by_difficulty": {
            difficulty: _metrics(group) for difficulty, group in sorted(groups.items())
        },
        "cases": [asdict(result) for result in results],
    }
