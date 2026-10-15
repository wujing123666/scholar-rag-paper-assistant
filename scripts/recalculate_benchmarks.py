"""Recalculate historical metrics from the public sanitized observations; no API calls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "docs/evaluation/data/historical-evidence.json"


def calculate(data: dict) -> dict:
    facets = data["facet_ab"]["cases"]
    top_k = data["facet_ab"]["top_k"]
    if not facets or len({row["id"] for row in facets}) != len(facets):
        raise ValueError("Facet cases must be nonempty with unique ids")
    for row in facets:
        if not row["baseline"]["aspect_ranks"] or len(row["baseline"]["aspect_ranks"]) != len(
            row["facet"]["aspect_ranks"]
        ):
            raise ValueError("Paired aspect denominators must match")
    total = sum(len(row["baseline"]["aspect_ranks"]) for row in facets)
    ab = {}
    for mode in ("baseline", "facet"):
        covered = sum(
            rank is not None and rank <= top_k
            for row in facets
            for rank in row[mode]["aspect_ranks"]
        )
        ab[mode] = {
            "covered_aspects": covered,
            "coverable_aspects": total,
            "coverage_at_7": covered / total,
            "all_aspects_covered": sum(
                all(rank is not None and rank <= top_k for rank in row[mode]["aspect_ranks"])
                for row in facets
            ),
            "page_recall_at_1": sum(row[mode]["union_rank"] == 1 for row in facets) / len(facets),
            "page_recall_at_5": sum(
                row[mode]["union_rank"] is not None and row[mode]["union_rank"] <= 5
                for row in facets
            )
            / len(facets),
        }
    cost_cases = data["claim_cost"]["cases"]
    baseline_tokens = sum(row["baseline_judge_tokens"] for row in cost_cases)
    cold_tokens = sum(row["cold"]["judge_tokens"] for row in cost_cases)
    if baseline_tokens <= 0:
        raise ValueError("Baseline token denominator must be positive")
    cost = {
        "cases": len(cost_cases),
        "claims": sum(row["claims"] for row in cost_cases),
        "baseline_tokens": baseline_tokens,
        "risk_cold_tokens": cold_tokens,
        "token_reduction_percent": (1 - cold_tokens / baseline_tokens) * 100,
        "cold_cache_hits": sum(row["cold"]["cache_hits"] for row in cost_cases),
        "warm_cache_hits": sum(row["warm"]["cache_hits"] for row in cost_cases),
        "warm_tokens": sum(row["warm"]["judge_tokens"] for row in cost_cases),
    }
    observations = data["holdout"]["cases"]
    known = [row for row in observations if row["kind"] == "known"]
    negative = [row for row in observations if row["kind"] != "known"]
    verdicts = data["claim_audit"]["verdict_counts"]
    claims = data["claim_audit"]["claims"]
    if sum(verdicts.values()) != claims or not claims:
        raise ValueError("Claim verdict counts must match the claim denominator")
    return {
        "purpose": "arithmetic_replay_of_historical_development_reports",
        "facet_ab": ab,
        "coverage_gain_percentage_points": (
            ab["facet"]["coverage_at_7"] - ab["baseline"]["coverage_at_7"]
        )
        * 100,
        "claim_cost": cost,
        "holdout": {
            "known": len(known),
            "known_accepted": sum(not row["rejected"] for row in known),
            "negative": len(negative),
            "negative_rejected": sum(row["rejected"] for row in negative),
        },
        "independent_model_screening": {
            "claims": claims,
            "verdict_counts": verdicts,
            "supported_fraction": verdicts["supported"] / claims,
        },
        "limits": "Recomputes measurements; does not rerun private PDFs, APIs or human annotation. No population accuracy or zero-hallucination claim.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DATA)
    parser.add_argument("--output", type=Path, default=ROOT / "tmp/public_benchmark_replay.json")
    args = parser.parse_args()
    report = calculate(json.loads(args.input.read_text(encoding="utf-8")))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
