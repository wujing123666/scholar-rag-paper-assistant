"""Export allowlisted measurements only from local reports; never copy PDF text or prompts.

Run from repository root after the historical reports are available.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read_report(relative: str) -> tuple[dict, str]:
    raw = (ROOT / relative).read_bytes()
    return json.loads(raw.decode("utf-8-sig")), hashlib.sha256(raw).hexdigest()


def export() -> dict:
    ab, ab_hash = read_report("tmp/ab-readonly/step3_eval_report.json")
    baseline, baseline_hash = read_report("tmp/answer_evaluation_claim_safety.json")
    replay, replay_hash = read_report("tmp/claim_cost_optimization.json")
    holdout, holdout_hash = read_report("tmp/holdout_eval/report.json")
    audit, audit_hash = read_report("tmp/holdout_eval/hallucination_audit_report.json")
    facets = []
    for item in ab["questions"]["fastembed"]:
        row = {"id": item["id"]}
        for mode in ("baseline", "facet"):
            row[mode] = {
                "aspect_ranks": [a["rank"] for a in item[mode]["per_aspect"] if a["coverable"]],
                "union_rank": item[mode]["union_rank"],
            }
        facets.append(row)
    cost = []
    base_by_id = {row["id"]: row for row in baseline["results"]}
    warm_by_id = {row["id"]: row for row in replay["warm"]["results"]}
    for cold in replay["cold"]["results"]:
        item = {
            "id": cold["id"],
            "claims": cold["original_claims"],
            "baseline_judge_tokens": base_by_id[cold["id"]]["claim_verification"]["judge_tokens"],
        }
        for mode, source in (("cold", cold), ("warm", warm_by_id[cold["id"]])):
            item[mode] = {
                key: source[key]
                for key in (
                    "judge_tokens",
                    "cache_hits",
                    "cache_misses",
                    "second_judge_claims",
                    "original_claims",
                    "accepted_claims",
                )
            }
        cost.append(item)
    return {
        "schema_version": 1,
        "scope": "Historical local development measurements; arithmetic replay only, not full retrieval reproduction or independent user validation.",
        "source_sha256": {
            "facet_report": ab_hash,
            "claim_baseline": baseline_hash,
            "claim_replay": replay_hash,
            "holdout": holdout_hash,
            "audit": audit_hash,
        },
        "facet_ab": {"top_k": 7, "reranker": "fastembed", "cases": facets},
        "claim_cost": {"second_judge_policy": "risk_based", "cases": cost},
        "holdout": {
            "scope": "100 metadata-derived known queries and 40 system-authored negatives",
            "cases": [
                {key: row[key] for key in ("case_id", "kind", "rejected", "target_rank")}
                for row in holdout["cases"]
            ],
        },
        "claim_audit": {
            "claims": audit["claims"],
            "verdict_counts": audit["verdict_counts"],
            "scope": "Independent model screening, not human gold labels. AI-assisted source spot-checks are separate.",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "docs/evaluation/data/historical-evidence.json"
    )
    args = parser.parse_args()
    data = export()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("Exported measurement fields only:", args.output.name)


if __name__ == "__main__":
    main()
