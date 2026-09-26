"""Replay multiple judges over frozen grounded answers and prepare an audit queue."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from src.paper_assistant.answer_evaluation import judge_frozen_claim_support
from src.paper_assistant.grounded_answer import ChatModel


def load_frozen_report(path: str | Path) -> dict[str, Any]:
    """Load an answer report that contains replayable full cited evidence."""
    report = json.loads(Path(path).read_text(encoding="utf-8"))
    results = report.get("results")
    if not isinstance(results, list) or not results:
        raise ValueError("Frozen answer report must contain results")
    for result in results:
        if not isinstance(result, dict) or not isinstance(result.get("id"), str):
            raise ValueError("Frozen answer report contains an invalid result")
        if result.get("status") == "answered" and not result.get("evidence"):
            raise ValueError(
                f"Frozen answer {result['id']} does not contain full cited evidence"
            )
    return report


def compare_frozen_judges(
    llm: ChatModel,
    report: dict[str, Any],
    judge_models: list[str],
    *,
    disable_thinking: bool = False,
) -> dict[str, Any]:
    """Judge identical frozen claims with every requested model."""
    models = list(dict.fromkeys(model.strip() for model in judge_models if model.strip()))
    if len(models) < 2:
        raise ValueError("At least two distinct judge models are required")

    case_judgements: dict[str, dict[str, Any]] = {}
    model_summaries = {
        model: {
            "requested_model": model,
            "actual_models": set(),
            "supported_claims": 0,
            "unsupported_claims": 0,
            "judge_failures": 0,
            "total_tokens": 0,
        }
        for model in models
    }
    for result in report["results"]:
        case_id = result["id"]
        case_judgements[case_id] = {}
        if result.get("status") != "answered":
            continue
        claim_count = len(result.get("claims", []))
        for model in models:
            judgement = judge_frozen_claim_support(
                llm,
                result,
                model=model,
                disable_thinking=disable_thinking,
            )
            supported = set(judgement.supported_claim_ids)
            case_judgements[case_id][model] = {
                "supported_claim_ids": list(judgement.supported_claim_ids),
                "model": judgement.model,
                "usage": judgement.usage,
                "error": judgement.error,
            }
            summary = model_summaries[model]
            if judgement.model:
                summary["actual_models"].add(judgement.model)
            if judgement.error:
                summary["judge_failures"] += 1
                continue
            summary["supported_claims"] += len(supported)
            summary["unsupported_claims"] += claim_count - len(supported)
            summary["total_tokens"] += int(
                (judgement.usage or {}).get("total_tokens", 0)
            )

    claim_rows: list[dict[str, Any]] = []
    for result in report["results"]:
        if result.get("status") != "answered":
            continue
        evidence_by_id = {
            item["citation_id"]: item for item in result.get("evidence", [])
        }
        judgements = case_judgements[result["id"]]
        for index, claim in enumerate(result.get("claims", []), start=1):
            claim_id = f"K{index}"
            decisions: dict[str, bool | None] = {}
            actual_models: dict[str, str | None] = {}
            for model in models:
                model_judgement = judgements.get(model, {})
                decisions[model] = (
                    claim_id in model_judgement.get("supported_claim_ids", [])
                    if not model_judgement.get("error")
                    else None
                )
                actual_models[model] = model_judgement.get("model")
            comparable = all(value is not None for value in decisions.values())
            decision_values = set(decisions.values())
            claim_rows.append(
                {
                    "review_id": f"{result['id']}:{claim_id}",
                    "case_id": result["id"],
                    "question": result.get("question"),
                    "claim_id": claim_id,
                    "claim": claim.get("text"),
                    "citation_ids": claim.get("citations", []),
                    "evidence": [
                        evidence_by_id[citation_id]
                        for citation_id in claim.get("citations", [])
                        if citation_id in evidence_by_id
                    ],
                    "decisions": decisions,
                    "actual_models": actual_models,
                    "comparable": comparable,
                    "agreement": comparable and len(decision_values) == 1,
                }
            )

    comparable_rows = [row for row in claim_rows if row["comparable"]]
    agreement_rows = [row for row in comparable_rows if row["agreement"]]
    disagreement_rows = [row for row in comparable_rows if not row["agreement"]]
    both_supported = [
        row for row in agreement_rows if all(row["decisions"].values())
    ]
    both_unsupported = [
        row
        for row in agreement_rows
        if all(value is False for value in row["decisions"].values())
    ]
    summaries = []
    for model in models:
        item = model_summaries[model]
        summaries.append(
            {
                **item,
                "actual_models": sorted(item["actual_models"]),
                "support_rate": (
                    item["supported_claims"]
                    / (item["supported_claims"] + item["unsupported_claims"])
                    if item["supported_claims"] + item["unsupported_claims"]
                    else 0.0
                ),
            }
        )
    return {
        "source_config": report.get("config"),
        "judge_models": models,
        "summary": {
            "cases": len(report["results"]),
            "answered_cases": sum(
                result.get("status") == "answered" for result in report["results"]
            ),
            "claims": len(claim_rows),
            "comparable_claims": len(comparable_rows),
            "agreement_claims": len(agreement_rows),
            "disagreement_claims": len(disagreement_rows),
            "agreement_rate": (
                len(agreement_rows) / len(comparable_rows) if comparable_rows else 0.0
            ),
            "unanimous_supported_claims": len(both_supported),
            "unanimous_unsupported_claims": len(both_unsupported),
            "models": summaries,
        },
        "case_judgements": case_judgements,
        "claims": claim_rows,
    }


def select_audit_claims(
    comparison: dict[str, Any], *, agreement_sample: int = 20, seed: int = 0
) -> list[dict[str, Any]]:
    """Select every risky claim plus a deterministic supported-agreement sample."""
    rows = comparison["claims"]
    required = [
        row
        for row in rows
        if not row["comparable"]
        or not row["agreement"]
        or all(value is False for value in row["decisions"].values())
    ]
    unanimous_supported = [
        row
        for row in rows
        if row["comparable"]
        and row["agreement"]
        and all(row["decisions"].values())
    ]
    rng = random.Random(seed)
    sampled = rng.sample(
        unanimous_supported, min(agreement_sample, len(unanimous_supported))
    )
    selected_ids = {row["review_id"] for row in required}
    selected = required + [row for row in sampled if row["review_id"] not in selected_ids]
    return sorted(selected, key=lambda row: (row["case_id"], row["claim_id"]))


def write_comparison(path: str | Path, comparison: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_audit_markdown(path: str | Path, claims: list[dict[str, Any]]) -> None:
    """Write a source-grounded queue for a later manual or agent audit."""
    lines = [
        "# 冻结答案 Claim 证据审查",
        "",
        "每条记录必须根据所列论文原文判断。模型一致只代表意见一致，不是标准答案。",
        "审查结论填写：`支持`、`不支持`或`无法确定`；并写一句理由。",
        "",
        f"待审查 Claim：{len(claims)} 条",
        "",
    ]
    for row in claims:
        lines.extend(
            [
                f"## {row['review_id']}",
                "",
                f"- 问题：{row['question']}",
                f"- Claim：{row['claim']}",
                f"- 模型判断：`{json.dumps(row['decisions'], ensure_ascii=False)}`",
                "- 审查结论：`待填写`",
                "- 审查理由：待填写",
                "",
            ]
        )
        for evidence in row["evidence"]:
            lines.extend(
                [
                    f"### {evidence['citation_id']}｜{evidence.get('paper_id')}｜"
                    f"第 {evidence.get('page_number')} 页｜{evidence.get('section')}",
                    "",
                ]
            )
            lines.extend(f"> {line}" if line else ">" for line in evidence["text"].splitlines())
            lines.append("")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def load_audit_labels(path: str | Path) -> dict[str, Any]:
    """Load traceable review labels for a selected claim audit."""
    audit = json.loads(Path(path).read_text(encoding="utf-8"))
    for field in ("reviewer", "review_type", "limitations"):
        if not isinstance(audit.get(field), str) or not audit[field].strip():
            raise ValueError(f"Audit label file must record {field}")
    labels = audit.get("labels")
    if not isinstance(labels, list) or not labels:
        raise ValueError("Audit label file must contain labels")
    seen: set[str] = set()
    for item in labels:
        if not isinstance(item, dict) or not isinstance(item.get("review_id"), str):
            raise ValueError("Audit label contains an invalid review_id")
        if item["review_id"] in seen:
            raise ValueError(f"Duplicate audit label {item['review_id']}")
        seen.add(item["review_id"])
        if item.get("label") not in {"supported", "unsupported", "uncertain"}:
            raise ValueError(f"Invalid audit label for {item['review_id']}")
        if not isinstance(item.get("reason"), str) or not item["reason"].strip():
            raise ValueError(f"Audit label {item['review_id']} requires a reason")
    return audit


def score_audit_labels(
    comparison: dict[str, Any], audit: dict[str, Any]
) -> dict[str, Any]:
    """Score each automatic judge against explicitly attributed audit labels."""
    claims = {row["review_id"]: row for row in comparison["claims"]}
    models = comparison["judge_models"]
    metrics = {
        model: {
            "model": model,
            "comparable_labels": 0,
            "correct": 0,
            "false_accepts": 0,
            "false_rejects": 0,
        }
        for model in models
    }
    supported = 0
    unsupported = 0
    uncertain = 0
    scored_labels = []
    for label in audit["labels"]:
        row = claims.get(label["review_id"])
        if row is None:
            raise ValueError(f"Unknown audited claim {label['review_id']}")
        expected = label["label"]
        if expected == "supported":
            supported += 1
        elif expected == "unsupported":
            unsupported += 1
        else:
            uncertain += 1
        scored_labels.append(
            {
                **label,
                "claim": row["claim"],
                "decisions": row["decisions"],
            }
        )
        if expected == "uncertain":
            continue
        expected_bool = expected == "supported"
        for model in models:
            decision = row["decisions"].get(model)
            if decision is None:
                continue
            item = metrics[model]
            item["comparable_labels"] += 1
            if decision == expected_bool:
                item["correct"] += 1
            elif decision:
                item["false_accepts"] += 1
            else:
                item["false_rejects"] += 1
    model_metrics = []
    for model in models:
        item = metrics[model]
        model_metrics.append(
            {
                **item,
                "accuracy": (
                    item["correct"] / item["comparable_labels"]
                    if item["comparable_labels"]
                    else 0.0
                ),
            }
        )
    determinate = supported + unsupported
    return {
        "reviewer": audit.get("reviewer"),
        "review_type": audit.get("review_type"),
        "limitations": audit.get("limitations"),
        "summary": {
            "labeled_claims": len(audit["labels"]),
            "determinate_claims": determinate,
            "supported_claims": supported,
            "unsupported_claims": unsupported,
            "uncertain_claims": uncertain,
            "audited_claim_support_rate": (
                supported / determinate if determinate else 0.0
            ),
            "models": model_metrics,
        },
        "labels": scored_labels,
    }
