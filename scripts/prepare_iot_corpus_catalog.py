"""Build a reviewed ScholarRAG catalog from the audited IoT discovery corpus."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path

import pymupdf

from src.paper_assistant.catalog import PaperCatalog

CATALOG_FIELDS = (
    "paper_id",
    "pdf_file",
    "canonical_title",
    "title_zh",
    "authors",
    "year",
    "venue",
    "language",
    "tags",
    "method_summary",
    "datasets",
    "memory_cues",
    "duplicate_group",
    "read_date",
    "rating",
    "notes_source",
    "corpus_tier",
    "is_user_read",
    "doi",
    "openalex_id",
    "topic_bucket",
    "audit_status",
)

TOPIC_TAGS = {
    "core_iot": "物联网;Internet of Things;IoT",
    "industrial_iot": "工业物联网;Industrial Internet of Things;IIoT",
    "iot_security": "物联网安全;IoT security;privacy;intrusion detection",
    "edge_iot": "边缘计算;雾计算;edge computing;fog computing;IoT",
    "sensor_networks": "无线传感器网络;wireless sensor networks;WSN;IoT",
    "smart_cities": "智慧城市;smart cities;urban IoT",
    "iot_healthcare": "医疗物联网;Internet of Medical Things;IoMT;healthcare IoT",
    "smart_agriculture": "智慧农业;smart agriculture;agricultural IoT",
    "federated_iot": "联邦学习;federated learning;IoT",
    "mobile_crowdsensing": "移动群智感知;mobile crowdsensing;participatory sensing",
}


def extract_pdf_abstract(path: Path, *, max_chars: int = 2_500) -> str:
    """Extract an opening abstract without invoking an LLM."""
    with pymupdf.open(path) as document:
        opening = "\n".join(
            document[index].get_text("text")
            for index in range(min(4, document.page_count))
        )
    opening = re.sub(r"(?<=\w)-\s*\n\s*(?=\w)", "", opening)
    opening = re.sub(r"[ \t]+", " ", opening)
    match = re.search(
        r"(?is)\babstract\b\s*[-—:]*\s*(.+?)"
        r"(?=\b(?:index terms?|key\s*words?)\b\s*[-—:]|"
        r"\n\s*(?:I\.?|1\.?)\s+INTRODUCTION\b)",
        opening,
    )
    if not match:
        return ""
    return re.sub(r"\s+", " ", match.group(1)).strip()[:max_chars]


def final_decision(audit_status: str, manual_decision: str = "") -> str:
    """Keep risk records pending unless a separate review explicitly decides."""
    if manual_decision in {"admit", "exclude"}:
        return manual_decision
    return "admit" if audit_status == "pass" else "pending"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidates",
        type=Path,
        default=Path("data/papers/discovery/iot_candidates.csv"),
    )
    parser.add_argument(
        "--audit",
        type=Path,
        default=Path("data/papers/corpus_iot_1000/audit/corpus_audit.csv"),
    )
    parser.add_argument(
        "--decisions",
        type=Path,
        default=Path("data/papers/corpus_iot_1000/audit/manual_review_decisions.csv"),
    )
    parser.add_argument(
        "--pdf-dir",
        type=Path,
        default=Path("data/papers/corpus_iot_1000/pdfs"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/papers/corpus_iot_1000/catalog"),
    )
    return parser.parse_args()


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    args = parse_args()
    candidates = {row["paper_id"]: row for row in _read_rows(args.candidates)}
    audits = _read_rows(args.audit)
    decisions = {row["paper_id"]: row for row in _read_rows(args.decisions)}

    args.output_dir.mkdir(parents=True, exist_ok=True)
    admission_path = args.output_dir / "admission_manifest.csv"
    admission_fields = (
        "paper_id",
        "machine_status",
        "machine_flags",
        "manual_decision",
        "final_decision",
        "review_reason",
        "reviewer",
        "reviewed_on",
    )
    admissions: list[dict[str, str]] = []
    catalog_rows: list[dict[str, str]] = []
    missing_abstracts = 0

    for audit in audits:
        paper_id = audit["paper_id"]
        candidate = candidates[paper_id]
        review = decisions.get(paper_id, {})
        decision = final_decision(audit["status"], review.get("decision", ""))
        admissions.append(
            {
                "paper_id": paper_id,
                "machine_status": audit["status"],
                "machine_flags": audit["flags"],
                "manual_decision": review.get("decision", ""),
                "final_decision": decision,
                "review_reason": review.get("reason", ""),
                "reviewer": review.get("reviewer", ""),
                "reviewed_on": review.get("reviewed_on", ""),
            }
        )
        if decision != "admit":
            continue
        pdf_file = f"{paper_id}.pdf"
        abstract = extract_pdf_abstract(args.pdf_dir / pdf_file)
        if not abstract:
            missing_abstracts += 1
        primary_topic = candidate.get("primary_topic", "").strip()
        tags = TOPIC_TAGS.get(candidate["topic_bucket"], "物联网;IoT")
        if primary_topic and primary_topic.casefold() not in tags.casefold():
            tags = f"{tags};{primary_topic}"
        catalog_rows.append(
            {
                "paper_id": paper_id,
                "pdf_file": pdf_file,
                "canonical_title": candidate["title"],
                "title_zh": "",
                "authors": candidate["authors"],
                "year": candidate["year"],
                "venue": candidate["journal"],
                "language": "en",
                "tags": tags,
                "method_summary": abstract,
                "datasets": "",
                "memory_cues": "",
                "duplicate_group": "",
                "read_date": "",
                "rating": "",
                "notes_source": (
                    "OpenAlex metadata and rule-extracted PDF abstract; discovery corpus, "
                    "not personally read"
                ),
                "corpus_tier": "discovery",
                "is_user_read": "false",
                "doi": candidate["doi"],
                "openalex_id": candidate["openalex_id"],
                "topic_bucket": candidate["topic_bucket"],
                "audit_status": audit["status"],
            }
        )

    with admission_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=admission_fields)
        writer.writeheader()
        writer.writerows(admissions)

    catalog_path = args.output_dir / "iot_paper_catalog.csv"
    with catalog_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CATALOG_FIELDS)
        writer.writeheader()
        writer.writerows(catalog_rows)

    # Re-read through the production loader so the generated catalog cannot drift
    # from the data contract used by retrieval and chunk indexing.
    validated = PaperCatalog.from_csv(catalog_path)
    summary = {
        "audit_records": len(audits),
        "admitted": len(catalog_rows),
        "excluded": sum(row["final_decision"] == "exclude" for row in admissions),
        "pending": sum(row["final_decision"] == "pending" for row in admissions),
        "missing_abstracts": missing_abstracts,
        "validated_profiles": len(validated),
        "topic_counts": dict(Counter(row["topic_bucket"] for row in catalog_rows)),
        "catalog": str(catalog_path),
        "admission_manifest": str(admission_path),
    }
    summary_path = args.output_dir / "catalog_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
