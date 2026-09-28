"""Audit downloaded IoT PDFs before they are admitted to the formal RAG corpus."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import pymupdf

CORE_PATTERNS = (
    r"\binternet of things\b",
    r"\biot\b",
    r"\bindustrial internet\b",
    r"\biiot\b",
    r"\bmobile crowd\s*sensing\b",
)
BUCKET_PATTERNS = {
    "core_iot": (r"\binternet of things\b", r"\biot\b"),
    "industrial_iot": (r"\bindustr(?:y|ial)\b", r"\biiot\b"),
    "iot_security": (r"\bsecurity\b", r"\bprivacy\b", r"\battack\b"),
    "edge_iot": (r"\bedge computing\b", r"\bfog computing\b", r"\bedge intelligence\b"),
    "sensor_networks": (r"\bwireless sensor", r"\bsensor network", r"\bwsn\b"),
    "smart_cities": (r"\bsmart cit", r"\burban\b", r"\bsmart transportation\b"),
    "iot_healthcare": (r"\bhealth", r"\bmedical\b", r"\bpatient\b"),
    "smart_agriculture": (r"\bagricultur", r"\bfarm", r"\bcrop\b"),
    "federated_iot": (r"\bfederated learning\b", r"\bfederated\b"),
    "mobile_crowdsensing": (r"\bcrowd\s*sensing\b", r"\bparticipatory sensing\b"),
}
RISK_TITLE_PATTERNS = (
    r"^retraction\b",
    r"^withdrawn\b",
    r"^withdrawal\b",
    r"^expression of concern\b",
    r"^correction(?:\s+to)?\b",
    r"^corrigendum\b",
    r"^erratum\b",
)
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "based", "by", "for", "from",
    "in", "into", "of", "on", "the", "to", "toward", "towards", "using",
    "via", "with",
}


@dataclass(frozen=True)
class AuditRow:
    paper_id: str
    title: str
    topic_bucket: str
    journal: str
    year: str
    doi: str
    pdf_file: str
    status: str
    flags: str
    page_count: int
    file_size_bytes: int
    text_char_count: int
    text_page_ratio: float
    title_token_coverage: float
    core_iot_hit: bool
    bucket_topic_hit: bool
    is_encrypted: bool
    is_repaired: bool
    sha256: str
    first_page_excerpt: str


def _normalized(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip().casefold()


def title_token_coverage(title: str, text: str) -> float:
    """Measure whether meaningful title tokens occur in the PDF opening pages."""
    tokens = {
        token
        for token in re.findall(r"[a-z0-9]+", title.casefold())
        if len(token) > 2 and token not in STOPWORDS
    }
    if not tokens:
        return 0.0
    text_tokens = set(re.findall(r"[a-z0-9]+", text.casefold()))
    return len(tokens & text_tokens) / len(tokens)


def _has_pattern(text: str, patterns: tuple[str, ...]) -> bool:
    return any(re.search(pattern, text, flags=re.I) for pattern in patterns)


def classify_audit(
    *,
    title: str,
    bucket: str,
    page_count: int,
    text_char_count: int,
    text_page_ratio: float,
    title_coverage: float,
    opening_text: str,
    encrypted: bool,
    repaired: bool,
) -> tuple[str, tuple[str, ...], bool, bool]:
    """Apply transparent admission rules and return status plus diagnostic flags."""
    normalized_title = _normalized(title)
    searchable = f"{normalized_title} {_normalized(opening_text)}"
    core_hit = _has_pattern(searchable, CORE_PATTERNS)
    bucket_hit = _has_pattern(searchable, BUCKET_PATTERNS.get(bucket, ()))
    flags: list[str] = []

    if encrypted:
        flags.append("encrypted")
    if repaired:
        flags.append("repaired_pdf")
    if page_count < 3:
        flags.append("too_few_pages")
    if text_char_count < 500:
        flags.append("no_extractable_text")
    elif text_char_count < 2_000 or text_page_ratio < 0.5:
        flags.append("low_extractable_text")
    if title_coverage < 0.35:
        flags.append("title_mismatch")
    if not core_hit and not bucket_hit:
        flags.append("low_iot_relevance")
    if _has_pattern(normalized_title, RISK_TITLE_PATTERNS):
        flags.append("correction_or_retraction_title")

    reject_flags = {
        "encrypted",
        "too_few_pages",
        "no_extractable_text",
        "correction_or_retraction_title",
    }
    if reject_flags.intersection(flags):
        status = "reject"
    elif flags:
        status = "needs_review"
    else:
        status = "pass"
    return status, tuple(flags), core_hit, bucket_hit


def audit_pdf(path: Path, metadata: dict[str, str]) -> AuditRow:
    """Inspect one local PDF for parseability, textual completeness, and relevance."""
    file_bytes = path.read_bytes()
    digest = hashlib.sha256(file_bytes).hexdigest()
    title = metadata["title"]
    bucket = metadata["topic_bucket"]
    try:
        with pymupdf.open(path) as document:
            encrypted = bool(document.needs_pass)
            repaired = bool(document.is_repaired)
            page_count = document.page_count
            page_texts: list[str] = []
            text_pages = 0
            for page in document:
                text = page.get_text("text")
                page_texts.append(text)
                if len(_normalized(text)) >= 80:
                    text_pages += 1
    except Exception as error:
        return AuditRow(
            paper_id=metadata["paper_id"],
            title=title,
            topic_bucket=bucket,
            journal=metadata["journal"],
            year=metadata["year"],
            doi=metadata["doi"],
            pdf_file=str(path),
            status="reject",
            flags=f"unreadable_pdf:{type(error).__name__}",
            page_count=0,
            file_size_bytes=len(file_bytes),
            text_char_count=0,
            text_page_ratio=0.0,
            title_token_coverage=0.0,
            core_iot_hit=False,
            bucket_topic_hit=False,
            is_encrypted=False,
            is_repaired=False,
            sha256=digest,
            first_page_excerpt="",
        )

    text_char_count = sum(len(_normalized(text)) for text in page_texts)
    text_page_ratio = text_pages / page_count if page_count else 0.0
    opening_text = "\n".join(page_texts[: min(4, page_count)])
    coverage = title_token_coverage(title, opening_text)
    status, flags, core_hit, bucket_hit = classify_audit(
        title=title,
        bucket=bucket,
        page_count=page_count,
        text_char_count=text_char_count,
        text_page_ratio=text_page_ratio,
        title_coverage=coverage,
        opening_text=opening_text,
        encrypted=encrypted,
        repaired=repaired,
    )
    return AuditRow(
        paper_id=metadata["paper_id"],
        title=title,
        topic_bucket=bucket,
        journal=metadata["journal"],
        year=metadata["year"],
        doi=metadata["doi"],
        pdf_file=str(path),
        status=status,
        flags=";".join(flags),
        page_count=page_count,
        file_size_bytes=len(file_bytes),
        text_char_count=text_char_count,
        text_page_ratio=round(text_page_ratio, 4),
        title_token_coverage=round(coverage, 4),
        core_iot_hit=core_hit,
        bucket_topic_hit=bucket_hit,
        is_encrypted=encrypted,
        is_repaired=repaired,
        sha256=digest,
        first_page_excerpt=_normalized(page_texts[0] if page_texts else "")[:500],
    )


def select_review_sample(rows: list[AuditRow], per_bucket: int = 5) -> list[dict[str, str]]:
    """Select risk-first and deterministic clean examples from every topic bucket."""
    grouped: dict[str, list[AuditRow]] = defaultdict(list)
    for row in rows:
        grouped[row.topic_bucket].append(row)

    sample: list[dict[str, str]] = []
    for bucket, bucket_rows in sorted(grouped.items()):
        risky = sorted(
            (row for row in bucket_rows if row.status != "pass"),
            key=lambda row: (row.status != "reject", row.title_token_coverage, row.paper_id),
        )
        clean = sorted(
            (row for row in bucket_rows if row.status == "pass"),
            key=lambda row: hashlib.sha256(row.paper_id.encode()).hexdigest(),
        )
        picks = risky[:2]
        picks.extend(clean[: max(0, per_bucket - len(picks))])
        for row in picks[:per_bucket]:
            record = {key: str(value) for key, value in asdict(row).items()}
            record["sample_reason"] = "risk" if row.status != "pass" else "representative"
            sample.append(record)
    return sample


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--catalog",
        type=Path,
        default=Path("data/papers/discovery/iot_candidates.csv"),
    )
    parser.add_argument(
        "--pdf-dir",
        type=Path,
        default=Path("data/papers/corpus_iot_1000/pdfs"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/papers/corpus_iot_1000/audit"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.catalog.open("r", encoding="utf-8-sig", newline="") as handle:
        catalog = {row["paper_id"]: row for row in csv.DictReader(handle)}

    pdfs = sorted(args.pdf_dir.glob("*.pdf"))
    missing_metadata = [path.stem for path in pdfs if path.stem not in catalog]
    if missing_metadata:
        raise ValueError(f"PDFs missing catalog metadata: {', '.join(missing_metadata[:5])}")

    rows: list[AuditRow] = []
    for index, path in enumerate(pdfs, start=1):
        rows.append(audit_pdf(path, catalog[path.stem]))
        if index % 25 == 0 or index == len(pdfs):
            print(f"audited={index}/{len(pdfs)}", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    audit_path = args.output_dir / "corpus_audit.csv"
    fields = tuple(AuditRow.__dataclass_fields__)
    with audit_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)

    sample = select_review_sample(rows)
    sample_path = args.output_dir / "review_sample.csv"
    sample_fields = (*fields, "sample_reason")
    with sample_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sample_fields)
        writer.writeheader()
        writer.writerows(sample)

    status_counts = Counter(row.status for row in rows)
    flag_counts = Counter(
        flag for row in rows for flag in row.flags.split(";") if flag
    )
    summary = {
        "pdf_count": len(rows),
        "status_counts": dict(status_counts),
        "flag_counts": dict(flag_counts),
        "topic_counts": dict(Counter(row.topic_bucket for row in rows)),
        "page_count": sum(row.page_count for row in rows),
        "text_char_count": sum(row.text_char_count for row in rows),
        "review_sample_count": len(sample),
        "audit_csv": str(audit_path),
        "review_sample_csv": str(sample_path),
    }
    summary_path = args.output_dir / "audit_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
