"""Create evidence-backed Paper Profile drafts from local PDFs."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pymupdf as fitz

from src.paper_assistant.inventory import build_paper_inventory, sha256_file

DOI_PATTERN = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+", re.IGNORECASE)
YEAR_PATTERN = re.compile(r"\b(?:19|20)\d{2}\b")
ABSTRACT_PATTERN = re.compile(
    r"(?:^|\n)\s*(?:abstract|摘要)\s*[:：—-]?\s*(.+?)"
    r"(?=\n\s*(?:keywords?|index\s+terms|关键词|(?:1|I)\.?\s+introduction|引言)\b|\Z)",
    re.IGNORECASE | re.DOTALL,
)
GENERIC_METADATA = {
    "untitled",
    "microsoft word",
    "document",
    "pdf",
    "acrobat distiller",
}


@dataclass(frozen=True)
class FieldEvidence:
    """A field value together with the PDF evidence used to derive it."""

    value: str
    source: str
    page: int | None
    source_text: str
    confidence: str


@dataclass(frozen=True)
class ExtractedPaperFacts:
    """Deterministic facts extracted from one PDF."""

    page_count: int
    title: FieldEvidence | None
    authors: FieldEvidence | None
    year: FieldEvidence | None
    abstract: FieldEvidence | None
    doi: FieldEvidence | None
    language: FieldEvidence | None
    warnings: tuple[str, ...]


def _clean_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _join_title_parts(parts: list[str]) -> str:
    result = ""
    for part in parts:
        if not result:
            result = part
        elif re.search(r"[\u4e00-\u9fff]$", result) and re.match(
            r"^[\u4e00-\u9fff]", part
        ):
            result += part
        else:
            result += f" {part}"
    return result


def _metadata_value(metadata: dict[str, Any], key: str) -> str:
    value = _clean_text(str(metadata.get(key) or ""))
    if not value or value.casefold() in GENERIC_METADATA:
        return ""
    return value


def _page_blocks(page: fitz.Page) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    page_dict = page.get_text("dict")
    for block in page_dict.get("blocks", []):
        if block.get("type") != 0:
            continue
        lines: list[str] = []
        sizes: list[float] = []
        for line in block.get("lines", []):
            line_text = "".join(span.get("text", "") for span in line.get("spans", []))
            if cleaned := _clean_text(line_text):
                lines.append(cleaned)
            sizes.extend(float(span.get("size", 0)) for span in line.get("spans", []))
        text = _clean_text(" ".join(lines))
        if text:
            blocks.append(
                {
                    "text": text,
                    "max_size": max(sizes, default=0),
                    "bbox": tuple(block.get("bbox", (0, 0, 0, 0))),
                }
            )
    return blocks


def _title_from_layout(page: fitz.Page) -> tuple[FieldEvidence | None, float]:
    page_height = page.rect.height
    candidates = [
        block
        for block in _page_blocks(page)
        if 2 <= len(block["text"]) <= 350
        and block["bbox"][1] <= page_height * 0.6
        and not DOI_PATTERN.search(block["text"])
        and block["text"].casefold() not in {"abstract", "摘要"}
    ]
    if not candidates:
        return None, 0
    largest_size = max(block["max_size"] for block in candidates)
    largest_blocks = sorted(
        (
            block
            for block in candidates
            if abs(block["max_size"] - largest_size) <= 0.5
        ),
        key=lambda block: block["bbox"][1],
    )
    merged = [largest_blocks[0]]
    for block in largest_blocks[1:]:
        vertical_gap = block["bbox"][1] - merged[-1]["bbox"][3]
        if vertical_gap > largest_size:
            break
        merged.append(block)
    title_text = _join_title_parts([block["text"] for block in merged])
    title_bottom = max(block["bbox"][3] for block in merged)
    return (
        FieldEvidence(
            value=title_text,
            source="page_layout",
            page=1,
            source_text=title_text,
            confidence="medium",
        ),
        float(title_bottom),
    )


def _authors_from_layout(page: fitz.Page, title_bottom: float) -> FieldEvidence | None:
    if not title_bottom:
        return None
    page_height = page.rect.height
    disallowed = (
        "abstract",
        "摘要",
        "university",
        "institute",
        "department",
        "laboratory",
        "http",
        "doi",
        "received",
    )
    candidates = []
    for block in sorted(_page_blocks(page), key=lambda item: item["bbox"][1]):
        text = block["text"]
        y0 = block["bbox"][1]
        if y0 <= title_bottom or y0 > page_height * 0.62:
            continue
        lowered = text.casefold()
        if any(token in lowered for token in disallowed) or len(text) > 240:
            continue
        candidates.append(text)
        break
    if not candidates:
        return None
    value = _clean_text("; ".join(candidates))
    return FieldEvidence(
        value=value,
        source="page_layout",
        page=1,
        source_text=value,
        confidence="low",
    )


def _abstract_from_pages(page_texts: list[str]) -> FieldEvidence | None:
    for page_number, text in enumerate(page_texts, start=1):
        match = ABSTRACT_PATTERN.search(text)
        if not match:
            continue
        value = _clean_text(match.group(1))[:2500]
        if value:
            return FieldEvidence(
                value=value,
                source="page_text",
                page=page_number,
                source_text=value,
                confidence="high",
            )
    return None


def _doi_from_pages(page_texts: list[str]) -> FieldEvidence | None:
    for page_number, text in enumerate(page_texts[:3], start=1):
        match = DOI_PATTERN.search(text)
        if match:
            value = match.group(0).rstrip(".,;)")
            return FieldEvidence(
                value=value,
                source="page_text",
                page=page_number,
                source_text=_clean_text(match.group(0)),
                confidence="high",
            )
    return None


def _year_from_metadata_or_pages(
    metadata: dict[str, Any], page_texts: list[str]
) -> FieldEvidence | None:
    for key in ("creationDate", "modDate"):
        raw = _metadata_value(metadata, key)
        match = re.search(r"D:((?:19|20)\d{2})", raw) or YEAR_PATTERN.search(raw)
        if match:
            return FieldEvidence(
                value=match.group(1) if match.lastindex else match.group(0),
                source=f"pdf_metadata.{key}",
                page=None,
                source_text=raw,
                confidence="low",
            )
    for page_number, text in enumerate(page_texts[:2], start=1):
        contextual = re.search(
            r"(?:published|accepted|copyright|©|出版|发表)[^\n]{0,80}?((?:19|20)\d{2})",
            text,
            re.IGNORECASE,
        )
        if contextual:
            return FieldEvidence(
                value=contextual.group(1),
                source="page_text",
                page=page_number,
                source_text=_clean_text(contextual.group(0)),
                confidence="medium",
            )
    return None


def _detect_language(page_texts: list[str]) -> FieldEvidence | None:
    sample = " ".join(page_texts)[:12000]
    if not sample.strip():
        return None
    chinese = len(re.findall(r"[\u4e00-\u9fff]", sample))
    latin = len(re.findall(r"[A-Za-z]", sample))
    value = "zh" if chinese > latin * 0.25 else "en"
    return FieldEvidence(
        value=value,
        source="text_language_detection",
        page=None,
        source_text=f"chinese_chars={chinese}; latin_chars={latin}",
        confidence="high" if max(chinese, latin) >= 100 else "low",
    )


def extract_pdf_facts(pdf_path: str | Path, max_pages: int = 5) -> ExtractedPaperFacts:
    """Extract factual fields and their evidence without using an LLM."""
    path = Path(pdf_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"PDF does not exist: {path}")
    if path.suffix.lower() != ".pdf":
        raise ValueError(f"File is not a PDF: {path.name}")
    if max_pages < 1:
        raise ValueError("max_pages must be at least 1")

    warnings: list[str] = []
    try:
        with fitz.open(path) as document:
            if document.needs_pass:
                raise ValueError("Encrypted PDF requires a password")
            page_count = document.page_count
            metadata = dict(document.metadata or {})
            pages = [document.load_page(index) for index in range(min(page_count, max_pages))]
            page_texts = [page.get_text("text") for page in pages]

            metadata_title = _metadata_value(metadata, "title")
            if metadata_title:
                title = FieldEvidence(
                    value=metadata_title,
                    source="pdf_metadata.title",
                    page=None,
                    source_text=metadata_title,
                    confidence="high",
                )
                _, title_bottom = _title_from_layout(pages[0]) if pages else (None, 0)
            elif pages:
                title, title_bottom = _title_from_layout(pages[0])
            else:
                title, title_bottom = None, 0

            metadata_authors = _metadata_value(metadata, "author")
            if metadata_authors:
                authors = FieldEvidence(
                    value=metadata_authors,
                    source="pdf_metadata.author",
                    page=None,
                    source_text=metadata_authors,
                    confidence="high",
                )
            elif pages:
                authors = _authors_from_layout(pages[0], title_bottom)
            else:
                authors = None

            if not any(text.strip() for text in page_texts):
                warnings.append("No selectable text was found; the PDF may require OCR")
            if page_count > max_pages:
                warnings.append(
                    f"Only the first {max_pages} of {page_count} pages were inspected"
                )
            return ExtractedPaperFacts(
                page_count=page_count,
                title=title,
                authors=authors,
                year=_year_from_metadata_or_pages(metadata, page_texts),
                abstract=_abstract_from_pages(page_texts),
                doi=_doi_from_pages(page_texts),
                language=_detect_language(page_texts),
                warnings=tuple(warnings),
            )
    except fitz.FileDataError as exc:
        raise ValueError(f"Invalid or damaged PDF: {path.name}") from exc


def _evidence_value(
    evidence: FieldEvidence | None, *, include_low_confidence: bool = False
) -> str:
    if not evidence or (evidence.confidence == "low" and not include_low_confidence):
        return ""
    return evidence.value


def prepare_profile_draft(
    pdf_path: str | Path,
    inbox_path: str | Path,
    catalog_path: str | Path,
    max_pages: int = 5,
) -> dict[str, Any]:
    """Build a local review draft without modifying the catalog."""
    pdf = Path(pdf_path).resolve()
    inbox = Path(inbox_path).resolve()
    try:
        relative_pdf = pdf.relative_to(inbox).as_posix()
    except ValueError as exc:
        raise ValueError("PDF must be located inside the configured inbox") from exc

    inventory = build_paper_inventory(inbox, catalog_path)
    inventory_file = next(
        (item for item in inventory.files if item.path.casefold() == relative_pdf.casefold()),
        None,
    )
    if inventory_file is None:
        raise ValueError("PDF is not visible in the configured inbox")

    duplicate_files = next(
        (
            tuple(file for file in group.files if file.casefold() != relative_pdf.casefold())
            for group in inventory.exact_duplicate_groups
            if relative_pdf in group.files
        ),
        (),
    )
    facts = extract_pdf_facts(pdf, max_pages=max_pages)
    fields = {
        "canonical_title": facts.title,
        "authors": facts.authors,
        "year": facts.year,
        "abstract": facts.abstract,
        "doi": facts.doi,
        "language": facts.language,
    }
    missing_fields = tuple(name for name, evidence in fields.items() if evidence is None)
    review_required_fields = tuple(
        name
        for name, evidence in fields.items()
        if evidence is not None and evidence.confidence == "low"
    )
    status = (
        "already_registered"
        if inventory_file.paper_ids
        else "duplicate_review_required"
        if duplicate_files
        else "needs_review"
    )
    return {
        "schema_version": 1,
        "status": status,
        "file": {
            "pdf_file": relative_pdf,
            "size_bytes": pdf.stat().st_size,
            "sha256": sha256_file(pdf),
            "page_count": facts.page_count,
        },
        "catalog_matches": inventory_file.paper_ids,
        "exact_duplicate_files": duplicate_files,
        "extracted_fields": {
            name: asdict(evidence) if evidence else None
            for name, evidence in fields.items()
        },
        "proposed_profile": {
            "paper_id": "",
            "pdf_file": relative_pdf,
            "canonical_title": _evidence_value(facts.title),
            "title_zh": "",
            "authors": _evidence_value(facts.authors),
            "year": _evidence_value(facts.year),
            "venue": "",
            "language": _evidence_value(facts.language),
            "tags": "",
            "method_summary": "",
            "datasets": "",
            "memory_cues": "",
            "duplicate_group": "",
        },
        "missing_fields": missing_fields,
        "review_required_fields": review_required_fields,
        "warnings": facts.warnings,
        "review_notes": "Fact fields require human verification; semantic fields remain blank.",
    }


def write_profile_draft(
    draft: dict[str, Any], output_path: str | Path, *, overwrite: bool = False
) -> Path:
    """Write a local draft atomically, refusing accidental overwrite by default."""
    output = Path(output_path).resolve()
    if output.exists() and not overwrite:
        raise FileExistsError(f"Draft already exists: {output.name}; use --force to replace it")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_text(
        json.dumps(draft, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(output)
    return output
