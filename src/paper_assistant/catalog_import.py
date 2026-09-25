"""Plan and apply controlled Paper Profile imports."""

from __future__ import annotations

import csv
import json
import re
import shutil
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

from src.paper_assistant.catalog import PaperCatalog
from src.paper_assistant.inventory import build_paper_inventory, sha256_file

PAPER_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]{0,79}$")
REQUIRED_PROFILE_FIELDS = (
    "paper_id",
    "pdf_file",
    "canonical_title",
    "language",
    "tags",
    "method_summary",
    "memory_cues",
)


def _clean_cell(value: Any) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        raise ValueError("Catalog profile values must be strings")
    return re.sub(r"\s+", " ", value).strip()


def _relative_pdf_path(value: str) -> str:
    normalized = PurePosixPath(value.strip().replace("\\", "/"))
    if normalized.is_absolute() or ".." in normalized.parts:
        raise ValueError("pdf_file must be a safe path relative to the paper inbox")
    result = normalized.as_posix()
    while result.startswith("./"):
        result = result[2:]
    if not result or Path(result).suffix.lower() != ".pdf":
        raise ValueError("pdf_file must identify a PDF inside the paper inbox")
    return result


def _read_catalog(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or ())
        missing = PaperCatalog.REQUIRED_COLUMNS - set(fields)
        if missing:
            raise ValueError(
                f"Catalog is missing required columns: {', '.join(sorted(missing))}"
            )
        return fields, [dict(row) for row in reader]


@dataclass(frozen=True)
class CatalogImportPlan:
    """A non-mutating review of one proposed catalog append."""

    ready: bool
    action: str
    paper_id: str
    pdf_file: str
    profile: dict[str, str]
    missing_fields: tuple[str, ...]
    errors: tuple[str, ...]
    warnings: tuple[str, ...]
    catalog_row_count_before: int
    catalog_sha256: str
    pdf_sha256: str
    pdf_size_bytes: int | None

    def summary_dict(self) -> dict[str, Any]:
        summary = asdict(self)
        summary.pop("catalog_sha256")
        summary.pop("pdf_sha256")
        summary.pop("pdf_size_bytes")
        return summary


@dataclass(frozen=True)
class CatalogImportResult:
    """Summary of a completed atomic append."""

    status: str
    paper_id: str
    pdf_file: str
    catalog_row_count: int
    backup_file: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def plan_catalog_import(
    draft_path: str | Path,
    inbox_path: str | Path,
    catalog_path: str | Path,
) -> CatalogImportPlan:
    """Validate a reviewed draft without modifying the catalog."""
    draft_file = Path(draft_path).resolve()
    inbox = Path(inbox_path).resolve()
    catalog_file = Path(catalog_path).resolve()
    draft = json.loads(draft_file.read_text(encoding="utf-8-sig"))
    if not isinstance(draft, dict):
        raise ValueError("Draft root must be a JSON object")
    fields, rows = _read_catalog(catalog_file)
    del fields

    errors: list[str] = []
    warnings: list[str] = []
    if draft.get("schema_version") != 1:
        errors.append("Unsupported or missing draft schema_version")
    if draft.get("status") != "needs_review":
        errors.append("Only a needs_review draft can be imported as a new paper")
    if draft.get("catalog_matches"):
        errors.append("The draft PDF is already registered in the catalog")
    if draft.get("exact_duplicate_files"):
        errors.append("The draft PDF is an exact duplicate of another inbox file")

    proposed = draft.get("proposed_profile")
    if not isinstance(proposed, dict):
        proposed = {}
        errors.append("Draft proposed_profile must be an object")
    try:
        profile = {key: _clean_cell(proposed.get(key, "")) for key in proposed}
    except ValueError as error:
        profile = {}
        errors.append(str(error))

    paper_id = profile.get("paper_id", "")
    if paper_id and not PAPER_ID_PATTERN.fullmatch(paper_id):
        errors.append(
            "paper_id must use lowercase letters, digits, dots, underscores, or hyphens"
        )

    file_section = draft.get("file")
    if not isinstance(file_section, dict):
        file_section = {}
        errors.append("Draft file metadata must be an object")
    draft_pdf = str(file_section.get("pdf_file", ""))
    path_is_safe = True
    try:
        pdf_file = _relative_pdf_path(profile.get("pdf_file", "") or draft_pdf)
        draft_pdf = _relative_pdf_path(draft_pdf)
        if pdf_file.casefold() != draft_pdf.casefold():
            errors.append("proposed_profile.pdf_file does not match the drafted PDF")
    except ValueError as error:
        path_is_safe = False
        pdf_file = profile.get("pdf_file", "") or draft_pdf
        errors.append(str(error))

    missing_fields = tuple(
        field for field in REQUIRED_PROFILE_FIELDS if not profile.get(field, "")
    )
    if missing_fields:
        errors.append(
            f"Complete the required profile fields: {', '.join(missing_fields)}"
        )

    existing_ids = {str(row.get("paper_id", "")).strip() for row in rows}
    existing_paths = {
        str(row.get("pdf_file", "")).strip().replace("\\", "/").casefold()
        for row in rows
    }
    existing_titles = {
        str(row.get("canonical_title", "")).strip().casefold(): str(
            row.get("paper_id", "")
        ).strip()
        for row in rows
        if str(row.get("canonical_title", "")).strip()
    }
    if paper_id in existing_ids:
        errors.append(f"paper_id already exists: {paper_id}")
    if pdf_file.replace("\\", "/").casefold() in existing_paths:
        errors.append(f"pdf_file is already registered: {pdf_file}")
    title_key = profile.get("canonical_title", "").casefold()
    if title_key and title_key in existing_titles:
        errors.append(
            "canonical_title already belongs to paper_id "
            f"{existing_titles[title_key]}"
        )

    expected_hash = str(file_section.get("sha256", "")).lower()
    expected_size = file_section.get("size_bytes")
    if path_is_safe:
        pdf_path = (inbox / Path(*PurePosixPath(pdf_file).parts)).resolve()
        try:
            pdf_path.relative_to(inbox)
        except ValueError:
            errors.append("Resolved PDF path escapes the configured inbox")
        else:
            if not pdf_path.is_file():
                errors.append(f"Drafted PDF is missing from the inbox: {pdf_file}")
            else:
                if not re.fullmatch(r"[0-9a-f]{64}", expected_hash):
                    errors.append("Draft is missing a valid PDF SHA-256")
                elif sha256_file(pdf_path) != expected_hash:
                    errors.append("PDF content changed after the draft was generated")
                if (
                    isinstance(expected_size, int)
                    and pdf_path.stat().st_size != expected_size
                ):
                    errors.append("PDF size changed after the draft was generated")

        inventory = build_paper_inventory(inbox, catalog_file)
        inventory_file = next(
            (
                item
                for item in inventory.files
                if item.path.casefold() == pdf_file.casefold()
            ),
            None,
        )
        if inventory_file and inventory_file.paper_ids:
            errors.append("The current inbox inventory says this PDF is already registered")
        current_duplicates = next(
            (
                group.files
                for group in inventory.exact_duplicate_groups
                if any(path.casefold() == pdf_file.casefold() for path in group.files)
            ),
            (),
        )
        if current_duplicates:
            errors.append(
                "The current inbox contains byte-identical PDF files: "
                + ", ".join(current_duplicates)
            )

    for optional in ("authors", "year", "venue", "datasets"):
        if not profile.get(optional, ""):
            warnings.append(f"Optional field is blank: {optional}")

    return CatalogImportPlan(
        ready=not errors,
        action="append_new_paper",
        paper_id=paper_id,
        pdf_file=pdf_file,
        profile=profile,
        missing_fields=missing_fields,
        errors=tuple(dict.fromkeys(errors)),
        warnings=tuple(warnings),
        catalog_row_count_before=len(rows),
        catalog_sha256=sha256_file(catalog_file),
        pdf_sha256=expected_hash,
        pdf_size_bytes=expected_size if isinstance(expected_size, int) else None,
    )


def apply_catalog_import(
    plan: CatalogImportPlan,
    inbox_path: str | Path,
    catalog_path: str | Path,
    backup_directory: str | Path,
) -> CatalogImportResult:
    """Append one validated row with backup, validation, and atomic replace."""
    if not plan.ready:
        raise ValueError("Import plan is not ready; resolve its validation errors first")
    inbox = Path(inbox_path).resolve()
    pdf_path = (inbox / Path(*PurePosixPath(plan.pdf_file).parts)).resolve()
    try:
        pdf_path.relative_to(inbox)
    except ValueError as error:
        raise RuntimeError("Drafted PDF path escaped the configured inbox") from error
    if not pdf_path.is_file():
        raise RuntimeError("Drafted PDF disappeared after preview")
    if sha256_file(pdf_path) != plan.pdf_sha256:
        raise RuntimeError("PDF content changed after preview; import cancelled")
    if plan.pdf_size_bytes is not None and pdf_path.stat().st_size != plan.pdf_size_bytes:
        raise RuntimeError("PDF size changed after preview; import cancelled")
    catalog_file = Path(catalog_path).resolve()
    if sha256_file(catalog_file) != plan.catalog_sha256:
        raise RuntimeError("Catalog changed after preview; generate a new import plan")

    fields, rows = _read_catalog(catalog_file)
    row = {field: plan.profile.get(field, "") for field in fields}
    row["paper_id"] = plan.paper_id
    row["pdf_file"] = plan.pdf_file
    if "notes_source" in row and not row["notes_source"]:
        row["notes_source"] = "人工确认的候选档案草稿"

    backup_dir = Path(backup_directory).resolve()
    backup_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    backup = backup_dir / f"{catalog_file.stem}.{stamp}.bak.csv"
    shutil.copy2(catalog_file, backup)
    if sha256_file(backup) != plan.catalog_sha256:
        raise RuntimeError("Catalog changed while creating its backup; import cancelled")

    temporary = catalog_file.with_name(f".{catalog_file.name}.import.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows([*rows, row])
        validated = PaperCatalog.from_csv(temporary)
        imported = validated.get(plan.paper_id)
        if plan.pdf_file not in imported.pdf_files:
            raise RuntimeError("Temporary catalog validation lost the imported PDF")
        if sha256_file(catalog_file) != plan.catalog_sha256:
            raise RuntimeError("Catalog changed during import; atomic replace cancelled")
        temporary.replace(catalog_file)
    finally:
        temporary.unlink(missing_ok=True)

    return CatalogImportResult(
        status="imported",
        paper_id=plan.paper_id,
        pdf_file=plan.pdf_file,
        catalog_row_count=len(rows) + 1,
        backup_file=backup.name,
    )
