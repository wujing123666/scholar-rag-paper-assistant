"""Audit local paper PDFs against the paper catalog."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from src.paper_assistant.catalog import PaperCatalog

HASH_CHUNK_SIZE = 1024 * 1024


def _normalized_relative_path(value: str) -> str:
    normalized = PurePosixPath(value.strip().replace("\\", "/")).as_posix()
    while normalized.startswith("./"):
        normalized = normalized[2:]
    return normalized


def sha256_file(path: str | Path) -> str:
    """Hash a file without loading the entire PDF into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(HASH_CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class InventoryFile:
    path: str
    size_bytes: int
    sha256: str
    paper_ids: tuple[str, ...]


@dataclass(frozen=True)
class ExactDuplicateGroup:
    sha256: str
    files: tuple[str, ...]


@dataclass(frozen=True)
class PaperVersionGroup:
    paper_id: str
    files: tuple[str, ...]


@dataclass(frozen=True)
class CatalogPathConflict:
    path: str
    paper_ids: tuple[str, ...]


@dataclass(frozen=True)
class PaperInventoryReport:
    status: str
    inbox: str
    catalog: str
    hash_algorithm: str
    scanned_pdf_count: int
    catalog_paper_count: int
    catalog_file_entry_count: int
    catalog_unique_path_count: int
    registered_pdf_count: int
    unregistered_files: tuple[str, ...]
    missing_files: tuple[str, ...]
    exact_duplicate_groups: tuple[ExactDuplicateGroup, ...]
    paper_version_groups: tuple[PaperVersionGroup, ...]
    catalog_path_conflicts: tuple[CatalogPathConflict, ...]
    files: tuple[InventoryFile, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable report."""
        return asdict(self)

    def summary_dict(self) -> dict[str, Any]:
        """Return a concise report without local absolute paths or file hashes."""
        return {
            "status": self.status,
            "scanned_pdf_count": self.scanned_pdf_count,
            "catalog_paper_count": self.catalog_paper_count,
            "catalog_file_entry_count": self.catalog_file_entry_count,
            "catalog_unique_path_count": self.catalog_unique_path_count,
            "registered_pdf_count": self.registered_pdf_count,
            "unregistered_files": self.unregistered_files,
            "missing_files": self.missing_files,
            "exact_duplicate_file_groups": tuple(
                group.files for group in self.exact_duplicate_groups
            ),
            "paper_version_groups": tuple(
                asdict(group) for group in self.paper_version_groups
            ),
            "catalog_path_conflicts": tuple(
                asdict(conflict) for conflict in self.catalog_path_conflicts
            ),
        }


def build_paper_inventory(
    inbox_path: str | Path,
    catalog_path: str | Path,
) -> PaperInventoryReport:
    """Scan local PDFs and compare them with catalog file entries."""
    inbox = Path(inbox_path).resolve()
    catalog_file = Path(catalog_path).resolve()
    if not inbox.is_dir():
        raise FileNotFoundError(f"Paper inbox does not exist: {inbox}")

    catalog = PaperCatalog.from_csv(catalog_file)
    catalog_paths: dict[str, set[str]] = {}
    display_paths: dict[str, str] = {}
    for profile in catalog.profiles:
        for pdf_file in profile.pdf_files:
            relative = _normalized_relative_path(pdf_file)
            key = relative.casefold()
            display_paths.setdefault(key, relative)
            catalog_paths.setdefault(key, set()).add(profile.paper_id)

    pdf_paths = sorted(
        (path for path in inbox.rglob("*") if path.is_file() and path.suffix.lower() == ".pdf"),
        key=lambda path: path.relative_to(inbox).as_posix().casefold(),
    )
    files: list[InventoryFile] = []
    scanned_keys: set[str] = set()
    hashes: dict[str, list[str]] = {}
    for path in pdf_paths:
        relative = path.relative_to(inbox).as_posix()
        key = relative.casefold()
        scanned_keys.add(key)
        digest = sha256_file(path)
        hashes.setdefault(digest, []).append(relative)
        files.append(
            InventoryFile(
                path=relative,
                size_bytes=path.stat().st_size,
                sha256=digest,
                paper_ids=tuple(sorted(catalog_paths.get(key, ()))),
            )
        )

    unregistered = tuple(file.path for file in files if not file.paper_ids)
    missing = tuple(
        display_paths[key] for key in sorted(catalog_paths) if key not in scanned_keys
    )
    exact_duplicates = tuple(
        ExactDuplicateGroup(sha256=digest, files=tuple(sorted(group)))
        for digest, group in sorted(hashes.items())
        if len(group) > 1
    )
    version_groups = tuple(
        PaperVersionGroup(paper_id=profile.paper_id, files=profile.pdf_files)
        for profile in sorted(catalog.profiles, key=lambda item: item.paper_id)
        if len(profile.pdf_files) > 1
    )
    path_conflicts = tuple(
        CatalogPathConflict(path=display_paths[key], paper_ids=tuple(sorted(paper_ids)))
        for key, paper_ids in sorted(catalog_paths.items())
        if len(paper_ids) > 1
    )
    status = (
        "attention_required"
        if unregistered or missing or exact_duplicates or path_conflicts
        else "ok"
    )
    return PaperInventoryReport(
        status=status,
        inbox=str(inbox),
        catalog=str(catalog_file),
        hash_algorithm="sha256",
        scanned_pdf_count=len(files),
        catalog_paper_count=len(catalog),
        catalog_file_entry_count=sum(
            len(profile.pdf_files) for profile in catalog.profiles
        ),
        catalog_unique_path_count=len(catalog_paths),
        registered_pdf_count=sum(bool(file.paper_ids) for file in files),
        unregistered_files=unregistered,
        missing_files=missing,
        exact_duplicate_groups=exact_duplicates,
        paper_version_groups=version_groups,
        catalog_path_conflicts=path_conflicts,
        files=tuple(files),
    )


def write_inventory_report(report: PaperInventoryReport, path: str | Path) -> Path:
    """Write a report atomically so interrupted runs do not leave partial JSON."""
    output = Path(path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_text(
        json.dumps(report.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
    return output
