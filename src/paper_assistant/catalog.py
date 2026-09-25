"""Load and normalize paper-level profiles from the catalog CSV."""

from __future__ import annotations

import csv
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path


def _unique(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(value.strip() for value in values if value.strip()))


def _split(values: Iterable[str], separator: str = ";") -> tuple[str, ...]:
    return _unique(
        item
        for value in values
        for item in value.split(separator)
    )


@dataclass(frozen=True)
class PaperProfile:
    """One retrievable paper, possibly backed by several local PDF versions."""

    paper_id: str
    pdf_files: tuple[str, ...]
    canonical_title: str
    title_zh: str
    authors: tuple[str, ...]
    year: str
    venue: str
    languages: tuple[str, ...]
    tags: tuple[str, ...]
    method_summaries: tuple[str, ...]
    datasets: tuple[str, ...]
    memory_cues: tuple[str, ...]
    duplicate_groups: tuple[str, ...]

    @property
    def display_title(self) -> str:
        """Prefer the Chinese title when it is available."""
        return self.title_zh or self.canonical_title

    def weighted_fields(self) -> tuple[tuple[str, int], ...]:
        """Return search fields and simple, explainable indexing weights."""
        return (
            (self.canonical_title, 3),
            (self.title_zh, 3),
            (" ".join(self.tags), 3),
            (" ".join(self.method_summaries), 2),
            (" ".join(self.datasets), 2),
            (" ".join(self.memory_cues), 2),
            (" ".join(self.authors), 1),
            (self.year, 1),
            (self.venue, 1),
        )


class PaperCatalog:
    """A validated collection of paper profiles keyed by ``paper_id``."""

    REQUIRED_COLUMNS = {
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
    }

    def __init__(self, profiles: Iterable[PaperProfile]) -> None:
        profile_list = list(profiles)
        if not profile_list:
            raise ValueError("Paper catalog cannot be empty")
        self._profiles = {profile.paper_id: profile for profile in profile_list}
        if len(self._profiles) != len(profile_list):
            raise ValueError("Paper profiles must have unique paper_id values")

    @property
    def profiles(self) -> tuple[PaperProfile, ...]:
        return tuple(self._profiles.values())

    def get(self, paper_id: str) -> PaperProfile:
        return self._profiles[paper_id]

    def __len__(self) -> int:
        return len(self._profiles)

    @classmethod
    def from_csv(cls, path: str | Path) -> PaperCatalog:
        catalog_path = Path(path)
        with catalog_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            columns = set(reader.fieldnames or ())
            missing = cls.REQUIRED_COLUMNS - columns
            if missing:
                raise ValueError(
                    f"Catalog is missing required columns: {', '.join(sorted(missing))}"
                )
            rows = list(reader)

        grouped: dict[str, list[dict[str, str]]] = {}
        for row_number, row in enumerate(rows, start=2):
            paper_id = (row.get("paper_id") or "").strip()
            if not paper_id:
                raise ValueError(f"Catalog row {row_number} has an empty paper_id")
            grouped.setdefault(paper_id, []).append(row)

        profiles = []
        for paper_id, paper_rows in grouped.items():
            first = paper_rows[0]
            profiles.append(
                PaperProfile(
                    paper_id=paper_id,
                    pdf_files=_unique(row["pdf_file"] for row in paper_rows),
                    canonical_title=first["canonical_title"].strip(),
                    title_zh=first["title_zh"].strip(),
                    authors=_split(row["authors"] for row in paper_rows),
                    year=first["year"].strip(),
                    venue=first["venue"].strip(),
                    languages=_unique(row["language"] for row in paper_rows),
                    tags=_split(row["tags"] for row in paper_rows),
                    method_summaries=_unique(
                        row["method_summary"] for row in paper_rows
                    ),
                    datasets=_split(row["datasets"] for row in paper_rows),
                    memory_cues=_unique(row["memory_cues"] for row in paper_rows),
                    duplicate_groups=_unique(
                        row["duplicate_group"] for row in paper_rows
                    ),
                )
            )
        return cls(profiles)
