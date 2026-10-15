"""Dataset selection shared by the regular page and the isolated demo."""

from __future__ import annotations

import os
from pathlib import Path

from src.paper_assistant.service import PaperAssistantConfig


def dataset_paths_from_environment() -> dict[str, Path]:
    defaults = PaperAssistantConfig()
    return {
        field: Path(os.environ.get(name) or str(getattr(defaults, field)))
        for field, name in {
            "catalog_path": "SCHOLARRAG_CATALOG",
            "inbox_path": "SCHOLARRAG_INBOX",
            "chroma_path": "SCHOLARRAG_CHROMA_PATH",
            "claim_cache_path": "SCHOLARRAG_CLAIM_CACHE",
        }.items()
    }


def citation_pdf_path(inbox: Path, filename: str) -> Path | None:
    """Serve only PDFs contained in the active dataset, including after symlink resolution."""
    base = inbox.resolve()
    target = (base / filename).resolve()
    if not target.is_relative_to(base) or target.suffix.casefold() != ".pdf":
        return None
    return target if target.is_file() else None
