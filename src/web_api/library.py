"""Immutable catalog snapshot and opaque PDF identity mapping."""

import hashlib
from pathlib import Path
from urllib.parse import quote

from src.paper_assistant.catalog import PaperCatalog
from src.paper_assistant.inventory import sha256_file
from src.web_api.schemas import Paper, PaperFile


class Library:
    def __init__(self, catalog_path: Path, inbox: Path, chroma_path: Path) -> None:
        self.catalog_path = catalog_path.resolve()
        self.inbox = inbox.resolve()
        self.catalog_bytes = self.catalog_path.read_bytes()
        self.catalog = PaperCatalog.from_csv(self.catalog_path)
        self.fingerprints: dict[Path, tuple[int, int]] = {}
        source_hashes: dict[Path, str] = {}
        for profile in self.catalog.profiles:
            for name in profile.pdf_files:
                path = (self.inbox / name).resolve()
                if not path.is_relative_to(self.inbox) or not path.is_file() or path.suffix.lower() != ".pdf":
                    raise ValueError("Catalog contains a missing or unsafe PDF")
                if path not in source_hashes:
                    before = path.stat()
                    source_hashes[path] = sha256_file(path)
                    after = path.stat()
                    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                        raise ValueError("PDF changed during startup")
                    self.fingerprints[path] = (after.st_size, after.st_mtime_ns)
        identity = self.catalog_bytes + str(self.inbox).encode() + str(chroma_path.resolve()).encode()
        for path, digest in sorted(source_hashes.items()):
            identity += b"\0" + str(path).encode() + b"\0" + digest.encode()
        self.identity_digest = hashlib.sha256(identity).hexdigest()
        self.dataset_id = self.identity_digest[:24]
        self.files: dict[tuple[str, str], Path] = {}
        self.names: dict[tuple[str, str], str] = {}
        self.papers: dict[str, Paper] = {}
        for profile in self.catalog.profiles:
            files = []
            for name in profile.pdf_files:
                path = (self.inbox / name).resolve()
                if not path.is_relative_to(self.inbox) or not path.is_file() or path.suffix.lower() != ".pdf":
                    raise ValueError("Catalog contains a missing or unsafe PDF")
                file_id = hashlib.sha256(f"{self.dataset_id}|{profile.paper_id}|{name}".encode()).hexdigest()[:24]
                self.files[profile.paper_id, file_id] = path
                self.names[profile.paper_id, name] = file_id
                files.append(PaperFile(
                    file_id=file_id, name=path.name,
                    url=f"/api/v1/papers/{quote(profile.paper_id, safe='')}/files/{file_id}",
                ))
            self.papers[profile.paper_id] = Paper(
                paper_id=profile.paper_id, title=profile.display_title,
                canonical_title=profile.canonical_title, authors=list(profile.authors),
                year=profile.year or None, venue=profile.venue, tags=list(profile.tags),
                summary="；".join(profile.method_summaries), files=files,
            )
        self.assert_unchanged()

    def assert_unchanged(self) -> None:
        if self.catalog_path.read_bytes() != self.catalog_bytes:
            raise ValueError("Dataset changed; restart after offline index preparation")
        for path, fingerprint in self.fingerprints.items():
            stat = path.stat()
            if (stat.st_size, stat.st_mtime_ns) != fingerprint or not path.resolve().is_relative_to(self.inbox):
                raise ValueError("PDF source changed; restart after offline preparation")

    def bind(self, state_path: Path) -> None:
        # Binding is outside Chroma and contains no credentials or source text.
        state_path.mkdir(parents=True, exist_ok=True)
        binding = state_path / "dataset.sha256"
        digest = self.identity_digest
        if binding.exists() and binding.read_text(encoding="utf-8").strip() != digest:
            raise ValueError("Dataset binding mismatch")
        binding.write_text(digest + "\n", encoding="utf-8")
