"""Download and verify open-access PDFs listed by ``collect_iot_corpus.py``."""

from __future__ import annotations

import argparse
import csv
import hashlib
import http.client
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path

MIN_PDF_BYTES = 10_000
MAX_PDF_BYTES = 100 * 1024 * 1024


@dataclass(frozen=True)
class DownloadResult:
    paper_id: str
    title: str
    doi: str
    pdf_url: str
    local_file: str
    status: str
    size_bytes: int
    sha256: str
    error: str


def _looks_like_pdf(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size < MIN_PDF_BYTES:
        return False
    with path.open("rb") as handle:
        return handle.read(1024).lstrip().startswith(b"%PDF-")


def _safe_url(value: str) -> str:
    """Percent-encode unsafe URL path/query characters without double encoding."""
    parts = urllib.parse.urlsplit(value)
    return urllib.parse.urlunsplit(
        (
            parts.scheme,
            parts.netloc,
            urllib.parse.quote(parts.path, safe="/%:@"),
            urllib.parse.quote(parts.query, safe="=&%:@/?+"),
            parts.fragment,
        )
    )


def _download_one(
    row: dict[str, str], output_dir: Path, attempts: int, timeout: int
) -> DownloadResult:
    paper_id = row["paper_id"]
    target = output_dir / f"{paper_id}.pdf"
    if _looks_like_pdf(target):
        digest = hashlib.sha256(target.read_bytes()).hexdigest()
        return DownloadResult(
            paper_id, row["title"], row["doi"], row["pdf_url"], str(target),
            "existing_valid", target.stat().st_size, digest, ""
        )

    part = target.with_suffix(".pdf.part")
    raw_alternatives = row.get("alternate_pdf_urls", "")
    try:
        alternatives = json.loads(raw_alternatives) if raw_alternatives else []
    except json.JSONDecodeError:
        alternatives = []
    urls = list(dict.fromkeys([row["pdf_url"], *alternatives]))
    errors: list[str] = []
    for pdf_url in urls:
        for attempt in range(attempts):
            digest = hashlib.sha256()
            size = 0
            try:
                deadline = time.monotonic() + timeout
                request = urllib.request.Request(
                    _safe_url(pdf_url),
                    headers={
                        "Accept": "application/pdf,*/*;q=0.8",
                        "User-Agent": "Mozilla/5.0 ScholarRAG-IoT-Corpus/1.0",
                    },
                )
                with urllib.request.urlopen(
                    request, timeout=min(timeout, 15)
                ) as response, part.open("wb") as handle:
                    while chunk := response.read(64 * 1024):
                        if time.monotonic() > deadline:
                            raise TimeoutError(f"download exceeded {timeout} seconds")
                        size += len(chunk)
                        if size > MAX_PDF_BYTES:
                            raise ValueError("PDF exceeds the 100 MiB safety limit")
                        digest.update(chunk)
                        handle.write(chunk)
                if not _looks_like_pdf(part):
                    raise ValueError("response is not a valid PDF or is smaller than 10 KiB")
                os.replace(part, target)
                return DownloadResult(
                    paper_id, row["title"], row["doi"], pdf_url, str(target),
                    "downloaded", size, digest.hexdigest(), ""
                )
            except (
                OSError,
                TimeoutError,
                ValueError,
                http.client.HTTPException,
                urllib.error.URLError,
            ) as error:
                errors.append(f"{pdf_url} -> {type(error).__name__}: {error}")
                part.unlink(missing_ok=True)
                if attempt < attempts - 1:
                    time.sleep(2**attempt)

    return DownloadResult(
        paper_id, row["title"], row["doi"], row["pdf_url"], str(target),
        "failed", 0, "", " | ".join(errors)
    )


def _write_manifest(path: Path, results: list[DownloadResult]) -> None:
    fields = tuple(DownloadResult.__dataclass_fields__)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(asdict(result) for result in results)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--catalog",
        type=Path,
        default=Path("data/papers/discovery/iot_candidates.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/papers/corpus_iot_1000/pdfs"),
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Output manifest path (defaults beside the PDF directory)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with args.catalog.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row.get("pdf_url")]

    results: list[DownloadResult] = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {
            executor.submit(
                _download_one, row, args.output_dir, args.attempts, args.timeout
            ): row
            for row in rows
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            row = futures[future]
            try:
                result = future.result()
            except Exception as error:  # keep one malformed provider response isolated
                result = DownloadResult(
                    row["paper_id"],
                    row["title"],
                    row["doi"],
                    row["pdf_url"],
                    str(args.output_dir / f"{row['paper_id']}.pdf"),
                    "failed",
                    0,
                    "",
                    f"{type(error).__name__}: {error}",
                )
            results.append(result)
            if completed % 25 == 0 or completed == len(futures):
                valid = sum(item.status != "failed" for item in results)
                print(
                    f"processed={completed}/{len(futures)} valid={valid} "
                    f"failed={completed - valid}",
                    flush=True,
                )

    results.sort(key=lambda item: item.paper_id)
    manifest = args.manifest or (args.output_dir.parent / "download_manifest.csv")
    manifest.parent.mkdir(parents=True, exist_ok=True)
    _write_manifest(manifest, results)
    valid = sum(item.status != "failed" for item in results)
    print(f"Finished: valid={valid}, failed={len(results) - valid}, manifest={manifest}")


if __name__ == "__main__":
    main()
