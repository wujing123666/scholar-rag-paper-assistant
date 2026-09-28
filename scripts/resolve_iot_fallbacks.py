"""Resolve alternate open PDF locations for failed IoT corpus downloads."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

from scripts.collect_iot_corpus import OPENALEX_WORKS_URL, _request_json


def alternate_pdf_urls(work: dict[str, Any], failed_url: str) -> list[str]:
    """Return unique alternatives, preferring repositories over publishers."""
    candidates: list[tuple[int, str]] = []
    seen = {failed_url.rstrip("/")}
    for location in work.get("locations") or []:
        url = str(location.get("pdf_url") or "").strip()
        if not url.startswith(("http://", "https://")) or url.rstrip("/") in seen:
            continue
        seen.add(url.rstrip("/"))
        source_type = str((location.get("source") or {}).get("type") or "")
        priority = 0 if source_type == "repository" else 1
        candidates.append((priority, url))
    return [url for _, url in sorted(candidates)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--catalog",
        type=Path,
        default=Path("data/papers/discovery/iot_candidates.csv"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/papers/corpus_iot_1000/download_manifest.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/papers/discovery/iot_fallback_candidates.csv"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        failed = {
            row["paper_id"]: row
            for row in csv.DictReader(handle)
            if row["status"] == "failed"
        }
    with args.catalog.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        catalog = {row["paper_id"]: row for row in reader if row["paper_id"] in failed}

    openalex_ids = [paper_id.removeprefix("oa_").upper() for paper_id in failed]
    cache_path = args.output.with_suffix(".openalex_locations.json")
    if cache_path.is_file():
        works: dict[str, dict[str, Any]] = json.loads(
            cache_path.read_text(encoding="utf-8")
        )
    else:
        works = {}
    remaining_ids = [work_id for work_id in openalex_ids if f"oa_{work_id.lower()}" not in works]
    for index, openalex_id in enumerate(remaining_ids, start=1):
        work = _request_json(
            f"{OPENALEX_WORKS_URL}/{openalex_id}?select=id,locations"
        )
        work_id = str(work.get("id") or "").rsplit("/", 1)[-1].lower()
        works[f"oa_{work_id}"] = work
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(works, ensure_ascii=False), encoding="utf-8")
        if index % 10 == 0 or index == len(remaining_ids):
            print(f"resolved {index}/{len(remaining_ids)}", flush=True)
        time.sleep(0.15)

    retry_rows: list[dict[str, str]] = []
    for paper_id, failed_row in failed.items():
        row = catalog.get(paper_id)
        work = works.get(paper_id)
        if row is None or work is None:
            continue
        alternatives = alternate_pdf_urls(work, failed_row["pdf_url"])
        if not alternatives:
            continue
        row = dict(row)
        row["pdf_url"] = alternatives[0]
        row["alternate_pdf_urls"] = json.dumps(alternatives[1:], ensure_ascii=False)
        retry_rows.append(row)

    output_fields = fields + (["alternate_pdf_urls"] if "alternate_pdf_urls" not in fields else [])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=output_fields)
        writer.writeheader()
        writer.writerows(retry_rows)
    print(
        f"failed={len(failed)} with_alternatives={len(retry_rows)} output={args.output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
