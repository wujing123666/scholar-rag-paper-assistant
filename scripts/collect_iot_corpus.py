"""Collect a balanced, open-access IoT journal corpus from OpenAlex.

The output is a discovery catalog. It is intentionally kept separate from the
reviewed ``paper_catalog.csv`` used by the RAG system: a paper only enters the
formal catalog after its PDF has been downloaded and validated.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

OPENALEX_WORKS_URL = "https://api.openalex.org/works"
OPENALEX_SELECT = ",".join(
    (
        "id",
        "doi",
        "title",
        "authorships",
        "publication_year",
        "primary_location",
        "best_oa_location",
        "open_access",
        "topics",
        "cited_by_count",
    )
)
DEFAULT_QUERIES = (
    ("core_iot", "internet of things"),
    ("industrial_iot", "industrial internet of things"),
    ("iot_security", "internet of things security"),
    ("edge_iot", "internet of things edge computing"),
    ("sensor_networks", "wireless sensor networks internet of things"),
    ("smart_cities", "smart cities internet of things"),
    ("iot_healthcare", "healthcare internet of things"),
    ("smart_agriculture", "smart agriculture internet of things"),
    ("federated_iot", "federated learning internet of things"),
    ("mobile_crowdsensing", "mobile crowdsensing"),
)
OUTPUT_COLUMNS = (
    "paper_id",
    "title",
    "authors",
    "year",
    "journal",
    "doi",
    "openalex_id",
    "topic_bucket",
    "matched_buckets",
    "primary_topic",
    "cited_by_count",
    "oa_status",
    "pdf_url",
    "landing_page_url",
    "issn",
    "publisher",
    "abstract",
    "acquisition_status",
)


@dataclass(frozen=True)
class CollectionConfig:
    target: int
    per_query: int
    from_date: str
    to_date: str
    output_dir: Path
    mailto: str = ""
    require_direct_pdf: bool = True


def reconstruct_abstract(inverted_index: dict[str, list[int]] | None) -> str:
    """Rebuild an OpenAlex abstract from its token-position index."""
    if not inverted_index:
        return ""
    positioned = [
        (position, token)
        for token, positions in inverted_index.items()
        for position in positions
    ]
    return " ".join(token for _, token in sorted(positioned))


def normalize_doi(value: str | None) -> str:
    """Return the bare, lowercase DOI used as the primary deduplication key."""
    if not value:
        return ""
    return re.sub(r"^https?://(?:dx\.)?doi\.org/", "", value.strip(), flags=re.I).lower()


def normalize_title(value: str) -> str:
    """Return a conservative title key for records that do not share a DOI."""
    return re.sub(r"[^a-z0-9]+", "", value.casefold())


def _request_json(url: str, *, attempts: int = 7) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": "ScholarRAG-IoT-Corpus/1.0",
        },
    )
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return json.load(response)
        except urllib.error.HTTPError as error:
            if attempt == attempts - 1:
                raise
            if error.code == 429:
                retry_after = error.headers.get("Retry-After", "")
                delay = int(retry_after) if retry_after.isdigit() else 5 * (2**attempt)
                time.sleep(min(delay, 60))
            else:
                time.sleep(min(2**attempt, 30))
        except (TimeoutError, urllib.error.URLError, json.JSONDecodeError):
            if attempt == attempts - 1:
                raise
            time.sleep(min(2**attempt, 30))
    raise RuntimeError("unreachable")


def _work_filter(config: CollectionConfig, query: str) -> str:
    clauses = (
        "type:article",
        "has_doi:true",
        "has_abstract:true",
        "is_oa:true",
        "language:en",
        f"from_publication_date:{config.from_date}",
        f"to_publication_date:{config.to_date}",
        "primary_location.source.type:journal",
        f"title_and_abstract.search:{query}",
    )
    return ",".join(clauses)


def fetch_query(config: CollectionConfig, bucket: str, query: str) -> list[dict[str, Any]]:
    """Fetch one ranked topic bucket with cursor pagination."""
    results: list[dict[str, Any]] = []
    cursor = "*"
    while len(results) < config.per_query and cursor:
        params = {
            "filter": _work_filter(config, query),
            "sort": "relevance_score:desc",
            # Abstract position maps are intentionally excluded, so OpenAlex's
            # 200-row page size stays small enough while reducing rate-limit load.
            "per-page": min(200, config.per_query - len(results)),
            "cursor": cursor,
            "select": OPENALEX_SELECT,
        }
        if config.mailto:
            params["mailto"] = config.mailto
        payload = _request_json(f"{OPENALEX_WORKS_URL}?{urllib.parse.urlencode(params)}")
        page = payload.get("results") or []
        if not page:
            break
        for work in page:
            work["_topic_bucket"] = bucket
        results.extend(page)
        cursor = (payload.get("meta") or {}).get("next_cursor")
        time.sleep(0.25)
    return results[: config.per_query]


def _cache_path(config: CollectionConfig, bucket: str) -> Path:
    return config.output_dir / ".openalex_cache" / f"{bucket}.json"


def load_cached_query(
    config: CollectionConfig, bucket: str, query: str
) -> list[dict[str, Any]] | None:
    path = _cache_path(config, bucket)
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "query": query,
        "per_query": config.per_query,
        "from_date": config.from_date,
        "to_date": config.to_date,
    }
    if payload.get("request") != expected:
        return None
    results = payload.get("results")
    return results if isinstance(results, list) else None


def save_cached_query(
    config: CollectionConfig,
    bucket: str,
    query: str,
    results: list[dict[str, Any]],
) -> None:
    path = _cache_path(config, bucket)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "request": {
            "query": query,
            "per_query": config.per_query,
            "from_date": config.from_date,
            "to_date": config.to_date,
        },
        "results": results,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _dedupe_key(work: dict[str, Any]) -> str:
    doi = normalize_doi(work.get("doi"))
    if doi:
        return f"doi:{doi}"
    return f"title:{normalize_title(str(work.get('title') or ''))}"


def merge_duplicates(
    bucket_results: dict[str, list[dict[str, Any]]],
) -> dict[str, dict[str, Any]]:
    """Merge duplicate DOI/title records and retain every matched topic bucket."""
    merged: dict[str, dict[str, Any]] = {}
    for bucket, works in bucket_results.items():
        for work in works:
            key = _dedupe_key(work)
            if not key or key == "title:":
                continue
            if key not in merged:
                merged[key] = work
                merged[key]["_matched_buckets"] = [bucket]
            elif bucket not in merged[key]["_matched_buckets"]:
                merged[key]["_matched_buckets"].append(bucket)
    return merged


def balanced_select(
    bucket_results: dict[str, list[dict[str, Any]]],
    target: int,
    *,
    require_direct_pdf: bool = False,
) -> list[dict[str, Any]]:
    """Round-robin buckets while deduplicating, then fill from remaining records."""
    eligible_results = {
        bucket: [
            work
            for work in works
            if not require_direct_pdf or (work.get("best_oa_location") or {}).get("pdf_url")
        ]
        for bucket, works in bucket_results.items()
    }
    queues = {bucket: deque(works) for bucket, works in eligible_results.items()}
    merged = merge_duplicates(eligible_results)
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()

    while len(selected) < target and any(queues.values()):
        made_progress = False
        for bucket in eligible_results:
            queue = queues[bucket]
            while queue:
                work = queue.popleft()
                key = _dedupe_key(work)
                if key in seen or key not in merged:
                    continue
                record = merged[key]
                record["_selected_bucket"] = bucket
                selected.append(record)
                seen.add(key)
                made_progress = True
                break
            if len(selected) >= target:
                break
        if not made_progress:
            break

    return selected


def work_to_row(work: dict[str, Any]) -> dict[str, str | int]:
    primary_location = work.get("primary_location") or {}
    source = primary_location.get("source") or {}
    oa_location = work.get("best_oa_location") or {}
    open_access = work.get("open_access") or {}
    authors = [
        ((authorship.get("author") or {}).get("display_name") or "").strip()
        for authorship in work.get("authorships") or []
    ]
    authors = [author for author in authors if author]
    topics = work.get("topics") or []
    doi = normalize_doi(work.get("doi"))
    openalex_id = str(work.get("id") or "").rsplit("/", 1)[-1]
    pdf_url = str(oa_location.get("pdf_url") or "")
    landing_url = str(oa_location.get("landing_page_url") or primary_location.get("landing_page_url") or "")
    return {
        "paper_id": f"oa_{openalex_id.lower()}",
        "title": str(work.get("title") or ""),
        "authors": ";".join(authors),
        "year": work.get("publication_year") or "",
        "journal": str(source.get("display_name") or ""),
        "doi": doi,
        "openalex_id": openalex_id,
        "topic_bucket": str(work.get("_selected_bucket") or work.get("_topic_bucket") or ""),
        "matched_buckets": ";".join(work.get("_matched_buckets") or []),
        "primary_topic": str((topics[0] if topics else {}).get("display_name") or ""),
        "cited_by_count": work.get("cited_by_count") or 0,
        "oa_status": str(open_access.get("oa_status") or ""),
        "pdf_url": pdf_url,
        "landing_page_url": landing_url,
        "issn": ";".join(source.get("issn") or []),
        "publisher": str(source.get("host_organization_name") or ""),
        # Full abstracts are intentionally omitted from the batch response because
        # OpenAlex encodes them as large position maps. The PDF ingestion pipeline
        # extracts the authoritative text after download.
        "abstract": "",
        "acquisition_status": "pdf_url_ready" if pdf_url else "oa_landing_page_only",
    }


def write_outputs(config: CollectionConfig, works: list[dict[str, Any]]) -> dict[str, Any]:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    rows = [work_to_row(work) for work in works]
    csv_path = config.output_dir / "iot_candidates.csv"
    jsonl_path = config.output_dir / "iot_candidates.jsonl"
    summary_path = config.output_dir / "iot_collection_summary.json"

    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    with jsonl_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "generated_on": date.today().isoformat(),
        "target": config.target,
        "selected": len(rows),
        "year_range": [config.from_date, config.to_date],
        "criteria": [
            "journal article",
            "English",
            "open access",
            "has DOI",
            "has abstract",
        ],
        "topic_buckets": dict(Counter(str(row["topic_bucket"]) for row in rows)),
        "direct_pdf_urls": sum(bool(row["pdf_url"]) for row in rows),
        "oa_landing_page_only": sum(not bool(row["pdf_url"]) for row in rows),
        "unique_journals": len({str(row["journal"]) for row in rows}),
        "csv": str(csv_path),
        "jsonl": str(jsonl_path),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=int, default=1000)
    parser.add_argument("--per-query", type=int, default=180)
    parser.add_argument("--from-date", default="2019-01-01")
    parser.add_argument("--to-date", default=date.today().isoformat())
    parser.add_argument("--output-dir", type=Path, default=Path("data/papers/discovery"))
    parser.add_argument("--mailto", default="", help="Optional contact email for OpenAlex polite pool")
    parser.add_argument(
        "--allow-landing-page-only",
        action="store_true",
        help="Allow OA records without a direct PDF URL",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = CollectionConfig(
        target=args.target,
        per_query=args.per_query,
        from_date=args.from_date,
        to_date=args.to_date,
        output_dir=args.output_dir,
        mailto=args.mailto,
        require_direct_pdf=not args.allow_landing_page_only,
    )
    bucket_results: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for bucket, query in DEFAULT_QUERIES:
        print(f"Collecting {bucket}: {query}", flush=True)
        cached = load_cached_query(config, bucket, query)
        if cached is not None:
            bucket_results[bucket] = cached
            print("  reused local cache", flush=True)
        else:
            bucket_results[bucket] = fetch_query(config, bucket, query)
            save_cached_query(config, bucket, query, bucket_results[bucket])
        print(f"  received {len(bucket_results[bucket])} records", flush=True)

    selected = balanced_select(
        bucket_results,
        config.target,
        require_direct_pdf=config.require_direct_pdf,
    )
    if len(selected) < config.target:
        raise RuntimeError(
            f"Only {len(selected)} unique records matched; target is {config.target}. "
            "Increase --per-query or broaden the date range."
        )
    summary = write_outputs(config, selected)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
