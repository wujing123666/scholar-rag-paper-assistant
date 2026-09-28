"""Measure production chunking for the audited IoT corpus without embedding it."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path

from src.paper_assistant.catalog import PaperCatalog
from src.paper_assistant.chunk_retriever import extract_paper_chunks


def summarize_chunk_rows(rows: list[dict[str, str | int]]) -> dict[str, int | float]:
    successful = [row for row in rows if row["status"] == "success"]
    counts = sorted(int(row["chunk_count"]) for row in successful)
    return {
        "papers": len(rows),
        "successful_papers": len(successful),
        "failed_papers": len(rows) - len(successful),
        "chunks": sum(counts),
        "chunks_median_per_paper": statistics.median(counts) if counts else 0,
        "chunks_p95_per_paper": counts[max(0, math.ceil(len(counts) * 0.95) - 1)] if counts else 0,
        "embedding_characters": sum(
            int(row["embedding_characters"]) for row in successful
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--catalog",
        type=Path,
        default=Path("data/papers/corpus_iot_1000/catalog/iot_paper_catalog.csv"),
    )
    parser.add_argument(
        "--pdf-dir",
        type=Path,
        default=Path("data/papers/corpus_iot_1000/pdfs"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/papers/corpus_iot_1000/chunk_estimate"),
    )
    parser.add_argument("--chunk-size", type=int, default=1200)
    parser.add_argument("--chunk-overlap", type=int, default=180)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    catalog = PaperCatalog.from_csv(args.catalog)
    rows: list[dict[str, str | int]] = []
    for index, profile in enumerate(catalog.profiles, start=1):
        pdf_file = profile.pdf_files[0]
        try:
            chunks = extract_paper_chunks(
                profile,
                args.pdf_dir / pdf_file,
                chunk_size=args.chunk_size,
                chunk_overlap=args.chunk_overlap,
            )
            rows.append(
                {
                    "paper_id": profile.paper_id,
                    "pdf_file": pdf_file,
                    "status": "success",
                    "chunk_count": len(chunks),
                    "evidence_characters": sum(len(chunk.text) for chunk in chunks),
                    "embedding_characters": sum(
                        len(chunk.embedding_text(profile)) for chunk in chunks
                    ),
                    "error": "",
                }
            )
        except Exception as error:
            rows.append(
                {
                    "paper_id": profile.paper_id,
                    "pdf_file": pdf_file,
                    "status": "failed",
                    "chunk_count": 0,
                    "evidence_characters": 0,
                    "embedding_characters": 0,
                    "error": f"{type(error).__name__}: {error}",
                }
            )
        if index % 25 == 0 or index == len(catalog):
            print(f"chunked={index}/{len(catalog)}", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    details_path = args.output_dir / "chunk_estimate.csv"
    fields = (
        "paper_id",
        "pdf_file",
        "status",
        "chunk_count",
        "evidence_characters",
        "embedding_characters",
        "error",
    )
    with details_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    summary = summarize_chunk_rows(rows)
    embedding_characters = int(summary["embedding_characters"])
    summary.update(
        {
            "chunk_size": args.chunk_size,
            "chunk_overlap": args.chunk_overlap,
            "approximate_english_tokens_at_4_chars_per_token": math.ceil(
                embedding_characters / 4
            ),
            "qwen_batches_at_10_chunks": math.ceil(int(summary["chunks"]) / 10),
            "details": str(details_path),
            "note": "Token count is an engineering estimate only; no embedding API was called.",
        }
    )
    summary_path = args.output_dir / "chunk_estimate_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
