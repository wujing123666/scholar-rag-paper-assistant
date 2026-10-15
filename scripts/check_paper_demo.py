"""Offline public-demo validation. No LLM, embeddings, Chroma or model downloads."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.paper_assistant.catalog import PaperCatalog
from src.paper_assistant.chunk_evaluation import load_chunk_evaluation_cases
from src.paper_assistant.chunk_retriever import extract_paper_chunks
from src.paper_assistant.evaluation import evaluate_search_service, load_evaluation_cases
from src.paper_assistant.service import PaperAssistantConfig, build_paper_search_service

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "examples" / "paper_demo"


def check_demo(demo: Path = DEMO) -> dict:
    catalog = PaperCatalog.from_csv(demo / "paper_catalog.csv")
    service = build_paper_search_service(
        PaperAssistantConfig(
            catalog_path=demo / "paper_catalog.csv",
            inbox_path=demo / "pdfs",
            retriever="bm25",
            paper_reranker="none",
            min_score=0.0,
        )
    )
    report = evaluate_search_service(service, load_evaluation_cases(demo / "paper_queries.jsonl"))
    chunks_by_paper = {
        profile.paper_id: extract_paper_chunks(
            profile, demo / "pdfs" / profile.pdf_files[0], chunk_size=1200, chunk_overlap=180
        )
        for profile in catalog.profiles
    }
    # Check source labels against parsed PDF pages, not against model-generated answers.
    failures = []
    for case in load_chunk_evaluation_cases(demo / "chunk_queries.jsonl"):
        texts = [
            chunk.text.casefold()
            for chunk in chunks_by_paper[case["expected_paper_id"]]
            if chunk.page_number in case["relevant_pages"]
        ]
        found = sum(
            any(term.casefold() in text for text in texts) for term in case["evidence_terms"]
        )
        if found < case["min_evidence_terms"]:
            failures.append(case["id"])
    successful = (
        all(item["rank"] == 1 for item in report["cases"] if item["expected_paper_id"])
        and report["rejection"]["false_accepts"] == 0
        and not failures
        and all(chunks_by_paper.values())
    )
    return {
        "schema_version": 1,
        "purpose": "synthetic_development_smoke",
        "network_calls": 0,
        "success": successful,
        "bm25_min_score": 0.0,
        "papers": len(catalog),
        "chunks": sum(map(len, chunks_by_paper.values())),
        "source_label_failures": failures,
        "paper_retrieval": report,
        "limits": "Lexical paper search and PDF/source-label checks only. No semantic retrieval, generation, Claim verification or hallucination measurement.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "tmp/paper_demo/offline_report.json")
    args = parser.parse_args()
    report = check_demo()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
