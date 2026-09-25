"""Command-line entry point for the ScholarRAG paper retriever."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.paper_assistant.catalog import PaperCatalog
from src.paper_assistant.evaluation import evaluate_retriever, load_evaluation_cases
from src.paper_assistant.retriever import PaperBM25Retriever

DEFAULT_CATALOG = Path("data/papers/paper_catalog.csv")
DEFAULT_EVALUATION = Path("data/papers/eval_queries.jsonl")


def _build_retriever(catalog_path: Path) -> PaperBM25Retriever:
    return PaperBM25Retriever(PaperCatalog.from_csv(catalog_path))


def main() -> int:
    parser = argparse.ArgumentParser(description="ScholarRAG paper-level retrieval")
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    subparsers = parser.add_subparsers(dest="command", required=True)

    search_parser = subparsers.add_parser("search", help="Find papers from a description")
    search_parser.add_argument("query")
    search_parser.add_argument("--top-k", type=int, default=3)

    evaluate_parser = subparsers.add_parser("evaluate", help="Evaluate the BM25 baseline")
    evaluate_parser.add_argument("--queries", type=Path, default=DEFAULT_EVALUATION)
    evaluate_parser.add_argument("--split", choices=("dev", "test_candidate"))

    args = parser.parse_args()
    retriever = _build_retriever(args.catalog)

    if args.command == "search":
        results = retriever.search(args.query, top_k=args.top_k)
        output = [
            {
                "rank": rank,
                "paper_id": result.paper.paper_id,
                "title": result.paper.display_title,
                "score": round(result.score, 4),
                "matched_terms": result.matched_terms,
                "pdf_files": result.paper.pdf_files,
            }
            for rank, result in enumerate(results, start=1)
        ]
    else:
        output = evaluate_retriever(
            retriever,
            load_evaluation_cases(args.queries),
            split=args.split,
        )
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
