"""Command-line entry point for the ScholarRAG paper retriever."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.libs.embedding.fastembed_embedding import FastEmbedEmbedding
from src.paper_assistant.catalog import PaperCatalog
from src.paper_assistant.dense_retriever import PaperDenseRetriever
from src.paper_assistant.evaluation import evaluate_retriever, load_evaluation_cases
from src.paper_assistant.hybrid_retriever import PaperHybridRetriever
from src.paper_assistant.retriever import PaperBM25Retriever

DEFAULT_CATALOG = Path("data/papers/paper_catalog.csv")
DEFAULT_EVALUATION = Path("data/papers/eval_queries.jsonl")


DEFAULT_DENSE_MODEL = "BAAI/bge-small-zh-v1.5"


def _build_retriever(
    catalog_path: Path,
    retriever_name: str,
    model: str,
    model_cache: Path | None,
) -> PaperBM25Retriever | PaperDenseRetriever | PaperHybridRetriever:
    catalog = PaperCatalog.from_csv(catalog_path)
    if retriever_name == "bm25":
        return PaperBM25Retriever(catalog)
    embedding = FastEmbedEmbedding(model=model, cache_dir=model_cache)
    dense = PaperDenseRetriever(catalog, embedding)
    if retriever_name == "dense":
        return dense
    return PaperHybridRetriever(catalog, PaperBM25Retriever(catalog), dense)


def main() -> int:
    parser = argparse.ArgumentParser(description="ScholarRAG paper-level retrieval")
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument(
        "--retriever", choices=("bm25", "dense", "hybrid"), default="bm25"
    )
    parser.add_argument("--model", default=DEFAULT_DENSE_MODEL)
    parser.add_argument("--model-cache", type=Path)
    subparsers = parser.add_subparsers(dest="command", required=True)

    search_parser = subparsers.add_parser("search", help="Find papers from a description")
    search_parser.add_argument("query")
    search_parser.add_argument("--top-k", type=int, default=3)

    evaluate_parser = subparsers.add_parser("evaluate", help="Evaluate a paper retriever")
    evaluate_parser.add_argument("--queries", type=Path, default=DEFAULT_EVALUATION)
    evaluate_parser.add_argument("--split", choices=("dev", "test_candidate"))

    args = parser.parse_args()
    retriever = _build_retriever(
        args.catalog, args.retriever, args.model, args.model_cache
    )

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
