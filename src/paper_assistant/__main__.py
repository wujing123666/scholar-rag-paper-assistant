"""Command-line entry point for the ScholarRAG paper retriever."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.libs.embedding.fastembed_embedding import FastEmbedEmbedding
from src.paper_assistant.catalog import PaperCatalog
from src.paper_assistant.dense_retriever import PaperDenseRetriever
from src.paper_assistant.evaluation import evaluate_retriever, load_evaluation_cases
from src.paper_assistant.hybrid_retriever import PaperHybridRetriever
from src.paper_assistant.inventory import build_paper_inventory, write_inventory_report
from src.paper_assistant.rejection import decide_retrieval, default_min_score
from src.paper_assistant.retriever import PaperBM25Retriever

DEFAULT_CATALOG = Path("data/papers/paper_catalog.csv")
DEFAULT_EVALUATION = Path("data/papers/eval_queries.jsonl")
DEFAULT_INBOX = Path("data/papers/inbox")
DEFAULT_PAPER_CHROMA = Path("data/db/chroma")
DEFAULT_PAPER_COLLECTION = "paper_profiles_v1"


DEFAULT_DENSE_MODEL = "BAAI/bge-small-zh-v1.5"


def _configure_stdout_utf8() -> None:
    """Keep Chinese JSON readable on Windows and in redirected output."""
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if callable(reconfigure):
        reconfigure(encoding="utf-8")


def _build_retriever(
    catalog_path: Path,
    retriever_name: str,
    model: str,
    model_cache: Path | None,
    *,
    chroma_mode: str,
    chroma_path: Path,
    chroma_host: str,
    chroma_port: int,
    chroma_ssl: bool,
) -> PaperBM25Retriever | PaperDenseRetriever | PaperHybridRetriever:
    catalog = PaperCatalog.from_csv(catalog_path)
    if retriever_name == "bm25":
        return PaperBM25Retriever(catalog)
    from src.libs.vector_store.chroma_store import ChromaStore

    embedding = FastEmbedEmbedding(model=model, cache_dir=model_cache)
    vector_store = ChromaStore(
        persist_directory=chroma_path,
        collection_name=DEFAULT_PAPER_COLLECTION,
        mode=chroma_mode,
        host=chroma_host,
        port=chroma_port,
        ssl=chroma_ssl,
    )
    dense = PaperDenseRetriever(catalog, embedding, vector_store)
    if retriever_name == "dense":
        return dense
    return PaperHybridRetriever(catalog, PaperBM25Retriever(catalog), dense)


def main() -> int:
    _configure_stdout_utf8()
    parser = argparse.ArgumentParser(description="ScholarRAG paper-level retrieval")
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument(
        "--retriever", choices=("bm25", "dense", "hybrid"), default="bm25"
    )
    parser.add_argument("--model", default=DEFAULT_DENSE_MODEL)
    parser.add_argument("--model-cache", type=Path)
    parser.add_argument("--chroma-mode", choices=("local", "server"), default="local")
    parser.add_argument("--chroma-path", type=Path, default=DEFAULT_PAPER_CHROMA)
    parser.add_argument("--chroma-host", default="localhost")
    parser.add_argument("--chroma-port", type=int, default=8000)
    parser.add_argument("--chroma-ssl", action="store_true")
    parser.add_argument(
        "--min-score",
        type=float,
        help="Override the retriever-specific unknown-paper rejection threshold.",
    )
    parser.add_argument(
        "--disable-rejection",
        action="store_true",
        help="Return raw rankings without the unknown-paper confidence gate.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    search_parser = subparsers.add_parser("search", help="Find papers from a description")
    search_parser.add_argument("query")
    search_parser.add_argument("--top-k", type=int, default=3)

    evaluate_parser = subparsers.add_parser("evaluate", help="Evaluate a paper retriever")
    evaluate_parser.add_argument("--queries", type=Path, default=DEFAULT_EVALUATION)
    evaluate_parser.add_argument(
        "--split", choices=("dev", "test_candidate", "dev_rejection")
    )

    inventory_parser = subparsers.add_parser(
        "inventory", help="Audit local PDFs against the paper catalog"
    )
    inventory_parser.add_argument("--inbox", type=Path, default=DEFAULT_INBOX)
    inventory_parser.add_argument("--output", type=Path)

    prepare_parser = subparsers.add_parser(
        "prepare-paper", help="Extract an evidence-backed Paper Profile draft"
    )
    prepare_parser.add_argument("pdf", type=Path)
    prepare_parser.add_argument("--inbox", type=Path, default=DEFAULT_INBOX)
    prepare_parser.add_argument("--output", type=Path)
    prepare_parser.add_argument("--max-pages", type=int, default=5)
    prepare_parser.add_argument("--force", action="store_true")

    args = parser.parse_args()
    if args.command == "inventory":
        report = build_paper_inventory(args.inbox, args.catalog)
        if args.output:
            write_inventory_report(report, args.output)
        print(json.dumps(report.summary_dict(), ensure_ascii=False, indent=2))
        return 0 if report.status == "ok" else 1

    if args.command == "prepare-paper":
        from src.paper_assistant.profile_draft import (
            prepare_profile_draft,
            write_profile_draft,
        )

        draft = prepare_profile_draft(
            args.pdf, args.inbox, args.catalog, max_pages=args.max_pages
        )
        output = args.output or Path("data/papers/drafts") / f"{args.pdf.stem}.json"
        write_profile_draft(draft, output, overwrite=args.force)
        fields = draft["extracted_fields"]
        print(
            json.dumps(
                {
                    "status": draft["status"],
                    "pdf_file": draft["file"]["pdf_file"],
                    "extracted": {
                        name: (
                            {
                                "value": fields[name]["value"],
                                "confidence": fields[name]["confidence"],
                            }
                            if fields[name]
                            else None
                        )
                        for name in ("canonical_title", "authors", "year", "doi")
                    },
                    "missing_fields": draft["missing_fields"],
                    "review_required_fields": draft["review_required_fields"],
                    "warnings": draft["warnings"],
                    "draft_file": str(output),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    retriever = _build_retriever(
        args.catalog,
        args.retriever,
        args.model,
        args.model_cache,
        chroma_mode=args.chroma_mode,
        chroma_path=args.chroma_path,
        chroma_host=args.chroma_host,
        chroma_port=args.chroma_port,
        chroma_ssl=args.chroma_ssl,
    )

    if args.command == "search":
        threshold = args.min_score
        if threshold is None and not args.disable_rejection:
            threshold = default_min_score(retriever.name)
        if threshold is None:
            results = retriever.search(args.query, top_k=args.top_k)
            rejected = not results
            top_score = results[0].score if results else None
        else:
            decision = decide_retrieval(
                retriever, args.query, top_k=args.top_k, min_score=threshold
            )
            results = list(decision.results)
            rejected = decision.rejected
            top_score = decision.top_score
        output = {
            "query": args.query,
            "retriever": args.retriever,
            "rejected": rejected,
            "top_score": round(top_score, 6) if top_score is not None else None,
            "min_score": threshold,
            "results": [
                {
                    "rank": rank,
                    "paper_id": result.paper.paper_id,
                    "title": result.paper.display_title,
                    "score": round(result.score, 4),
                    "matched_terms": result.matched_terms,
                    "pdf_files": result.paper.pdf_files,
                }
                for rank, result in enumerate(results, start=1)
            ],
        }
    else:
        threshold = args.min_score
        if threshold is None and not args.disable_rejection:
            threshold = default_min_score(retriever.name)
        output = evaluate_retriever(
            retriever,
            load_evaluation_cases(args.queries),
            split=args.split,
            min_score=threshold,
        )
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
