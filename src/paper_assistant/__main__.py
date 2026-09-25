"""Command-line entry point for the ScholarRAG paper retriever."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
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
DEFAULT_CHUNK_COLLECTION = "paper_chunks_v1"


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

    import_parser = subparsers.add_parser(
        "import-paper", help="Preview or confirm a reviewed Paper Profile import"
    )
    import_parser.add_argument("draft", type=Path)
    import_parser.add_argument("--inbox", type=Path, default=DEFAULT_INBOX)
    import_parser.add_argument(
        "--backup-dir", type=Path, default=Path("data/papers/backups")
    )
    import_parser.add_argument(
        "--confirm",
        action="store_true",
        help="Write the validated row after creating a catalog backup.",
    )

    chunk_parser = subparsers.add_parser(
        "search-chunks", help="Find page-level evidence inside candidate papers"
    )
    chunk_parser.add_argument("query")
    chunk_parser.add_argument("--inbox", type=Path, default=DEFAULT_INBOX)
    chunk_parser.add_argument("--paper-id", action="append", default=[])
    chunk_parser.add_argument("--candidate-papers", type=int, default=3)
    chunk_parser.add_argument("--top-k", type=int, default=5)
    chunk_parser.add_argument("--chunk-size", type=int, default=1200)
    chunk_parser.add_argument("--chunk-overlap", type=int, default=180)

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

    if args.command == "import-paper":
        from src.paper_assistant.catalog_import import (
            apply_catalog_import,
            plan_catalog_import,
        )

        try:
            plan = plan_catalog_import(args.draft, args.inbox, args.catalog)
            if not args.confirm or not plan.ready:
                print(
                    json.dumps(
                        {"status": "ready" if plan.ready else "blocked", **plan.summary_dict()},
                        ensure_ascii=False,
                        indent=2,
                    )
                )
                return 0 if plan.ready else 1
            result = apply_catalog_import(
                plan, args.inbox, args.catalog, args.backup_dir
            )
            print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
            return 0
        except (FileNotFoundError, ValueError, RuntimeError, json.JSONDecodeError) as error:
            print(
                json.dumps(
                    {"status": "error", "message": str(error)},
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 1

    if args.command == "search-chunks":
        from src.libs.vector_store.chroma_store import ChromaStore
        from src.paper_assistant.chunk_retriever import (
            PaperChunkRetriever,
            apply_paper_routing_prior,
        )

        if args.candidate_papers < 1:
            parser.error("--candidate-papers must be at least one")
        catalog = PaperCatalog.from_csv(args.catalog)
        embedding = FastEmbedEmbedding(model=args.model, cache_dir=args.model_cache)
        candidate_ids = tuple(dict.fromkeys(args.paper_id))
        automatic_routing = not candidate_ids
        if not candidate_ids:
            if args.retriever == "bm25":
                paper_retriever = PaperBM25Retriever(catalog)
            else:
                profile_store = ChromaStore(
                    persist_directory=args.chroma_path,
                    collection_name=DEFAULT_PAPER_COLLECTION,
                    mode=args.chroma_mode,
                    host=args.chroma_host,
                    port=args.chroma_port,
                    ssl=args.chroma_ssl,
                )
                dense = PaperDenseRetriever(catalog, embedding, profile_store)
                paper_retriever = (
                    dense
                    if args.retriever == "dense"
                    else PaperHybridRetriever(
                        catalog, PaperBM25Retriever(catalog), dense
                    )
                )
            candidate_ids = tuple(
                result.paper.paper_id
                for result in paper_retriever.search(
                    args.query, top_k=min(args.candidate_papers, len(catalog))
                )
            )
        if not candidate_ids:
            print(
                json.dumps(
                    {
                        "query": args.query,
                        "candidate_papers": [],
                        "results": [],
                        "message": "No candidate papers matched the query.",
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 0

        chunk_store = ChromaStore(
            persist_directory=args.chroma_path,
            collection_name=DEFAULT_CHUNK_COLLECTION,
            mode=args.chroma_mode,
            host=args.chroma_host,
            port=args.chroma_port,
            ssl=args.chroma_ssl,
        )
        chunk_retriever = PaperChunkRetriever(
            catalog,
            args.inbox,
            embedding,
            chunk_store,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
        )
        if automatic_routing:
            results = []
            for paper_id in candidate_ids:
                results.extend(
                    chunk_retriever.search(
                        args.query,
                        top_k=args.top_k,
                        paper_ids=(paper_id,),
                    )
                )
            results = apply_paper_routing_prior(results, candidate_ids)[: args.top_k]
        else:
            results = chunk_retriever.search(
                args.query,
                top_k=args.top_k,
                paper_ids=candidate_ids,
            )
        print(
            json.dumps(
                {
                    "query": args.query,
                    "paper_retriever": args.retriever,
                    "candidate_papers": list(candidate_ids),
                    "index_sync": asdict(chunk_retriever.index_sync),
                    "results": [
                        {
                            "rank": rank,
                            "chunk_id": result.chunk_id,
                            "score": round(result.score, 4),
                            "dense_score": round(result.dense_score, 4),
                            "sparse_score": round(result.sparse_score, 4),
                            "routing_rank": result.routing_rank or None,
                            "paper_id": result.paper.paper_id,
                            "paper_title": result.paper.display_title,
                            "pdf_file": result.pdf_file,
                            "page_number": result.page_number,
                            "section": result.section,
                            "text": result.text,
                        }
                        for rank, result in enumerate(results, start=1)
                    ],
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
