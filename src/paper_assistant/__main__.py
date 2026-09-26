"""Command-line entry point for the ScholarRAG paper retriever."""

from __future__ import annotations

import argparse
import json
import sys
import time
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
DEFAULT_CHUNK_EVALUATION = Path("data/papers/chunk_eval_queries.jsonl")
DEFAULT_ANSWER_EVALUATION = Path("data/papers/answer_eval_queries.jsonl")
DEFAULT_INBOX = Path("data/papers/inbox")
DEFAULT_PAPER_CHROMA = Path("data/db/chroma")
DEFAULT_PAPER_COLLECTION = "paper_profiles_v1"
DEFAULT_CHUNK_COLLECTION = "paper_chunks_v1"


DEFAULT_DENSE_MODEL = "BAAI/bge-small-zh-v1.5"
DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-base"


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
        "--chunk-reranker", choices=("none", "fastembed"), default="none"
    )
    parser.add_argument("--reranker-model", default=DEFAULT_RERANKER_MODEL)
    parser.add_argument(
        "--reranker-cache", type=Path, default=Path("data/models/fastembed")
    )
    parser.add_argument("--rerank-candidates", type=int, default=14)
    parser.add_argument("--rerank-weight", type=float, default=0.35)
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

    answer_parser = subparsers.add_parser(
        "answer", help="Answer a question from cited paper chunks"
    )
    answer_parser.add_argument("query")
    answer_parser.add_argument("--inbox", type=Path, default=DEFAULT_INBOX)
    answer_parser.add_argument("--paper-id", action="append", default=[])
    answer_parser.add_argument("--candidate-papers", type=int, default=3)
    answer_parser.add_argument("--top-k", type=int, default=7)
    answer_parser.add_argument("--chunk-size", type=int, default=1200)
    answer_parser.add_argument("--chunk-overlap", type=int, default=180)
    answer_parser.add_argument(
        "--settings", type=Path, default=Path("config/settings.yaml")
    )
    answer_parser.add_argument("--max-context-chars", type=int, default=16000)
    answer_parser.add_argument("--min-evidence-chunks", type=int, default=1)

    chunk_evaluate_parser = subparsers.add_parser(
        "evaluate-chunks", help="Evaluate page-level evidence retrieval"
    )
    chunk_evaluate_parser.add_argument(
        "--queries", type=Path, default=DEFAULT_CHUNK_EVALUATION
    )
    chunk_evaluate_parser.add_argument("--inbox", type=Path, default=DEFAULT_INBOX)
    chunk_evaluate_parser.add_argument("--candidate-papers", type=int, default=3)
    chunk_evaluate_parser.add_argument("--top-k", type=int, default=5)
    chunk_evaluate_parser.add_argument("--chunk-size", type=int, default=1200)
    chunk_evaluate_parser.add_argument("--chunk-overlap", type=int, default=180)
    chunk_evaluate_parser.add_argument(
        "--oracle-paper",
        action="store_true",
        help="Evaluate chunk ranking inside the known target paper only.",
    )

    answer_evaluate_parser = subparsers.add_parser(
        "evaluate-answers", help="Evaluate routed, cited answers against PDF labels"
    )
    answer_evaluate_parser.add_argument(
        "--queries", type=Path, default=DEFAULT_ANSWER_EVALUATION
    )
    answer_evaluate_parser.add_argument("--inbox", type=Path, default=DEFAULT_INBOX)
    answer_evaluate_parser.add_argument("--candidate-papers", type=int, default=3)
    answer_evaluate_parser.add_argument("--top-k", type=int, default=7)
    answer_evaluate_parser.add_argument("--chunk-size", type=int, default=1200)
    answer_evaluate_parser.add_argument("--chunk-overlap", type=int, default=180)
    answer_evaluate_parser.add_argument(
        "--settings", type=Path, default=Path("config/settings.yaml")
    )
    answer_evaluate_parser.add_argument(
        "--judge-model",
        help="Use a different model from the same configured provider for judging",
    )
    answer_evaluate_parser.add_argument("--max-context-chars", type=int, default=16000)
    answer_evaluate_parser.add_argument("--min-evidence-chunks", type=int, default=1)
    answer_evaluate_parser.add_argument(
        "--output", type=Path, default=Path("tmp/answer_evaluation_report.json")
    )
    answer_evaluate_parser.add_argument("--limit", type=int)
    answer_evaluate_parser.add_argument("--resume", action="store_true")

    compare_judges_parser = subparsers.add_parser(
        "compare-judges",
        help="Replay multiple judges over identical frozen answers and evidence",
    )
    compare_judges_parser.add_argument("--input", type=Path, required=True)
    compare_judges_parser.add_argument(
        "--settings", type=Path, default=Path("config/settings.yaml")
    )
    compare_judges_parser.add_argument(
        "--judge-model",
        action="append",
        required=True,
        help="Judge model to run; provide this option at least twice",
    )
    compare_judges_parser.add_argument(
        "--output", type=Path, default=Path("tmp/judge_comparison.json")
    )
    compare_judges_parser.add_argument(
        "--audit-output", type=Path, default=Path("tmp/claim_audit.md")
    )
    compare_judges_parser.add_argument("--agreement-sample", type=int, default=20)
    compare_judges_parser.add_argument("--seed", type=int, default=20260926)

    score_audit_parser = subparsers.add_parser(
        "score-judge-audit",
        help="Score frozen-answer judges against attributed audit labels",
    )
    score_audit_parser.add_argument("--comparison", type=Path, required=True)
    score_audit_parser.add_argument("--labels", type=Path, required=True)
    score_audit_parser.add_argument(
        "--output", type=Path, default=Path("tmp/judge_audit_score.json")
    )

    args = parser.parse_args()
    if args.command == "score-judge-audit":
        from src.paper_assistant.judge_comparison import (
            load_audit_labels,
            score_audit_labels,
            write_comparison,
        )

        comparison = json.loads(args.comparison.read_text(encoding="utf-8"))
        score = score_audit_labels(comparison, load_audit_labels(args.labels))
        write_comparison(args.output, score)
        print(
            json.dumps(
                {"output": str(args.output), **score["summary"]},
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    if args.command == "compare-judges":
        from src.core.settings import load_settings
        from src.libs.llm import LLMFactory
        from src.paper_assistant.judge_comparison import (
            compare_frozen_judges,
            load_frozen_report,
            select_audit_claims,
            write_audit_markdown,
            write_comparison,
        )

        if args.agreement_sample < 0:
            parser.error("--agreement-sample cannot be negative")
        settings = load_settings(args.settings)
        report = load_frozen_report(args.input)
        comparison = compare_frozen_judges(
            LLMFactory.create(settings),
            report,
            args.judge_model,
            disable_thinking=settings.llm.provider == "deepseek",
        )
        audit_claims = select_audit_claims(
            comparison,
            agreement_sample=args.agreement_sample,
            seed=args.seed,
        )
        write_comparison(args.output, comparison)
        write_audit_markdown(args.audit_output, audit_claims)
        print(
            json.dumps(
                {
                    "output": str(args.output),
                    "audit_output": str(args.audit_output),
                    "audit_claims": len(audit_claims),
                    **comparison["summary"],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
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

    if args.command in {
        "search-chunks",
        "answer",
        "evaluate-chunks",
        "evaluate-answers",
    }:
        from src.libs.vector_store.chroma_store import ChromaStore
        from src.paper_assistant.chunk_retriever import (
            PaperChunkRetriever,
            apply_paper_routing_prior,
        )

        if args.candidate_papers < 1:
            parser.error("--candidate-papers must be at least one")
        if args.rerank_candidates < 1:
            parser.error("--rerank-candidates must be at least one")
        if not 0 <= args.rerank_weight <= 1:
            parser.error("--rerank-weight must be between zero and one")
        if args.command in {"answer", "evaluate-answers"}:
            if args.max_context_chars < 1000:
                parser.error("--max-context-chars must be at least 1000")
            if args.min_evidence_chunks < 1:
                parser.error("--min-evidence-chunks must be at least one")
        if args.command == "evaluate-answers" and args.limit is not None and args.limit < 1:
            parser.error("--limit must be at least one")
        catalog = PaperCatalog.from_csv(args.catalog)
        embedding = FastEmbedEmbedding(model=args.model, cache_dir=args.model_cache)
        paper_retriever = None
        needs_paper_router = args.command == "evaluate-chunks" and not args.oracle_paper
        if args.command == "evaluate-answers":
            needs_paper_router = True
        if args.command in {"search-chunks", "answer"}:
            needs_paper_router = not args.paper_id
        if needs_paper_router:
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

        def resolve_candidates(query: str) -> tuple[str, ...]:
            if paper_retriever is None:
                return ()
            return tuple(
                result.paper.paper_id
                for result in paper_retriever.search(
                    query, top_k=min(args.candidate_papers, len(catalog))
                )
            )

        if args.command in {"search-chunks", "answer"}:
            candidate_ids = tuple(dict.fromkeys(args.paper_id))
            automatic_routing = not candidate_ids
            if automatic_routing:
                if args.command == "answer" and not args.disable_rejection:
                    if paper_retriever is None:
                        candidate_ids = ()
                    else:
                        threshold = args.min_score
                        if threshold is None:
                            threshold = default_min_score(paper_retriever.name)
                        decision = decide_retrieval(
                            paper_retriever,
                            args.query,
                            top_k=min(args.candidate_papers, len(catalog)),
                            min_score=threshold,
                        )
                        candidate_ids = tuple(
                            result.paper.paper_id for result in decision.results
                        )
                else:
                    candidate_ids = resolve_candidates(args.query)
        else:
            candidate_ids = ()
            automatic_routing = False

        if args.command in {"search-chunks", "answer"} and not candidate_ids:
            if args.command == "answer":
                from src.paper_assistant.grounded_answer import REFUSAL_TEXT

                print(
                    json.dumps(
                        {
                            "query": args.query,
                            "status": "insufficient_evidence",
                            "answer": REFUSAL_TEXT,
                            "claims": [],
                            "citations": [],
                            "reason": "no_candidate_papers",
                            "candidate_papers": [],
                        },
                        ensure_ascii=False,
                        indent=2,
                    )
                )
                return 0
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
        index_sync = chunk_retriever.index_sync
        chunk_reranker = None
        if args.chunk_reranker == "fastembed":
            from src.paper_assistant.chunk_reranker import FastEmbedChunkReranker

            chunk_reranker = FastEmbedChunkReranker(
                args.reranker_model,
                cache_dir=args.reranker_cache,
                weight=args.rerank_weight,
            )
        if args.command == "evaluate-answers":
            from src.core.settings import load_settings
            from src.libs.llm import LLMFactory
            from src.paper_assistant.answer_evaluation import (
                ClaimSupportJudgement,
                FactJudgement,
                build_answer_case_result,
                judge_claim_support,
                judge_required_facts,
                load_answer_evaluation_cases,
                summarize_answer_results,
                write_answer_evaluation_report,
            )
            from src.paper_assistant.chunk_reranker import (
                rerank_with_fallback,
                select_rerank_candidates,
            )
            from src.paper_assistant.grounded_answer import answer_from_evidence

            cases = load_answer_evaluation_cases(args.queries)
            if args.limit is not None:
                cases = cases[: args.limit]
            llm_settings = load_settings(args.settings)
            llm = LLMFactory.create(llm_settings)
            disable_judge_thinking = bool(
                args.judge_model and llm_settings.llm.provider == "deepseek"
            )
            run_config = {
                "paper_retriever": args.retriever,
                "chunk_reranker": args.chunk_reranker,
                "reranker_model": (
                    args.reranker_model if args.chunk_reranker != "none" else None
                ),
                "rerank_candidates": args.rerank_candidates,
                "candidate_papers": args.candidate_papers,
                "top_k": args.top_k,
                "queries": str(args.queries),
                "claim_support_judge": True,
                "frozen_evidence_schema": 1,
                "judge_model": args.judge_model,
                "judge_thinking": (
                    "disabled" if disable_judge_thinking else None
                ),
            }
            completed: dict[str, dict[str, object]] = {}
            if args.resume and args.output.exists():
                previous = json.loads(args.output.read_text(encoding="utf-8"))
                if previous.get("config") != run_config:
                    raise ValueError(
                        "Cannot resume an answer evaluation with different settings"
                    )
                completed = {
                    result["id"]: result for result in previous.get("results", [])
                }
            results = []
            retrieval_k = (
                max(args.top_k, args.rerank_candidates)
                if chunk_reranker
                else args.top_k
            )
            threshold = args.min_score
            if threshold is None and not args.disable_rejection:
                threshold = default_min_score(paper_retriever.name)

            for index, case in enumerate(cases, start=1):
                if case["id"] in completed:
                    previous_result = completed[case["id"]]
                    if any(
                        previous_result.get(field) != case[field]
                        for field in (
                            "question",
                            "expected_paper_id",
                            "relevant_pages",
                            "reference_answer",
                            "required_facts",
                        )
                    ):
                        raise ValueError(
                            f"Cannot resume changed evaluation case {case['id']}"
                        )
                    results.append(previous_result)
                    continue
                started = time.perf_counter()
                if threshold is None:
                    candidate_ids = resolve_candidates(case["question"])
                else:
                    decision = decide_retrieval(
                        paper_retriever,
                        case["question"],
                        top_k=min(args.candidate_papers, len(catalog)),
                        min_score=threshold,
                    )
                    candidate_ids = tuple(
                        result.paper.paper_id for result in decision.results
                    )
                matches = []
                for paper_id in candidate_ids:
                    matches.extend(
                        chunk_retriever.search(
                            case["question"],
                            top_k=retrieval_k,
                            paper_ids=(paper_id,),
                        )
                    )
                matches = apply_paper_routing_prior(matches, candidate_ids)
                reranker_fallback = None
                if chunk_reranker:
                    matches = select_rerank_candidates(
                        matches, top_k=retrieval_k
                    )
                    matches, reranker_fallback = rerank_with_fallback(
                        chunk_reranker,
                        case["question"],
                        matches,
                        top_k=args.top_k,
                    )
                else:
                    matches = matches[: args.top_k]
                answer = answer_from_evidence(
                    llm,
                    case["question"],
                    matches,
                    max_context_chars=args.max_context_chars,
                    min_evidence_chunks=args.min_evidence_chunks,
                )
                judgement = (
                    judge_required_facts(
                        llm,
                        answer.answer,
                        case["required_facts"],
                        model=args.judge_model,
                        disable_thinking=disable_judge_thinking,
                    )
                    if answer.status == "answered"
                    else FactJudgement((), None, None, "answer_not_generated")
                )
                claim_judgement = (
                    judge_claim_support(
                        llm,
                        answer,
                        matches,
                        model=args.judge_model,
                        disable_thinking=disable_judge_thinking,
                    )
                    if answer.status == "answered"
                    else ClaimSupportJudgement(
                        (), None, None, "answer_not_generated"
                    )
                )
                result = build_answer_case_result(
                    case,
                    candidate_ids,
                    answer,
                    judgement,
                    claim_judgement,
                    latency_seconds=time.perf_counter() - started,
                    reranker_fallback=reranker_fallback,
                    evidence_results=matches,
                )
                results.append(result)
                report = {
                    "config": run_config,
                    "summary": summarize_answer_results(results),
                    "results": results,
                }
                write_answer_evaluation_report(args.output, report)
                print(
                    f"[{index}/{len(cases)}] {case['id']}: {answer.status}",
                    file=sys.stderr,
                )

            report = {
                "config": run_config,
                "summary": summarize_answer_results(results),
                "results": results,
            }
            write_answer_evaluation_report(args.output, report)
            print(
                json.dumps(
                    {"output": str(args.output), **report["summary"]},
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 0
        if args.command == "evaluate-chunks":
            from src.paper_assistant.chunk_evaluation import (
                evaluate_chunk_retriever,
                load_chunk_evaluation_cases,
            )

            report = evaluate_chunk_retriever(
                chunk_retriever,
                load_chunk_evaluation_cases(args.queries),
                top_k=args.top_k,
                candidate_resolver=None if args.oracle_paper else resolve_candidates,
                reranker=chunk_reranker,
                rerank_candidates=args.rerank_candidates,
            )
            report["paper_retriever"] = (
                "oracle" if args.oracle_paper else args.retriever
            )
            report["candidate_papers"] = (
                1 if args.oracle_paper else args.candidate_papers
            )
            report["chunk_reranker"] = args.chunk_reranker
            report["reranker_model"] = (
                args.reranker_model if args.chunk_reranker != "none" else None
            )
            report["index_sync"] = asdict(index_sync)
            print(json.dumps(report, ensure_ascii=False, indent=2))
            return 0

        retrieval_k = (
            max(args.top_k, args.rerank_candidates)
            if chunk_reranker
            else args.top_k
        )
        reranker_fallback = None
        if automatic_routing:
            results = []
            for paper_id in candidate_ids:
                results.extend(
                    chunk_retriever.search(
                        args.query,
                        top_k=retrieval_k,
                        paper_ids=(paper_id,),
                    )
                )
            results = apply_paper_routing_prior(results, candidate_ids)
        else:
            results = chunk_retriever.search(
                args.query,
                top_k=retrieval_k,
                paper_ids=candidate_ids,
            )
        if chunk_reranker:
            from src.paper_assistant.chunk_reranker import rerank_with_fallback

            results = results[:retrieval_k]
            results, reranker_fallback = rerank_with_fallback(
                chunk_reranker, args.query, results, top_k=args.top_k
            )
        else:
            results = results[: args.top_k]
        retrieval_payload = {
            "query": args.query,
            "paper_retriever": args.retriever,
            "chunk_reranker": args.chunk_reranker,
            "reranker_model": (
                args.reranker_model if args.chunk_reranker != "none" else None
            ),
            "reranker_fallback": reranker_fallback,
            "candidate_papers": list(candidate_ids),
            "index_sync": asdict(index_sync),
        }
        if args.command == "answer":
            from src.core.settings import load_settings
            from src.libs.llm import LLMFactory
            from src.paper_assistant.grounded_answer import answer_from_evidence

            try:
                llm = LLMFactory.create(load_settings(args.settings))
                answer = answer_from_evidence(
                    llm,
                    args.query,
                    results,
                    max_context_chars=args.max_context_chars,
                    min_evidence_chunks=args.min_evidence_chunks,
                )
            except (OSError, RuntimeError, ValueError):
                print(
                    json.dumps(
                        {
                            **retrieval_payload,
                            "status": "error",
                            "message": (
                                "LLM 配置或服务不可用，请检查 --settings 指向的私有配置"
                                "以及对应服务端日志。"
                            ),
                        },
                        ensure_ascii=False,
                        indent=2,
                    )
                )
                return 1
            print(
                json.dumps(
                    {**retrieval_payload, **answer.to_dict()},
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 0

        print(
            json.dumps(
                {
                    **retrieval_payload,
                    "results": [
                        {
                            "rank": rank,
                            "chunk_id": result.chunk_id,
                            "score": round(result.score, 4),
                            "dense_score": round(result.dense_score, 4),
                            "sparse_score": round(result.sparse_score, 4),
                            "rerank_score": (
                                round(result.rerank_score, 4)
                                if result.rerank_score is not None
                                else None
                            ),
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
