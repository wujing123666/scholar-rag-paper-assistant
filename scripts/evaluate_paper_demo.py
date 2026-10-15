"""Opt-in live Qwen retrieval evaluation on the public synthetic PDFs (uses API quota).

This uses the production search gate, chunk retriever and ranking metrics.
It measures retrieval, not generated-answer quality.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.check_paper_demo import DEMO, ROOT
from src.core.settings import load_settings
from src.libs.embedding.embedding_factory import EmbeddingFactory
from src.libs.vector_store.chroma_store import ChromaStore
from src.paper_assistant.catalog import PaperCatalog
from src.paper_assistant.chunk_evaluation import (
    evaluate_chunk_retriever,
    load_chunk_evaluation_cases,
)
from src.paper_assistant.chunk_retriever import PaperChunkRetriever
from src.paper_assistant.evaluation import evaluate_search_service, load_evaluation_cases
from src.paper_assistant.service import (
    DEFAULT_CHUNK_COLLECTION,
    PaperAssistantConfig,
    build_paper_search_service,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--settings", type=Path, default=DEMO / "settings.yaml")
    parser.add_argument("--output", type=Path, default=ROOT / "tmp/paper_demo/live_retrieval.json")
    args = parser.parse_args()
    settings = load_settings(args.settings)
    if settings.embedding.provider.casefold() != "qwen":
        raise ValueError("Demo live retrieval requires Qwen, matching production")
    config = PaperAssistantConfig(
        catalog_path=DEMO / "paper_catalog.csv",
        inbox_path=DEMO / "pdfs",
        settings_path=args.settings,
        chroma_path=ROOT / "tmp/paper_demo/chroma",
        retriever="hybrid",
        paper_reranker="none",
        candidate_papers=1,
        top_k=3,
    )
    catalog = PaperCatalog.from_csv(config.catalog_path)
    service = build_paper_search_service(config)
    chunks = PaperChunkRetriever(
        catalog,
        config.inbox_path,
        EmbeddingFactory.create(settings),
        ChromaStore(persist_directory=config.chroma_path, collection_name=DEFAULT_CHUNK_COLLECTION),
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
    )

    def candidates(query: str) -> tuple[str, ...]:
        return tuple(item.paper.paper_id for item in service.find_papers(query, top_k=1).results)

    report = {
        "purpose": "synthetic_development_retrieval",
        "config": {
            "embedding": settings.embedding.model,
            "dimensions": settings.embedding.dimensions,
            "paper_retriever": "hybrid",
            "paper_reranker": "none",
            "chunk_reranker": "none",
            "candidate_papers": 1,
            "top_k": 3,
            "chunk_size": config.chunk_size,
            "chunk_overlap": config.chunk_overlap,
        },
        "paper": evaluate_search_service(
            service, load_evaluation_cases(DEMO / "paper_queries.jsonl")
        ),
        "chunk": evaluate_chunk_retriever(
            chunks,
            load_chunk_evaluation_cases(DEMO / "chunk_queries.jsonl"),
            top_k=3,
            candidate_resolver=candidates,
        ),
        "limits": "Small synthetic same-source development set; not an independent benchmark. No answer generation or hallucination measurement.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
