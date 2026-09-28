"""Build the audited IoT corpus chunk index with resumable progress reporting."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

from src.core.settings import load_settings
from src.libs.embedding.embedding_factory import EmbeddingFactory
from src.libs.vector_store.chroma_store import ChromaStore
from src.paper_assistant.catalog import PaperCatalog
from src.paper_assistant.chunk_retriever import ChunkIndexProgress, PaperChunkRetriever


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
        "--embedding-settings",
        type=Path,
        default=Path("config/settings.qwen.local.yaml"),
    )
    parser.add_argument(
        "--chroma-path",
        type=Path,
        default=Path("data/db/chroma_iot_scale"),
    )
    parser.add_argument("--collection", default="paper_chunks_qwen_v4")
    parser.add_argument(
        "--progress-file",
        type=Path,
        default=Path("data/papers/corpus_iot_1000/index/chunk_index_progress.json"),
    )
    parser.add_argument("--chunk-size", type=int, default=1200)
    parser.add_argument("--chunk-overlap", type=int, default=180)
    parser.add_argument(
        "--embedding-workers",
        type=int,
        default=4,
        help="Concurrent Qwen request batches; token usage is unchanged",
    )
    return parser.parse_args()


def write_progress(path: Path, payload: dict) -> None:
    """Atomically replace the local progress snapshot."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def main() -> int:
    args = parse_args()
    catalog = PaperCatalog.from_csv(args.catalog)
    settings = load_settings(args.embedding_settings)
    if settings.embedding.provider.casefold() != "qwen":
        raise ValueError("The scale index must use embedding.provider=qwen")
    if args.embedding_workers < 1:
        raise ValueError("embedding-workers must be at least one")
    embedding = EmbeddingFactory.create(
        settings,
        batch_size=10,
        max_workers=args.embedding_workers,
    )
    store = ChromaStore(
        persist_directory=args.chroma_path,
        collection_name=args.collection,
    )
    started = time.monotonic()
    last_payload = {
        "status": "starting",
        "completed_papers": 0,
        "total_papers": len(catalog),
        "stored_chunks_before_run": store.get_collection_stats()["count"],
    }
    write_progress(args.progress_file, last_payload)

    def report(progress: ChunkIndexProgress) -> None:
        nonlocal last_payload
        elapsed = time.monotonic() - started
        rate = progress.run_completed_papers / elapsed if elapsed else 0.0
        remaining = progress.run_total_papers - progress.run_completed_papers
        eta_seconds = remaining / rate if rate else None
        last_payload = {
            "status": "running",
            **asdict(progress),
            "elapsed_seconds": round(elapsed, 1),
            "estimated_remaining_seconds": (
                round(eta_seconds, 1) if eta_seconds is not None else None
            ),
        }
        write_progress(args.progress_file, last_payload)
        print(
            f"indexed={progress.completed_papers}/{progress.total_papers} "
            f"paper={progress.paper_id} paper_chunks={progress.paper_chunks} "
            f"embedded_chunks={progress.embedded_chunks} "
            f"reused_chunks={progress.reused_chunks} "
            f"elapsed_s={elapsed:.1f} eta_s={eta_seconds:.1f}",
            flush=True,
        )

    try:
        retriever = PaperChunkRetriever(
            catalog,
            args.pdf_dir,
            embedding,
            store,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            progress_callback=report,
            build_sparse_index=False,
        )
        elapsed = time.monotonic() - started
        final_payload = {
            "status": "completed",
            **asdict(retriever.index_sync),
            "elapsed_seconds": round(elapsed, 1),
            "collection": args.collection,
            "chroma_path": str(args.chroma_path),
        }
        write_progress(args.progress_file, final_payload)
        print(json.dumps(final_payload, ensure_ascii=False, indent=2), flush=True)
        return 0
    except KeyboardInterrupt:
        write_progress(
            args.progress_file,
            {**last_payload, "status": "interrupted", "error": "KeyboardInterrupt"},
        )
        raise
    except Exception as error:
        write_progress(
            args.progress_file,
            {
                **last_payload,
                "status": "failed",
                "error": f"{type(error).__name__}: {error}",
            },
        )
        raise
    finally:
        store.close()


if __name__ == "__main__":
    raise SystemExit(main())
