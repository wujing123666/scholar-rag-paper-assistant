"""Benchmark the completed IoT profile and chunk indexes without generating answers."""

from __future__ import annotations

import argparse
import csv
import ctypes
import json
import os
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from ctypes import wintypes
from pathlib import Path

from src.core.settings import load_settings
from src.libs.embedding.embedding_factory import EmbeddingFactory
from src.libs.vector_store.chroma_store import ChromaStore
from src.paper_assistant.catalog import PaperCatalog
from src.paper_assistant.chunk_retriever import PaperChunkRetriever
from src.paper_assistant.dense_retriever import PaperDenseRetriever
from src.paper_assistant.hybrid_retriever import PaperHybridRetriever
from src.paper_assistant.request_gate import RequestGate, RequestGateError
from src.paper_assistant.retriever import PaperBM25Retriever

TOPIC_QUERIES = {
    "core_iot": "物联网设备如何实现低功耗连接、设备管理和协议互操作？",
    "iot_security": "物联网设备认证、入侵检测和隐私保护有哪些方法？",
    "edge_iot": "物联网边缘计算中的任务卸载与资源分配如何实现？",
    "sensor_networks": "无线传感器网络如何优化覆盖、节点定位和能耗？",
    "industrial_iot": "工业物联网如何进行预测性维护和设备异常检测？",
    "smart_cities": "智慧城市如何利用物联网进行交通监测和城市治理？",
    "smart_agriculture": "智慧农业如何监测土壤、自动灌溉并识别作物病害？",
    "iot_healthcare": "医疗物联网如何实现患者监测和可穿戴健康感知？",
    "mobile_crowdsensing": "移动群智感知如何进行用户招募、激励和隐私保护？",
    "federated_iot": "物联网联邦学习如何处理非独立同分布和通信开销？",
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
        "--embedding-settings",
        type=Path,
        default=Path("config/settings.qwen.local.yaml"),
    )
    parser.add_argument(
        "--chroma-path", type=Path, default=Path("data/db/chroma_iot_scale")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "data/papers/corpus_iot_1000/benchmark/full_retrieval_benchmark.json"
        ),
    )
    return parser.parse_args()


def process_rss_mib() -> float | None:
    """Return current process RSS on Windows without an extra dependency."""
    if os.name != "nt":
        return None

    class ProcessMemoryCounters(ctypes.Structure):
        _fields_ = [
            ("cb", ctypes.c_ulong),
            ("PageFaultCount", ctypes.c_ulong),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
            ("PrivateUsage", ctypes.c_size_t),
        ]

    counters = ProcessMemoryCounters()
    counters.cb = ctypes.sizeof(counters)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    psapi.GetProcessMemoryInfo.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(ProcessMemoryCounters),
        wintypes.DWORD,
    ]
    psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
    process = kernel32.GetCurrentProcess()
    ok = psapi.GetProcessMemoryInfo(
        process, ctypes.byref(counters), counters.cb
    )
    return round(counters.WorkingSetSize / (1024 * 1024), 2) if ok else None


def latency_summary(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    p95_index = max(0, min(len(ordered) - 1, int(len(ordered) * 0.95 + 0.999) - 1))
    return {
        "mean_ms": round(statistics.mean(ordered), 1),
        "median_ms": round(statistics.median(ordered), 1),
        "p95_ms": round(ordered[p95_index], 1),
        "max_ms": round(max(ordered), 1),
    }


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args()
    rows = list(csv.DictReader(args.catalog.open(encoding="utf-8-sig")))
    topic_by_paper = {row["paper_id"]: row["topic_bucket"] for row in rows}
    first_by_topic = {}
    for row in rows:
        first_by_topic.setdefault(row["topic_bucket"], row)

    catalog = PaperCatalog.from_csv(args.catalog)
    settings = load_settings(args.embedding_settings)
    embedding = EmbeddingFactory.create(settings)
    profile_store = ChromaStore(
        persist_directory=args.chroma_path,
        collection_name="paper_profiles_qwen_v4",
    )
    chunk_store = ChromaStore(
        persist_directory=args.chroma_path,
        collection_name="paper_chunks_qwen_v4",
    )
    rss_before = process_rss_mib()
    started = time.perf_counter()
    dense = PaperDenseRetriever(catalog, embedding, profile_store)
    chunk_retriever = PaperChunkRetriever(
        catalog,
        args.pdf_dir,
        embedding,
        chunk_store,
        build_sparse_index=False,
    )
    sync_seconds = time.perf_counter() - started
    rss_after_sync = process_rss_mib()

    sparse_started = time.perf_counter()
    chunk_retriever._build_sparse_index()
    sparse_seconds = time.perf_counter() - sparse_started
    rss_after_sparse = process_rss_mib()
    paper_retriever = PaperHybridRetriever(
        catalog, PaperBM25Retriever(catalog), dense
    )

    exact_cases = []
    for topic, row in sorted(first_by_topic.items()):
        query = f"帮我找题目为 {row['canonical_title']} 的论文"
        case_started = time.perf_counter()
        results = paper_retriever.search(query, top_k=3)
        latency_ms = (time.perf_counter() - case_started) * 1000
        ranked_ids = [result.paper.paper_id for result in results]
        exact_cases.append(
            {
                "topic": topic,
                "query": query,
                "expected_paper_id": row["paper_id"],
                "rank": (
                    ranked_ids.index(row["paper_id"]) + 1
                    if row["paper_id"] in ranked_ids
                    else None
                ),
                "latency_ms": round(latency_ms, 1),
                "result_ids": ranked_ids,
            }
        )

    topic_cases = []
    for topic, query in TOPIC_QUERIES.items():
        route_started = time.perf_counter()
        papers = paper_retriever.search(query, top_k=3)
        route_ms = (time.perf_counter() - route_started) * 1000
        candidate_ids = tuple(result.paper.paper_id for result in papers)
        chunk_started = time.perf_counter()
        chunks = chunk_retriever.search(query, top_k=5, paper_ids=candidate_ids)
        chunk_ms = (time.perf_counter() - chunk_started) * 1000
        result_topics = [topic_by_paper.get(paper_id, "") for paper_id in candidate_ids]
        topic_cases.append(
            {
                "topic": topic,
                "query": query,
                "route_latency_ms": round(route_ms, 1),
                "chunk_latency_ms": round(chunk_ms, 1),
                "top1_bucket_match": bool(result_topics and result_topics[0] == topic),
                "top3_bucket_hit": topic in result_topics,
                "papers": [
                    {
                        "rank": rank,
                        "paper_id": result.paper.paper_id,
                        "title": result.paper.canonical_title,
                        "topic_bucket": topic_by_paper.get(result.paper.paper_id, ""),
                        "score": round(result.score, 6),
                    }
                    for rank, result in enumerate(papers, start=1)
                ],
                "chunks": [
                    {
                        "rank": rank,
                        "paper_id": result.paper.paper_id,
                        "page_number": result.page_number,
                        "section": result.section,
                        "score": round(result.score, 6),
                        "text_preview": result.text[:240],
                    }
                    for rank, result in enumerate(chunks, start=1)
                ],
            }
        )

    # Repeat every warmed topic query twice and release all callers together.
    # This measures the shared single-process retrievers plus the same 8-worker
    # admission gate used by the Streamlit search page, without conflating the
    # result with first-call model or index startup time.
    gate = RequestGate(
        kind="scale_benchmark",
        max_workers=8,
        max_queue_size=20,
        max_queue_wait_seconds=60,
    )

    def retrieve(query: str) -> tuple[int, int]:
        papers = paper_retriever.search(query, top_k=3)
        candidate_ids = tuple(result.paper.paper_id for result in papers)
        chunks = chunk_retriever.search(query, top_k=5, paper_ids=candidate_ids)
        return len(papers), len(chunks)

    concurrent_started = time.perf_counter()
    concurrent_results = []
    concurrent_errors = []
    requests = [
        (f"{topic}-{repeat}", query)
        for repeat in range(2)
        for topic, query in TOPIC_QUERIES.items()
    ]
    with ThreadPoolExecutor(max_workers=len(requests)) as callers:
        futures = {
            callers.submit(gate.run, retrieve, query): request_id
            for request_id, query in requests
        }
        for future in as_completed(futures):
            request_id = futures[future]
            try:
                execution = future.result()
                paper_count, chunk_count = execution.value
                concurrent_results.append(
                    {
                        "request_id": request_id,
                        "paper_count": paper_count,
                        "chunk_count": chunk_count,
                        "queue_wait_ms": round(
                            execution.record.queue_wait_seconds * 1000, 1
                        ),
                        "execution_ms": round(
                            execution.record.execution_seconds * 1000, 1
                        ),
                        "total_ms": round(execution.record.total_seconds * 1000, 1),
                    }
                )
            except RequestGateError as error:
                concurrent_errors.append(
                    {
                        "request_id": request_id,
                        "status": error.record.status,
                        "error_type": error.record.error_type,
                    }
                )
            except Exception as error:  # pragma: no cover - benchmark diagnostic
                concurrent_errors.append(
                    {
                        "request_id": request_id,
                        "status": "caller_error",
                        "error_type": type(error).__name__,
                    }
                )
    concurrent_wall_seconds = time.perf_counter() - concurrent_started
    gate_snapshot = gate.snapshot()
    gate.shutdown()

    report = {
        "scope": "scale smoke benchmark; topic buckets and exact titles are not blind gold labels",
        "papers": len(catalog),
        "chunks": chunk_retriever.index_sync.chunks,
        "profile_sync": vars(dense.index_sync),
        "chunk_sync": vars(chunk_retriever.index_sync),
        "startup": {
            "sync_seconds": round(sync_seconds, 2),
            "sparse_build_seconds": round(sparse_seconds, 2),
            "total_seconds": round(sync_seconds + sparse_seconds, 2),
            "rss_before_mib": rss_before,
            "rss_after_sync_mib": rss_after_sync,
            "rss_after_sparse_mib": rss_after_sparse,
        },
        "exact_title": {
            "cases": len(exact_cases),
            "recall_at_1": round(
                sum(case["rank"] == 1 for case in exact_cases) / len(exact_cases), 4
            ),
            "recall_at_3": round(
                sum(case["rank"] is not None for case in exact_cases)
                / len(exact_cases),
                4,
            ),
            "latency": latency_summary(
                [float(case["latency_ms"]) for case in exact_cases]
            ),
            "details": exact_cases,
        },
        "topic_smoke": {
            "cases": len(topic_cases),
            "top1_bucket_accuracy": round(
                sum(case["top1_bucket_match"] for case in topic_cases)
                / len(topic_cases),
                4,
            ),
            "top3_bucket_recall": round(
                sum(case["top3_bucket_hit"] for case in topic_cases)
                / len(topic_cases),
                4,
            ),
            "nonempty_chunk_results": sum(bool(case["chunks"]) for case in topic_cases),
            "route_latency": latency_summary(
                [float(case["route_latency_ms"]) for case in topic_cases]
            ),
            "chunk_latency": latency_summary(
                [float(case["chunk_latency_ms"]) for case in topic_cases]
            ),
            "details": topic_cases,
        },
        "warm_concurrency": {
            "requests": len(requests),
            "gate_workers": 8,
            "gate_queue": 20,
            "completed": len(concurrent_results),
            "failed": len(concurrent_errors),
            "wall_seconds": round(concurrent_wall_seconds, 2),
            "latency": (
                latency_summary(
                    [float(result["total_ms"]) for result in concurrent_results]
                )
                if concurrent_results
                else None
            ),
            "max_queue_wait_ms": (
                round(
                    max(result["queue_wait_ms"] for result in concurrent_results),
                    1,
                )
                if concurrent_results
                else None
            ),
            "snapshot": gate_snapshot.to_dict(),
            "errors": concurrent_errors,
            "details": sorted(
                concurrent_results, key=lambda result: result["request_id"]
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    profile_store.close()
    chunk_store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
