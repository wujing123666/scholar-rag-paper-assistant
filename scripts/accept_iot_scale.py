"""Run production-path acceptance checks against the 676-paper IoT corpus."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path

from scripts.benchmark_iot_retrieval import TOPIC_QUERIES, process_rss_mib
from src.paper_assistant.request_gate import RequestGate, RequestGateError
from src.paper_assistant.service import (
    PaperAnswerResponse,
    PaperAssistantConfig,
    build_paper_assistant_service,
)

SECOND_QUERY_PHRASINGS = {
    "core_iot": "请结合相关论文，说明物联网节点怎样兼顾低功耗通信、设备运维以及消息协议互操作。",
    "iot_security": "请比较物联网中的身份认证、入侵检测与隐私防护技术。",
    "edge_iot": "边缘计算场景下，物联网任务应如何卸载并进行计算资源调度？",
    "sensor_networks": "无线传感器网络有哪些覆盖增强、节点定位和节能路由方法？",
    "industrial_iot": "工业物联网怎样利用传感数据开展预测维护与异常识别？",
    "smart_cities": "相关研究如何使用物联网支持智慧交通监控和城市治理？",
    "smart_agriculture": "物联网农业系统如何完成土壤感知、灌溉控制与病害检测？",
    "iot_healthcare": "医疗物联网有哪些患者连续监护和可穿戴健康感知方案？",
    "mobile_crowdsensing": "移动群智感知研究怎样处理参与者招募、激励设计和隐私保护？",
    "federated_iot": "物联网联邦学习如何缓解Non-IID数据与通信成本问题？",
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
        "--chroma-path",
        type=Path,
        default=Path("data/db/chroma_iot_scale"),
    )
    parser.add_argument(
        "--llm-settings",
        type=Path,
        default=Path("config/settings.deepseek.local.yaml"),
    )
    parser.add_argument(
        "--embedding-settings",
        type=Path,
        default=Path("config/settings.qwen.local.yaml"),
    )
    parser.add_argument(
        "--fallback-settings",
        type=Path,
        default=Path("config/settings.qwen.local.yaml"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "data/papers/corpus_iot_1000/benchmark/production_acceptance.json"
        ),
    )
    parser.add_argument("--requests", type=int, default=20)
    parser.add_argument("--answer-workers", type=int, default=3)
    parser.add_argument("--queue-size", type=int, default=20)
    parser.add_argument("--queue-wait-seconds", type=float, default=180.0)
    parser.add_argument(
        "--skip-verified",
        action="store_true",
        help=(
            "Skip the live dual-model Claim check and run only the queued "
            "concurrency acceptance."
        ),
    )
    return parser.parse_args()


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int(len(ordered) * fraction + 0.999) - 1))
    return round(ordered[index], 1)


def latency_summary(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean_ms": None, "median_ms": None, "p95_ms": None, "max_ms": None}
    return {
        "mean_ms": round(statistics.mean(values), 1),
        "median_ms": round(statistics.median(values), 1),
        "p95_ms": percentile(values, 0.95),
        "max_ms": round(max(values), 1),
    }


def classify_answer(response: PaperAnswerResponse) -> tuple[str, str | None]:
    return response.answer.status, response.answer.reason


def build_requests(count: int) -> list[tuple[str, str]]:
    """Build distinct requests while keeping repeated topic families comparable."""
    topics = list(TOPIC_QUERIES.items())
    requests = []
    for index in range(count):
        topic, original_query = topics[index % len(topics)]
        repeat = index // len(topics)
        query = original_query if repeat == 0 else SECOND_QUERY_PHRASINGS[topic]
        if repeat > 1:
            query = f"{query} 请作为第{repeat + 1}组独立请求作答。"
        requests.append((f"{topic}-{repeat + 1}", query))
    return requests


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args()
    if args.requests < 1 or args.answer_workers < 1 or args.queue_size < 0:
        raise ValueError("requests/workers must be positive and queue-size non-negative")

    config = PaperAssistantConfig(
        catalog_path=args.catalog,
        inbox_path=args.pdf_dir,
        settings_path=args.llm_settings,
        embedding_settings_path=args.embedding_settings,
        fallback_settings_path=args.fallback_settings,
        chroma_path=args.chroma_path,
        retriever="hybrid",
        paper_reranker="fastembed",
        paper_rerank_candidates=20,
        paper_min_score=2.5,
        chunk_reranker="fastembed",
        verify_claims="consensus",
        claim_judge_models=("deepseek-chat", "deepseek-v4-pro"),
        claim_second_judge_policy="risk_based",
        disable_claim_cache=True,
        top_k=7,
    )

    runtime_config = config
    if args.skip_verified:
        runtime_config = replace(
            config,
            verify_claims="off",
            claim_judge_models=(),
        )

    startup_rss = process_rss_mib()
    startup_started = time.perf_counter()
    service = build_paper_assistant_service(runtime_config)
    service_build_seconds = time.perf_counter() - startup_started

    verified_query = "物联网设备如何实现低功耗连接、设备管理和协议互操作？"
    verified_response: PaperAnswerResponse | None = None
    verified_seconds: float | None = None
    verified_payload: dict[str, object] | None = None
    if not args.skip_verified:
        verified_started = time.perf_counter()
        verified_response = service.answer_question(verified_query)
        verified_seconds = time.perf_counter() - verified_started
        verified_payload = verified_response.to_dict(include_evidence=True)

    # Exercise the same answer route and queue used by Streamlit. Claim judging
    # is already covered above; disabling it here prevents 20 requests from
    # expanding into hundreds of external judge calls.
    service.config = replace(
        service.config,
        verify_claims="off",
        claim_judge_models=(),
    )
    gate = RequestGate(
        kind="production_acceptance",
        max_workers=args.answer_workers,
        max_queue_size=args.queue_size,
        max_queue_wait_seconds=args.queue_wait_seconds,
    )
    requests = build_requests(args.requests)

    stop_sampling = threading.Event()
    rss_samples: list[float] = []

    def sample_memory() -> None:
        while not stop_sampling.wait(0.2):
            value = process_rss_mib()
            if value is not None:
                rss_samples.append(value)

    sampler = threading.Thread(target=sample_memory, name="rss-sampler", daemon=True)
    sampler.start()
    concurrent_started = time.perf_counter()
    completed = []
    errors = []
    try:
        with ThreadPoolExecutor(max_workers=len(requests)) as callers:
            futures = {
                callers.submit(
                    gate.run,
                    service.answer_question,
                    query,
                    result_classifier=classify_answer,
                ): request_id
                for request_id, query in requests
            }
            for future in as_completed(futures):
                request_id = futures[future]
                try:
                    execution = future.result()
                    response = execution.value
                    record = execution.record
                    completed.append(
                        {
                            "request_id": request_id,
                            "status": response.answer.status,
                            "reason": response.answer.reason,
                            "candidate_papers": len(response.candidate_paper_ids),
                            "evidence_chunks": len(response.evidence_results),
                            "claims": len(response.answer.claims),
                            "paper_fallback": response.paper_retriever_fallback,
                            "reranker_fallback": response.reranker_fallback,
                            "queue_wait_ms": round(record.queue_wait_seconds * 1000, 1),
                            "execution_ms": round(record.execution_seconds * 1000, 1),
                            "total_ms": round(record.total_seconds * 1000, 1),
                        }
                    )
                except RequestGateError as error:
                    errors.append(
                        {
                            "request_id": request_id,
                            "status": error.record.status,
                            "error_type": error.record.error_type,
                            "queue_wait_ms": round(
                                error.record.queue_wait_seconds * 1000, 1
                            ),
                            "total_ms": round(error.record.total_seconds * 1000, 1),
                        }
                    )
                except Exception as error:
                    errors.append(
                        {
                            "request_id": request_id,
                            "status": "caller_error",
                            "error_type": type(error).__name__,
                        }
                    )
    finally:
        concurrent_wall_seconds = time.perf_counter() - concurrent_started
        stop_sampling.set()
        sampler.join(timeout=2)
        gate_snapshot = gate.snapshot().to_dict()
        gate.shutdown()

    totals = [float(item["total_ms"]) for item in completed]
    queue_waits = [float(item["queue_wait_ms"]) for item in completed]
    execution_times = [float(item["execution_ms"]) for item in completed]
    status_counts = Counter(str(item["status"]) for item in completed)
    report = {
        "scope": (
            "production-path concurrency acceptance with claim verification disabled"
            if args.skip_verified
            else "production-path acceptance; one live consensus-verified answer plus "
            "queued answers with claim verification disabled"
        ),
        "config": {
            "papers": len(service.catalog),
            "paper_reranker": config.paper_reranker,
            "paper_min_score": config.paper_min_score,
            "chunk_reranker": config.chunk_reranker,
            "verified_claim_mode": config.verify_claims,
            "verified_judge_models": list(config.claim_judge_models),
            "concurrent_claim_mode": "off",
            "answer_workers": args.answer_workers,
            "queue_size": args.queue_size,
            "queue_wait_seconds": args.queue_wait_seconds,
        },
        "startup": {
            "service_build_seconds": round(service_build_seconds, 2),
            "rss_before_mib": startup_rss,
            "rss_after_verified_answer_mib": process_rss_mib(),
        },
        "verified_answer": (
            {"skipped": True}
            if args.skip_verified
            else {
                "query": verified_query,
                "elapsed_seconds": round(verified_seconds or 0.0, 2),
                "response": verified_payload,
            }
        ),
        "concurrency": {
            "requests": len(requests),
            "unique_queries": len({query for _, query in requests}),
            "topic_families": min(len(TOPIC_QUERIES), len(requests)),
            "completed": len(completed),
            "failed": len(errors),
            "status_counts": dict(sorted(status_counts.items())),
            "wall_seconds": round(concurrent_wall_seconds, 2),
            "total_latency": latency_summary(totals),
            "execution_latency": latency_summary(execution_times),
            "queue_wait_latency": latency_summary(queue_waits),
            "peak_rss_mib": max(rss_samples, default=process_rss_mib()),
            "rss_after_mib": process_rss_mib(),
            "gate": gate_snapshot,
            "fallback_requests": sum(
                bool(item["paper_fallback"] or item["reranker_fallback"])
                for item in completed
            ),
            "errors": sorted(errors, key=lambda item: item["request_id"]),
            "details": sorted(completed, key=lambda item: item["request_id"]),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    summary = {
        "output": str(args.output),
        "verified_status": (
            "skipped" if verified_response is None else verified_response.answer.status
        ),
        "verified_claims": (
            None if verified_response is None else len(verified_response.answer.claims)
        ),
        "verified_claim_report": (
            None if verified_response is None else verified_response.claim_verification
        ),
        "completed": len(completed),
        "failed": len(errors),
        "unique_queries": report["concurrency"]["unique_queries"],
        "wall_seconds": round(concurrent_wall_seconds, 2),
        "p95_ms": report["concurrency"]["total_latency"]["p95_ms"],
        "max_queue_wait_ms": report["concurrency"]["queue_wait_latency"]["max_ms"],
        "peak_rss_mib": report["concurrency"]["peak_rss_mib"],
        "fallback_requests": report["concurrency"]["fallback_requests"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    verified_ok = (
        verified_response is None or verified_response.answer.status == "answered"
    )
    return 0 if verified_ok and not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
