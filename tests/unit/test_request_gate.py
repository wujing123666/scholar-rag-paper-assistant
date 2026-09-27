"""Tests for the single-process ScholarRAG request gate."""

from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from src.paper_assistant.request_gate import (
    RequestExecutionError,
    RequestGate,
    RequestQueueFullError,
    RequestQueueWaitTimeoutError,
)


def _wait_until(predicate, timeout: float = 1.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.005)
    raise AssertionError("condition did not become true")


def test_gate_never_exceeds_worker_limit(tmp_path):
    gate = RequestGate(
        kind="answer",
        max_workers=2,
        max_queue_size=10,
        max_queue_wait_seconds=1,
        log_path=tmp_path / "requests.jsonl",
    )
    lock = threading.Lock()
    active = 0
    maximum = 0

    def work(value: int) -> int:
        nonlocal active, maximum
        with lock:
            active += 1
            maximum = max(maximum, active)
        time.sleep(0.03)
        with lock:
            active -= 1
        return value * 2

    with ThreadPoolExecutor(max_workers=6) as callers:
        futures = [callers.submit(gate.run, work, value) for value in range(6)]
        executions = [future.result() for future in futures]

    assert maximum == 2
    assert [item.value for item in executions] == [0, 2, 4, 6, 8, 10]
    snapshot = gate.snapshot()
    assert snapshot.completed == 6
    assert snapshot.active == 0
    assert snapshot.queued == 0
    gate.shutdown()


def test_gate_rejects_when_running_and_waiting_slots_are_full():
    gate = RequestGate(
        kind="answer",
        max_workers=1,
        max_queue_size=1,
        max_queue_wait_seconds=1,
    )
    release = threading.Event()
    started = threading.Event()

    def blocking() -> str:
        started.set()
        release.wait(timeout=1)
        return "done"

    with ThreadPoolExecutor(max_workers=2) as callers:
        first = callers.submit(gate.run, blocking)
        assert started.wait(timeout=1)
        second = callers.submit(gate.run, lambda: "queued")
        _wait_until(lambda: gate.snapshot().queued == 1)

        with pytest.raises(RequestQueueFullError) as error:
            gate.run(lambda: "rejected")

        assert error.value.record.status == "rejected_queue_full"
        release.set()
        assert first.result().value == "done"
        assert second.result().value == "queued"

    assert gate.snapshot().rejected == 1
    gate.shutdown()


def test_expired_queued_task_does_not_call_rag():
    gate = RequestGate(
        kind="answer",
        max_workers=1,
        max_queue_size=1,
        max_queue_wait_seconds=0.02,
    )
    release = threading.Event()
    started = threading.Event()
    second_called = threading.Event()

    def first_work() -> None:
        started.set()
        release.wait(timeout=1)

    def second_work() -> None:
        second_called.set()

    with ThreadPoolExecutor(max_workers=2) as callers:
        first = callers.submit(gate.run, first_work)
        assert started.wait(timeout=1)
        second = callers.submit(gate.run, second_work)
        _wait_until(lambda: gate.snapshot().queued == 1)
        time.sleep(0.04)
        release.set()
        first.result()
        with pytest.raises(RequestQueueWaitTimeoutError) as error:
            second.result()

    assert error.value.record.status == "queue_timeout"
    assert second_called.is_set() is False
    assert gate.snapshot().queue_timeouts == 1
    gate.shutdown()


def test_failure_log_contains_type_but_not_sensitive_message(tmp_path):
    log_path = tmp_path / "requests.jsonl"
    gate = RequestGate(
        kind="answer",
        max_workers=1,
        max_queue_size=0,
        max_queue_wait_seconds=1,
        log_path=log_path,
    )

    def fail() -> None:
        raise ValueError("secret paper text and API key")

    with pytest.raises(RequestExecutionError) as error:
        gate.run(fail)

    assert error.value.record.error_type == "ValueError"
    payload = json.loads(log_path.read_text(encoding="utf-8"))
    assert payload["status"] == "failed"
    assert payload["error_type"] == "ValueError"
    assert "secret paper text" not in log_path.read_text(encoding="utf-8")
    assert gate.snapshot().failed == 1
    gate.shutdown()


def test_result_classifier_records_safe_application_status(tmp_path):
    log_path = tmp_path / "requests.jsonl"
    gate = RequestGate(
        kind="answer",
        max_workers=1,
        max_queue_size=0,
        max_queue_wait_seconds=1,
        log_path=log_path,
    )

    execution = gate.run(
        lambda: {"status": "insufficient_evidence", "reason": "model_failed"},
        result_classifier=lambda result: (result["status"], result["reason"]),
    )

    assert execution.record.status == "completed"
    assert execution.record.result_status == "insufficient_evidence"
    assert execution.record.result_reason == "model_failed"
    payload = json.loads(log_path.read_text(encoding="utf-8"))
    assert payload["result_status"] == "insufficient_evidence"
    assert payload["result_reason"] == "model_failed"
    gate.shutdown()
