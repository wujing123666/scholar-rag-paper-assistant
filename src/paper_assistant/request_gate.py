"""Single-process concurrency limits and bounded queues for ScholarRAG."""

from __future__ import annotations

import json
import threading
import time
import uuid
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Generic, TypeVar, cast

T = TypeVar("T")
_LOG_LOCK = threading.Lock()


@dataclass(frozen=True)
class RequestRecord:
    """Privacy-safe timing and status data for one submitted request."""

    request_id: str
    kind: str
    submitted_at: str
    status: str
    queue_wait_seconds: float
    execution_seconds: float
    total_seconds: float
    error_type: str | None = None
    result_status: str | None = None
    result_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RequestExecution(Generic[T]):
    value: T
    record: RequestRecord


@dataclass(frozen=True)
class RequestGateSnapshot:
    kind: str
    max_workers: int
    max_queue_size: int
    active: int
    queued: int
    completed: int
    failed: int
    rejected: int
    queue_timeouts: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class RequestGateError(RuntimeError):
    """Base class for errors returned by the request admission layer."""

    def __init__(self, message: str, record: RequestRecord) -> None:
        super().__init__(message)
        self.record = record


class RequestQueueFullError(RequestGateError):
    """Raised when all running and waiting slots are occupied."""


class RequestQueueWaitTimeoutError(RequestGateError):
    """Raised when a queued request waited longer than its allowed budget."""


class RequestExecutionError(RequestGateError):
    """Raised with sanitized diagnostics when the admitted task fails."""


@dataclass(frozen=True)
class _TaskOutcome(Generic[T]):
    value: T | None
    error: Exception | None
    record: RequestRecord


class RequestGate:
    """Run work in a bounded process-local executor and record diagnostics."""

    def __init__(
        self,
        *,
        kind: str,
        max_workers: int,
        max_queue_size: int,
        max_queue_wait_seconds: float,
        log_path: Path | None = None,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        if not kind.strip():
            raise ValueError("kind cannot be empty")
        if max_workers < 1:
            raise ValueError("max_workers must be at least one")
        if max_queue_size < 0:
            raise ValueError("max_queue_size cannot be negative")
        if max_queue_wait_seconds < 0:
            raise ValueError("max_queue_wait_seconds cannot be negative")
        self.kind = kind.strip()
        self.max_workers = max_workers
        self.max_queue_size = max_queue_size
        self.max_queue_wait_seconds = max_queue_wait_seconds
        self.log_path = log_path
        self._monotonic = monotonic
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix=f"scholarrag-{self.kind}",
        )
        self._capacity = threading.BoundedSemaphore(
            max_workers + max_queue_size
        )
        self._state_lock = threading.Lock()
        self._active = 0
        self._queued = 0
        self._completed = 0
        self._failed = 0
        self._rejected = 0
        self._queue_timeouts = 0

    def run(
        self,
        function: Callable[..., T],
        /,
        *args: Any,
        result_classifier: Callable[[T], tuple[str | None, str | None]] | None = None,
        **kwargs: Any,
    ) -> RequestExecution[T]:
        """Submit one task, wait for it, and return result plus queue diagnostics."""
        request_id = uuid.uuid4().hex
        submitted_wall = datetime.now(UTC).isoformat()
        submitted_at = self._monotonic()
        if not self._capacity.acquire(blocking=False):
            record = self._new_record(
                request_id=request_id,
                submitted_at=submitted_wall,
                status="rejected_queue_full",
                queue_wait=0.0,
                execution=0.0,
                total=0.0,
                error_type="RequestQueueFullError",
            )
            with self._state_lock:
                self._rejected += 1
            self._write_record(record)
            raise RequestQueueFullError("request queue is full", record)

        with self._state_lock:
            self._queued += 1
        try:
            future: Future[_TaskOutcome[T]] = self._executor.submit(
                self._execute,
                function,
                args,
                kwargs,
                request_id,
                submitted_wall,
                submitted_at,
                result_classifier,
            )
        except Exception:
            with self._state_lock:
                self._queued -= 1
            self._capacity.release()
            raise

        outcome = future.result()
        if outcome.error is not None:
            if outcome.record.status == "queue_timeout":
                raise RequestQueueWaitTimeoutError(
                    "request expired while waiting in the queue", outcome.record
                ) from outcome.error
            raise RequestExecutionError(
                "admitted request failed", outcome.record
            ) from outcome.error
        return RequestExecution(cast(T, outcome.value), outcome.record)

    def snapshot(self) -> RequestGateSnapshot:
        with self._state_lock:
            return RequestGateSnapshot(
                kind=self.kind,
                max_workers=self.max_workers,
                max_queue_size=self.max_queue_size,
                active=self._active,
                queued=self._queued,
                completed=self._completed,
                failed=self._failed,
                rejected=self._rejected,
                queue_timeouts=self._queue_timeouts,
            )

    def shutdown(self, *, wait: bool = True) -> None:
        self._executor.shutdown(wait=wait, cancel_futures=True)

    def _execute(
        self,
        function: Callable[..., T],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        request_id: str,
        submitted_wall: str,
        submitted_at: float,
        result_classifier: Callable[[T], tuple[str | None, str | None]] | None,
    ) -> _TaskOutcome[T]:
        started_at = self._monotonic()
        queue_wait = max(0.0, started_at - submitted_at)
        with self._state_lock:
            self._queued -= 1
            self._active += 1
        try:
            if queue_wait > self.max_queue_wait_seconds:
                error = TimeoutError("queue wait budget exceeded")
                record = self._new_record(
                    request_id=request_id,
                    submitted_at=submitted_wall,
                    status="queue_timeout",
                    queue_wait=queue_wait,
                    execution=0.0,
                    total=self._monotonic() - submitted_at,
                    error_type=type(error).__name__,
                )
                with self._state_lock:
                    self._queue_timeouts += 1
                return _TaskOutcome(None, error, record)
            try:
                value = function(*args, **kwargs)
            except Exception as error:
                finished_at = self._monotonic()
                record = self._new_record(
                    request_id=request_id,
                    submitted_at=submitted_wall,
                    status="failed",
                    queue_wait=queue_wait,
                    execution=finished_at - started_at,
                    total=finished_at - submitted_at,
                    error_type=type(error).__name__,
                )
                with self._state_lock:
                    self._failed += 1
                return _TaskOutcome(None, error, record)
            finished_at = self._monotonic()
            result_status = None
            result_reason = None
            if result_classifier is not None:
                try:
                    result_status, result_reason = result_classifier(value)
                except Exception:
                    # Result observability must never turn a successful request
                    # into a user-visible failure.
                    pass
            record = self._new_record(
                request_id=request_id,
                submitted_at=submitted_wall,
                status="completed",
                queue_wait=queue_wait,
                execution=finished_at - started_at,
                total=finished_at - submitted_at,
                result_status=result_status,
                result_reason=result_reason,
            )
            with self._state_lock:
                self._completed += 1
            return _TaskOutcome(value, None, record)
        finally:
            with self._state_lock:
                self._active -= 1
            self._capacity.release()
            if "record" in locals():
                self._write_record(record)

    def _new_record(
        self,
        *,
        request_id: str,
        submitted_at: str,
        status: str,
        queue_wait: float,
        execution: float,
        total: float,
        error_type: str | None = None,
        result_status: str | None = None,
        result_reason: str | None = None,
    ) -> RequestRecord:
        return RequestRecord(
            request_id=request_id,
            kind=self.kind,
            submitted_at=submitted_at,
            status=status,
            queue_wait_seconds=round(max(0.0, queue_wait), 6),
            execution_seconds=round(max(0.0, execution), 6),
            total_seconds=round(max(0.0, total), 6),
            error_type=error_type,
            result_status=result_status,
            result_reason=result_reason,
        )

    def _write_record(self, record: RequestRecord) -> None:
        if self.log_path is None:
            return
        try:
            with _LOG_LOCK:
                self.log_path.parent.mkdir(parents=True, exist_ok=True)
                with self.log_path.open("a", encoding="utf-8") as handle:
                    handle.write(
                        json.dumps(record.to_dict(), ensure_ascii=False) + "\n"
                    )
        except OSError:
            # Observability failures must not fail an admitted RAG request.
            return
