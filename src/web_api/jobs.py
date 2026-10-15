"""Bounded task receipts persisted across API restarts, not a second work queue."""

import hashlib
import json
import sqlite3
import threading
import time
import uuid
from collections.abc import Iterator
from concurrent.futures import Future
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from src.paper_assistant.request_gate import (
    RequestGate,
    RequestQueueFullError,
    RequestQueueWaitTimeoutError,
)
from src.web_api.schemas import ApiError, Job, QueryRequest


class JobConflictError(ValueError):
    pass


class JobCapacityError(ValueError):
    pass


class Jobs:
    def __init__(self, path: Path, registry: Any, *, max_rows: int = 200, ttl: float = 86400,
                 answer_workers: int = 3, search_workers: int = 8, queue_size: int = 20,
                 queue_wait: float = 180, search_queue_wait: float = 60) -> None:
        self.path = path
        self.registry = registry
        self.max_rows, self.ttl = max_rows, ttl
        self.lock = threading.RLock()
        path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as conn:
            conn.execute("""CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY, owner TEXT NOT NULL, idem TEXT NOT NULL,
                fingerprint TEXT NOT NULL, payload TEXT NOT NULL, updated REAL NOT NULL,
                UNIQUE(owner, idem))""")
            for job_id, payload in conn.execute("SELECT id,payload FROM jobs").fetchall():
                job = Job.model_validate_json(payload)
                if job.state in {"queued", "running"}:
                    job.state = "interrupted"
                    job.error = ApiError(code="INTERRUPTED", message="服务已重启，原任务已中断。请重新提问。", request_id=job_id)
                    conn.execute("UPDATE jobs SET payload=?,updated=? WHERE id=?", (job.model_dump_json(), time.time(), job_id))
        self.gates = {
            "answer": RequestGate(kind="web_answer", max_workers=answer_workers,
                                  max_queue_size=queue_size, max_queue_wait_seconds=queue_wait),
            "search": RequestGate(kind="web_search", max_workers=search_workers,
                                  max_queue_size=queue_size, max_queue_wait_seconds=search_queue_wait),
        }

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=10)
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def get(self, job_id: str, owner: str) -> Job:
        with self.connect() as conn:
            row = conn.execute("SELECT payload,updated FROM jobs WHERE id=? AND owner=?", (job_id, owner)).fetchone()
        if row is None:
            raise KeyError(job_id)
        job = Job.model_validate_json(row[0])
        if row[1] < time.time() - self.ttl and job.state not in {"queued", "running"}:
            raise KeyError(job_id)
        return job

    def _update(self, job_id: str, owner: str, **changes: Any) -> Job:
        with self.lock:
            job = self.get(job_id, owner).model_copy(update=changes)
            with self.connect() as conn:
                conn.execute("UPDATE jobs SET payload=?,updated=? WHERE id=? AND owner=?",
                             (job.model_dump_json(), time.time(), job_id, owner))
            return job

    def submit(self, owner: str, idem: str, kind: str, request: QueryRequest) -> Job:
        fingerprint = hashlib.sha256(json.dumps([kind, request.query, sorted(request.paper_ids), request.dataset_id],
                                                ensure_ascii=False).encode()).hexdigest()
        with self.lock:
            with self.connect() as conn:
                rows = conn.execute("SELECT id,payload,updated FROM jobs").fetchall()
                for job_id, payload, updated in rows:
                    if updated < time.time() - self.ttl and Job.model_validate_json(payload).state not in {"queued", "running"}:
                        conn.execute("DELETE FROM jobs WHERE id=?", (job_id,))
                existing = conn.execute("SELECT fingerprint,payload FROM jobs WHERE owner=? AND idem=?", (owner, idem)).fetchone()
                if existing:
                    if existing[0] != fingerprint:
                        raise JobConflictError("Idempotency key was reused for a different request")
                    return Job.model_validate_json(existing[1])
                if conn.execute("SELECT count(*) FROM jobs").fetchone()[0] >= self.max_rows:
                    raise JobCapacityError("Task receipt capacity reached; wait for expiration")
                job = Job(job_id=uuid.uuid4().hex, kind=kind, state="queued", dataset_id=request.dataset_id)
                conn.execute("INSERT INTO jobs VALUES (?,?,?,?,?,?)",
                             (job.job_id, owner, idem, fingerprint, job.model_dump_json(), time.time()))
            function = self.registry.answer if kind == "answer" else self.registry.search
            try:
                future = self.gates[kind].submit(
                    function, request.query, request.paper_ids,
                    on_start=lambda: self._update(job.job_id, owner, state="running"),
                )
            except RequestQueueFullError:
                self._update(job.job_id, owner, state="failed", error=ApiError(
                    code="QUEUE_FULL", message="请求队列已满，请稍后再试。", request_id=job.job_id))
                raise
            except Exception:
                self._update(job.job_id, owner, state="failed", error=ApiError(
                    code="SERVICE_UNAVAILABLE", message="服务暂时无法接纳任务。", request_id=job.job_id))
                raise

            def finished(done: Future) -> None:
                try:
                    result = done.result().value
                    self._update(job.job_id, owner, state="completed", result=result)
                except RequestQueueWaitTimeoutError:
                    self._update(job.job_id, owner, state="queue_timeout", error=ApiError(
                        code="QUEUE_TIMEOUT", message="排队等待已超时，本次未调用模型。请稍后重新提问。", request_id=job.job_id))
                except BaseException:
                    self._update(job.job_id, owner, state="failed", error=ApiError(
                        code="TASK_FAILED", message="请求未能完成，请检查服务状态后重试。", request_id=job.job_id))

            future.add_done_callback(finished)
            return self.get(job.job_id, owner)

    def close(self) -> None:
        for gate in self.gates.values():
            gate.shutdown(wait=True)
