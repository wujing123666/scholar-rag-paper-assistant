"""Offline HTTP boundary tests: no model calls and no production indexes."""

from __future__ import annotations

import hashlib
import shutil
import threading
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from src.paper_assistant.catalog import PaperCatalog
from src.paper_assistant.chunk_retriever import PaperChunkRetriever
from src.paper_assistant.dense_retriever import PaperDenseRetriever
from src.paper_assistant.request_gate import RequestGate
from src.paper_assistant.retriever import PaperSearchResult
from src.paper_assistant.service import PaperAssistantConfig, PaperAssistantService
from src.web_api.app import create_app
from src.web_api.jobs import Jobs
from src.web_api.library import Library
from src.web_api.registry import Registry
from src.web_api.schemas import AnswerResult, Job, SearchResult, Verification
from src.web_api.settings import WebSettings

DEMO = Path(__file__).resolve().parents[2] / "examples/paper_demo"


class FakeRegistry:
    def __init__(self, settings):
        self.library = Library(settings.rag.catalog_path, settings.rag.inbox_path, settings.rag.chroma_path)
        self.started = threading.Event()
        self.release = threading.Event()
        self.block = False
        self.fail = False
        self.calls = 0

    def answer(self, query, paper_ids):
        self.calls += 1
        self.started.set()
        if self.block:
            self.release.wait(2)
        if self.fail:
            raise ValueError("private API key and private directory")
        return AnswerResult(status="refused", claims=[], citations=[], reason="insufficient_evidence",
                            verification=Verification(status="skipped"))

    def search(self, query, paper_ids):
        return SearchResult(papers=[self.library.papers["demo_mesh"]], rejected=False, reason=None)

    def close(self):
        self.release.set()


@pytest.fixture
def web(tmp_path):
    # File changes in tests must never alter the committed fixture files.
    shutil.copytree(DEMO, tmp_path / "demo")
    settings = WebSettings(
        rag=PaperAssistantConfig(catalog_path=tmp_path / "demo/paper_catalog.csv", inbox_path=tmp_path / "demo/pdfs"),
        state_path=tmp_path / "web", answer_workers=1, queue_size=1,
    )
    registry = FakeRegistry(settings)
    app = create_app(settings, registry_factory=lambda _: registry)
    with TestClient(app) as client:
        health = client.get("/api/v1/health").json()
        yield client, registry, health
        registry.release.set()


def body(health, query="test"):
    return {"query": query, "paper_ids": [], "dataset_id": health["dataset_id"]}


def post(client, health, query="test", key="idempotency_test_0001"):
    return client.post("/api/v1/answers", json=body(health, query), headers={"Idempotency-Key": key})


def poll(client, job_id):
    for _ in range(100):
        job = client.get(f"/api/v1/jobs/{job_id}").json()
        if job["state"] not in {"queued", "running"}:
            return job
        time.sleep(.005)
    raise AssertionError("job did not finish")


def test_http_idempotency_and_session_ownership(web):
    client, registry, health = web
    first = post(client, health)
    assert first.status_code == 202
    job_id = first.json()["job_id"]
    assert post(client, health).json()["job_id"] == job_id
    assert post(client, health, query="changed").status_code == 409
    job = poll(client, job_id)
    assert job["state"] == "completed" and job["result"]["status"] == "refused"
    assert registry.calls == 1
    cookie = client.cookies.get("scholarrag_session")
    client.cookies.clear()
    assert client.get("/api/v1/health").json()["session_id"] != health["session_id"]
    assert client.get(f"/api/v1/jobs/{job_id}").status_code == 404
    client.cookies.set("scholarrag_session", cookie)


def test_static_requests_never_create_or_replace_api_session(web):
    client, _, health = web
    # The API cookie's path excludes these requests, even in the same browser.
    assert "set-cookie" not in client.get("/").headers
    assert "set-cookie" not in client.get("/assets/review-missing.js").headers
    assert client.get("/api/v1/health").json()["session_id"] == health["session_id"]


def test_only_health_initializes_a_missing_session(web):
    client, registry, health = web
    client.cookies.clear()
    assert "set-cookie" not in client.get("/api/v1/papers").headers
    missing = post(client, health)
    assert missing.status_code == 409 and missing.json()["code"] == "SESSION_REQUIRED"
    assert "set-cookie" not in missing.headers and registry.calls == 0
    assert client.get("/api/v1/jobs/unknown").status_code == 409
    renewed = client.get("/api/v1/health")
    assert "set-cookie" in renewed.headers and renewed.json()["session_id"] != health["session_id"]
    assert post(client, renewed.json()).status_code == 202


@pytest.mark.parametrize("patch", [{"query": " "}, {"query": "x" * 2001}, {"paper_ids": ["unknown"]},
                                  {"paper_ids": ["a", "b", "c", "d"]}, {"settings_path": "secret"}])
def test_invalid_request_does_not_reach_rag(web, patch):
    client, registry, health = web
    response = client.post("/api/v1/answers", json={**body(health), **patch},
                           headers={"Idempotency-Key": "idempotency_test_0001"})
    assert response.status_code == 422 and registry.calls == 0
    assert "secret" not in response.text


def test_pdf_mapping_and_range_support(web):
    client, registry, _ = web
    paper = client.get("/api/v1/papers/demo_mesh").json()
    url = paper["files"][0]["url"]
    response = client.get(url)
    assert response.headers["content-type"] == "application/pdf"
    assert hashlib.sha256(response.content).hexdigest() == hashlib.sha256((DEMO / "pdfs/demo_mesh.pdf").read_bytes()).hexdigest()
    partial = client.get(url, headers={"Range": "bytes=0-9"})
    assert partial.status_code == 206 and len(partial.content) == 10
    assert client.get("/api/v1/papers/demo_mesh/files/not-a-file").status_code == 404
    assert str(registry.library.inbox) not in client.get("/api/v1/papers").text


def test_queue_full_and_failure_are_sanitized(web):
    client, registry, health = web
    registry.block = True
    first = post(client, health, key="idempotency_first_01").json()
    assert registry.started.wait(1)
    second = post(client, health, key="idempotency_second_02").json()
    assert second["state"] == "queued"
    assert post(client, health, key="idempotency_third_03").status_code == 429
    registry.fail = True
    registry.release.set()
    failed = poll(client, first["job_id"])
    assert failed["state"] == "failed"
    assert "private" not in str(failed)
    poll(client, second["job_id"])


def test_origin_json_and_dataset_boundaries(web):
    client, registry, health = web
    assert client.post("/api/v1/answers", json=body(health), headers={"Origin": "https://untrusted.example"}).status_code == 403
    assert client.post("/api/v1/answers", content="query=x").status_code == 415
    invalid = {**body(health), "dataset_id": "different"}
    assert client.post("/api/v1/answers", json=invalid, headers={"Idempotency-Key": "idempotency_test_0001"}).status_code == 409
    registry.library.catalog_path.write_text("changed", encoding="utf-8")
    assert post(client, health).status_code == 409
    assert registry.calls == 0


def test_shutdown_releases_cancelled_queued_capacity():
    gate = RequestGate(kind="test", max_workers=1, max_queue_size=1, max_queue_wait_seconds=1)
    started, release = threading.Event(), threading.Event()

    def block():
        started.set()
        release.wait(1)

    first = gate.submit(block)
    assert started.wait(1)
    queued = gate.submit(lambda: None)
    gate.shutdown(wait=False)
    assert queued.cancelled() and gate.snapshot().queued == 0
    release.set()
    first.result(timeout=1)
    assert gate.snapshot().active == 0


def test_restart_marks_receipts_interrupted_without_resubmission(tmp_path):
    settings = WebSettings(rag=PaperAssistantConfig(catalog_path=DEMO / "paper_catalog.csv", inbox_path=DEMO / "pdfs"))
    registry = FakeRegistry(settings)
    path = tmp_path / "jobs.sqlite3"
    jobs = Jobs(path, registry)
    receipt = Job(job_id="old", dataset_id=registry.library.dataset_id, kind="answer", state="running")
    with jobs.connect() as conn:
        conn.execute("INSERT INTO jobs VALUES (?,?,?,?,?,?)", ("old", "owner", "idem", "fingerprint", receipt.model_dump_json(), time.time()))
    jobs.close()
    restarted = Jobs(path, registry)
    assert restarted.get("old", "owner").state == "interrupted" and registry.calls == 0
    restarted.close()


def test_read_only_indexes_fail_before_embedding_or_writes():
    class Embedding:
        model = "test"
        def get_dimension(self): return 2
        def embed(self, *args, **kwargs): raise AssertionError("must not charge for indexing")

    class EmptyStore:
        def list_ids(self): return []
        def upsert(self, *args): raise AssertionError("must not write")
        def delete(self, *args): raise AssertionError("must not delete")

    catalog = PaperCatalog.from_csv(DEMO / "paper_catalog.csv")
    with pytest.raises(ValueError, match="rebuild offline"):
        PaperDenseRetriever(catalog, Embedding(), EmptyStore(), read_only=True)
    with pytest.raises(ValueError, match="rebuild offline"):
        PaperChunkRetriever(catalog, DEMO / "pdfs", Embedding(), EmptyStore(), read_only=True)


def test_scoped_search_filters_before_the_candidate_limit():
    library = Library(DEMO / "paper_catalog.csv", DEMO / "pdfs", Path("unused"))
    profiles = library.catalog.profiles
    extra = replace(profiles[0], paper_id="extra")
    full = PaperCatalog([profiles[0], profiles[1], extra, profiles[2]])

    class Retriever:
        name = "paper_bm25"
        catalog = full
        def search(self, query, top_k=3):
            return [PaperSearchResult(paper, 12.0, ()) for paper in full.profiles][:top_k]

    registry = Registry.__new__(Registry)
    registry.library = library
    registry.service = PaperAssistantService(
        catalog=full, paper_retriever=Retriever(), requested_retriever="bm25",
        config=PaperAssistantConfig(paper_reranker="none"),
    )
    assert [paper.paper_id for paper in registry.search("query", [profiles[2].paper_id]).papers] == [profiles[2].paper_id]


def test_provider_failure_is_not_reported_as_insufficient_paper_evidence():
    registry = Registry.__new__(Registry)
    registry.library = Library(DEMO / "paper_catalog.csv", DEMO / "pdfs", Path("unused"))
    registry.service = SimpleNamespace(answer_question=lambda *args, **kwargs: SimpleNamespace(
        answer=SimpleNamespace(status="insufficient_evidence", reason="llm_generation_failed")))
    with pytest.raises(RuntimeError, match="service did not complete"):
        registry.answer("question", [])


def test_second_process_cannot_recover_another_processes_live_jobs(web):
    client, registry, _ = web
    settings = WebSettings(rag=PaperAssistantConfig(
        catalog_path=registry.library.catalog_path, inbox_path=registry.library.inbox),
        state_path=Path(client.app.state.jobs.path).parent.parent)
    called = []
    second = create_app(settings, registry_factory=lambda _: called.append(True))
    with pytest.raises(RuntimeError, match="Only one API process"):
        with TestClient(second):
            pass
    assert not called


def test_replacing_pdf_without_changing_csv_creates_new_source_identity(tmp_path):
    shutil.copytree(DEMO, tmp_path / "demo")
    catalog, inbox = tmp_path / "demo/paper_catalog.csv", tmp_path / "demo/pdfs"
    old = Library(catalog, inbox, tmp_path / "index")
    pdf = inbox / "demo_mesh.pdf"
    pdf.write_bytes(pdf.read_bytes() + b"\n% New PDF revision\n")
    with pytest.raises(ValueError, match="PDF source changed"):
        old.assert_unchanged()
    new = Library(catalog, inbox, tmp_path / "index")
    assert new.dataset_id != old.dataset_id
    assert new.papers["demo_mesh"].files[0].file_id != old.papers["demo_mesh"].files[0].file_id
