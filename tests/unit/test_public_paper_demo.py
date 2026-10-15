"""Public demo runs offline and keeps source metadata/configuration isolated."""

import socket
from pathlib import Path
from unittest.mock import patch

import pytest

from scripts.check_paper_demo import DEMO, check_demo
from src.paper_assistant.catalog import PaperCatalog
from src.paper_assistant.chunk_retriever import extract_paper_chunks
from src.paper_assistant.web_config import citation_pdf_path, dataset_paths_from_environment


def test_public_demo_is_complete_without_network(monkeypatch):
    def forbid_network(*args, **kwargs):
        raise AssertionError("Offline demo attempted network access")

    monkeypatch.setattr(socket.socket, "connect", forbid_network)
    monkeypatch.setattr(socket.socket, "connect_ex", forbid_network)
    monkeypatch.setattr(socket, "create_connection", forbid_network)
    report = check_demo()
    assert report["success"]
    assert report["papers"] == 3
    assert report["chunks"] >= 6
    assert report["paper_retrieval"]["overall"]["queries"] == 6
    assert report["paper_retrieval"]["rejection"]["queries"] == 2


def test_public_pdf_chunks_have_reproducible_ids_and_real_pages():
    for profile in PaperCatalog.from_csv(DEMO / "paper_catalog.csv").profiles:
        pdf = DEMO / "pdfs" / profile.pdf_files[0]
        first = extract_paper_chunks(profile, pdf)
        second = extract_paper_chunks(profile, pdf)
        assert [chunk.chunk_id for chunk in first] == [chunk.chunk_id for chunk in second]
        assert {chunk.page_number for chunk in first} == {1, 2}
        assert all(chunk.paper_id == profile.paper_id for chunk in first)
        assert all(chunk.section for chunk in first)


def test_smoke_detects_incorrect_source_page(tmp_path):
    import json
    import shutil

    shutil.copytree(DEMO, tmp_path / "demo")
    queries = tmp_path / "demo" / "chunk_queries.jsonl"
    cases = [
        json.loads(line)
        for line in queries.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    cases[0]["relevant_pages"] = [2]
    queries.write_text("\n".join(json.dumps(case) for case in cases) + "\n", encoding="utf-8")
    report = check_demo(tmp_path / "demo")
    assert not report["success"]
    assert "demo_c01" in report["source_label_failures"]


def test_dataset_selection_does_not_fall_back_to_production_paths():
    with patch.dict(
        "os.environ",
        {
            "SCHOLARRAG_CATALOG": "demo/catalog.csv",
            "SCHOLARRAG_INBOX": "demo/pdfs",
            "SCHOLARRAG_CHROMA_PATH": "tmp/demo/chroma",
            "SCHOLARRAG_CLAIM_CACHE": "tmp/demo/claims.sqlite3",
        },
        clear=True,
    ):
        paths = dataset_paths_from_environment()
    assert paths == {
        "catalog_path": Path("demo/catalog.csv"),
        "inbox_path": Path("demo/pdfs"),
        "chroma_path": Path("tmp/demo/chroma"),
        "claim_cache_path": Path("tmp/demo/claims.sqlite3"),
    }


@pytest.mark.parametrize("filename", ["../outside.pdf", "note.txt", "missing.pdf"])
def test_pdf_download_cannot_leave_dataset(tmp_path, filename):
    inbox = tmp_path / "pdfs"
    inbox.mkdir()
    (tmp_path / "outside.pdf").write_bytes(b"%PDF")
    (inbox / "note.txt").write_text("not a PDF")
    assert citation_pdf_path(inbox, filename) is None


def test_pdf_download_uses_current_dataset(tmp_path):
    path = tmp_path / "source.pdf"
    path.write_bytes(b"%PDF")
    assert citation_pdf_path(tmp_path, "source.pdf") == path.resolve()


def _render_page_for_test():
    from src.observability.dashboard.pages.paper_assistant import render

    render()


def test_web_config_change_discards_old_answers(monkeypatch):
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("SCHOLARRAG_CATALOG", "first/catalog.csv")
    app = AppTest.from_function(_render_page_for_test).run()
    assert not app.exception
    app.session_state["paper_answer_response"] = "old answer"
    app.session_state["paper_search_response"] = "old search"
    monkeypatch.setenv("SCHOLARRAG_CATALOG", "second/catalog.csv")
    app.run()
    assert not app.exception
    assert "paper_answer_response" not in app.session_state
    assert "paper_search_response" not in app.session_state


def test_shared_citation_pdf_is_offered_only_once(monkeypatch, tmp_path):
    from unittest.mock import MagicMock

    from src.observability.dashboard.pages import paper_assistant as page
    from src.paper_assistant.grounded_answer import (
        GroundedAnswer,
        GroundedCitation,
        GroundedClaim,
    )
    from src.paper_assistant.service import PaperAnswerResponse

    pdf = tmp_path / "demo.pdf"
    pdf.write_bytes(b"%PDF")
    citation = GroundedCitation(
        citation_id="C1",
        paper_id="demo",
        paper_title="Demo",
        pdf_file="demo.pdf",
        page_number=1,
        section="METHOD",
        chunk_id="demo:1",
        score=0.9,
        excerpt="Evidence",
    )
    answer = GroundedAnswer(
        status="answered",
        answer="Example",
        claims=(GroundedClaim("First", ("C1",)), GroundedClaim("Second", ("C1",))),
        citations=(citation,),
        reason=None,
        model="test",
        usage={},
    )
    response = PaperAnswerResponse(
        query="question",
        candidate_paper_ids=("demo",),
        answer=answer,
        evidence_results=(),
        paper_retriever="bm25",
        paper_retriever_fallback=None,
        chunk_reranker="none",
        reranker_model=None,
        reranker_fallback=None,
        index_sync={},
        claim_verification={},
    )
    ui = MagicMock()
    monkeypatch.setattr(page, "st", ui)
    monkeypatch.setattr(page, "_render_verification", lambda value: None)
    monkeypatch.setattr(page, "_render_request_record", lambda value: None)
    page._render_answer(response, inbox_path=tmp_path)
    assert ui.download_button.call_count == 1
    assert ui.download_button.call_args.kwargs["data"] == b"%PDF"
