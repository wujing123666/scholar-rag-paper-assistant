"""Tests for evidence-backed Paper Profile draft generation."""

from __future__ import annotations

import csv
import json
import shutil
import subprocess
import sys

import pymupdf as fitz
import pytest

from src.paper_assistant.profile_draft import (
    extract_pdf_facts,
    prepare_profile_draft,
    write_profile_draft,
)

CATALOG_FIELDS = [
    "paper_id",
    "pdf_file",
    "canonical_title",
    "title_zh",
    "authors",
    "year",
    "venue",
    "language",
    "tags",
    "method_summary",
    "datasets",
    "memory_cues",
    "duplicate_group",
]


def _make_pdf(path, *, metadata=True, split_title=False):
    document = fitz.open()
    page = document.new_page()
    if split_title:
        page.insert_text((72, 70), "Evidence Based Paper", fontsize=20)
        page.insert_text((72, 94), "AI", fontsize=20)
        author_y = 125
    else:
        page.insert_text((72, 80), "Evidence Based Paper Retrieval", fontsize=20)
        author_y = 115
    page.insert_text((72, author_y), "Alice Zhang, Bob Li", fontsize=11)
    page.insert_textbox(
        fitz.Rect(72, 150, 520, 400),
        (
            "Abstract\n"
            "We propose a retrieval pipeline grounded in PDF evidence.\n"
            "Keywords: retrieval, evidence\n"
            "Published 2024\n"
            "doi:10.1234/example.2024.7"
        ),
        fontsize=11,
    )
    if metadata:
        document.set_metadata(
            {"title": "Evidence Based Paper Retrieval", "author": "Alice Zhang; Bob Li"}
        )
    document.save(path)
    document.close()


def _write_catalog(path, pdf_file="registered.pdf"):
    row = {
        "paper_id": "registered",
        "pdf_file": pdf_file,
        "canonical_title": "Registered Paper",
        "title_zh": "",
        "authors": "Author",
        "year": "2023",
        "venue": "Journal",
        "language": "en",
        "tags": "retrieval",
        "method_summary": "method",
        "datasets": "",
        "memory_cues": "cue",
        "duplicate_group": "",
    }
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CATALOG_FIELDS)
        writer.writeheader()
        writer.writerow(row)


def test_extract_pdf_facts_preserves_page_evidence(tmp_path):
    pdf = tmp_path / "paper.pdf"
    _make_pdf(pdf)

    facts = extract_pdf_facts(pdf)

    assert facts.page_count == 1
    assert facts.title.value == "Evidence Based Paper Retrieval"
    assert facts.title.source == "pdf_metadata.title"
    assert facts.authors.value == "Alice Zhang; Bob Li"
    assert facts.abstract.page == 1
    assert facts.abstract.value.startswith("We propose a retrieval pipeline")
    assert facts.doi.value == "10.1234/example.2024.7"
    assert facts.doi.page == 1
    assert facts.year.value == "2024"
    assert facts.language.value == "en"


def test_extract_pdf_facts_prefers_article_doi_and_published_year(tmp_path):
    pdf = tmp_path / "journal-version.pdf"
    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 72), "Journal Paper", fontsize=20)
    page.insert_text(
        (72, 120),
        "Earlier version DOI: 10.1109/OLD.2024.1234567",
        fontsize=10,
    )
    page.insert_text(
        (72, 160),
        "Accepted for publication. Citation information: DOI 10.1109/TMC.2026.7654321",
        fontsize=10,
    )
    page.insert_text((72, 190), "© 2026 IEEE", fontsize=10)
    document.set_metadata(
        {
            "subject": "IEEE Transactions;10.1109/TMC.2026.7654321",
            "creationDate": "D:20240901000000Z",
        }
    )
    document.save(pdf)
    document.close()

    facts = extract_pdf_facts(pdf)

    assert facts.doi.value == "10.1109/TMC.2026.7654321"
    assert facts.doi.source == "pdf_metadata.subject"
    assert facts.year.value == "2026"
    assert facts.year.source == "page_text"
    assert facts.year.confidence == "medium"


def test_prepare_draft_uses_layout_fallback_and_does_not_change_catalog(tmp_path):
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    registered = inbox / "registered.pdf"
    candidate = inbox / "candidate.pdf"
    _make_pdf(registered)
    _make_pdf(candidate, metadata=False, split_title=True)
    catalog = tmp_path / "paper_catalog.csv"
    _write_catalog(catalog)
    original_catalog = catalog.read_bytes()

    draft = prepare_profile_draft(candidate, inbox, catalog)

    assert draft["status"] == "needs_review"
    assert draft["file"]["pdf_file"] == "candidate.pdf"
    assert len(draft["file"]["sha256"]) == 64
    assert draft["extracted_fields"]["canonical_title"]["source"] == "page_layout"
    assert draft["proposed_profile"]["canonical_title"] == "Evidence Based Paper AI"
    assert draft["extracted_fields"]["authors"]["confidence"] == "low"
    assert draft["proposed_profile"]["authors"] == ""
    assert "authors" in draft["review_required_fields"]
    assert draft["proposed_profile"]["tags"] == ""
    assert draft["proposed_profile"]["method_summary"] == ""
    assert catalog.read_bytes() == original_catalog


def test_prepare_draft_reports_exact_duplicate_before_import(tmp_path):
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    registered = inbox / "registered.pdf"
    candidate = inbox / "candidate.pdf"
    duplicate = inbox / "copy.pdf"
    _make_pdf(registered)
    _make_pdf(candidate)
    shutil.copyfile(candidate, duplicate)
    catalog = tmp_path / "paper_catalog.csv"
    _write_catalog(catalog)

    draft = prepare_profile_draft(candidate, inbox, catalog)

    assert draft["status"] == "duplicate_review_required"
    assert draft["exact_duplicate_files"] == ("copy.pdf",)


def test_prepare_draft_requires_pdf_inside_inbox(tmp_path):
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    outside = tmp_path / "outside.pdf"
    _make_pdf(outside)
    catalog = tmp_path / "paper_catalog.csv"
    _write_catalog(catalog)

    with pytest.raises(ValueError, match="inside the configured inbox"):
        prepare_profile_draft(outside, inbox, catalog)


def test_write_profile_draft_refuses_accidental_overwrite(tmp_path):
    output = tmp_path / "drafts" / "paper.json"
    draft = {"status": "needs_review", "title": "中文论文"}

    written = write_profile_draft(draft, output)
    with pytest.raises(FileExistsError, match="--force"):
        write_profile_draft(draft, output)
    write_profile_draft({"status": "updated"}, output, overwrite=True)

    assert written == output.resolve()
    assert json.loads(output.read_text(encoding="utf-8"))["status"] == "updated"


def test_prepare_paper_cli_writes_draft_without_changing_catalog(tmp_path):
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    registered = inbox / "registered.pdf"
    candidate = inbox / "候选论文.pdf"
    _make_pdf(registered)
    _make_pdf(candidate)
    catalog = tmp_path / "paper_catalog.csv"
    _write_catalog(catalog)
    original_catalog = catalog.read_bytes()
    output = tmp_path / "drafts" / "candidate.json"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.paper_assistant",
            "--catalog",
            str(catalog),
            "prepare-paper",
            str(candidate),
            "--inbox",
            str(inbox),
            "--output",
            str(output),
        ],
        capture_output=True,
        encoding="utf-8",
        check=False,
    )

    summary = json.loads(result.stdout)
    saved = json.loads(output.read_text(encoding="utf-8"))
    assert result.returncode == 0
    assert summary["status"] == "needs_review"
    assert summary["pdf_file"] == "候选论文.pdf"
    assert saved["proposed_profile"]["canonical_title"] == (
        "Evidence Based Paper Retrieval"
    )
    assert catalog.read_bytes() == original_catalog
