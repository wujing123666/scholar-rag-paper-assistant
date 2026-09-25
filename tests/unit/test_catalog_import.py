"""Tests for reviewed Paper Profile imports."""

from __future__ import annotations

import csv
import json
import shutil
import subprocess
import sys

import pytest

from src.paper_assistant.catalog import PaperCatalog
from src.paper_assistant.catalog_import import (
    apply_catalog_import,
    plan_catalog_import,
)
from src.paper_assistant.inventory import sha256_file
from src.paper_assistant.retriever import PaperBM25Retriever

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
    "read_date",
    "rating",
    "notes_source",
]


def _write_catalog(path):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CATALOG_FIELDS)
        writer.writeheader()
        writer.writerow(
            {
                "paper_id": "registered",
                "pdf_file": "registered.pdf",
                "canonical_title": "Registered Paper",
                "language": "en",
                "tags": "retrieval",
                "method_summary": "An existing method",
                "memory_cues": "existing paper",
                "notes_source": "manual",
            }
        )


def _write_draft(path, pdf, **profile_overrides):
    profile = {
        "paper_id": "new_paper_2026",
        "pdf_file": pdf.name,
        "canonical_title": "A New Paper",
        "title_zh": "一篇新论文",
        "authors": "Alice;Bob",
        "year": "2026",
        "venue": "TestConf",
        "language": "zh",
        "tags": "论文检索;知识管理",
        "method_summary": "构建经过人工确认的论文档案",
        "datasets": "TestSet",
        "memory_cues": "那篇演示受控增量入库的论文",
        "duplicate_group": "",
    }
    profile.update(profile_overrides)
    draft = {
        "schema_version": 1,
        "status": "needs_review",
        "file": {
            "pdf_file": pdf.name,
            "size_bytes": pdf.stat().st_size,
            "sha256": sha256_file(pdf),
            "page_count": 1,
        },
        "catalog_matches": [],
        "exact_duplicate_files": [],
        "proposed_profile": profile,
    }
    path.write_text(json.dumps(draft, ensure_ascii=False), encoding="utf-8")
    return draft


@pytest.fixture
def import_files(tmp_path):
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    pdf = inbox / "new-paper.pdf"
    pdf.write_bytes(b"%PDF-1.4\ncontrolled import test\n%%EOF")
    catalog = tmp_path / "paper_catalog.csv"
    _write_catalog(catalog)
    draft = tmp_path / "draft.json"
    _write_draft(draft, pdf)
    return inbox, pdf, catalog, draft


def test_import_preview_is_read_only_and_ready(import_files):
    inbox, _, catalog, draft = import_files
    original = catalog.read_bytes()

    plan = plan_catalog_import(draft, inbox, catalog)

    assert plan.ready is True
    assert plan.action == "append_new_paper"
    assert plan.paper_id == "new_paper_2026"
    assert plan.catalog_row_count_before == 1
    assert catalog.read_bytes() == original


def test_import_preview_blocks_incomplete_and_stale_drafts(import_files):
    inbox, pdf, catalog, draft = import_files
    _write_draft(draft, pdf, memory_cues="")
    pdf.write_bytes(b"%PDF-1.4\nchanged after draft\n%%EOF")

    plan = plan_catalog_import(draft, inbox, catalog)

    assert plan.ready is False
    assert "memory_cues" in plan.missing_fields
    assert any("content changed" in error for error in plan.errors)


def test_import_preview_blocks_registered_or_duplicate_identity(import_files):
    inbox, pdf, catalog, draft_path = import_files
    draft = _write_draft(
        draft_path,
        pdf,
        paper_id="registered",
        canonical_title="Registered Paper",
    )
    draft["status"] = "already_registered"
    draft["catalog_matches"] = ["registered"]
    draft_path.write_text(json.dumps(draft), encoding="utf-8")

    plan = plan_catalog_import(draft_path, inbox, catalog)

    assert plan.ready is False
    assert any("needs_review" in error for error in plan.errors)
    assert any("already registered" in error for error in plan.errors)
    assert any("paper_id already exists" in error for error in plan.errors)
    assert any("canonical_title" in error for error in plan.errors)


def test_import_preview_rechecks_new_exact_duplicates(import_files):
    inbox, pdf, catalog, draft = import_files
    shutil.copyfile(pdf, inbox / "late-copy.pdf")

    plan = plan_catalog_import(draft, inbox, catalog)

    assert plan.ready is False
    assert any("byte-identical" in error for error in plan.errors)


def test_confirmed_import_backs_up_validates_and_atomically_appends(import_files):
    inbox, _, catalog, draft = import_files
    backup_dir = catalog.parent / "backups"
    original = catalog.read_bytes()
    plan = plan_catalog_import(draft, inbox, catalog)

    result = apply_catalog_import(plan, inbox, catalog, backup_dir)

    assert result.status == "imported"
    assert result.catalog_row_count == 2
    assert (backup_dir / result.backup_file).read_bytes() == original
    imported = PaperCatalog.from_csv(catalog).get("new_paper_2026")
    assert imported.pdf_files == ("new-paper.pdf",)
    results = PaperBM25Retriever(PaperCatalog.from_csv(catalog)).search(
        "受控增量入库", top_k=2
    )
    assert results[0].paper.paper_id == "new_paper_2026"
    with catalog.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[-1]["notes_source"] == "人工确认的候选档案草稿"

    second_plan = plan_catalog_import(draft, inbox, catalog)
    assert second_plan.ready is False
    assert any("paper_id already exists" in error for error in second_plan.errors)


def test_confirmed_import_stops_if_catalog_changed_after_preview(import_files):
    inbox, _, catalog, draft = import_files
    plan = plan_catalog_import(draft, inbox, catalog)
    with catalog.open("a", encoding="utf-8") as handle:
        handle.write("\n")

    with pytest.raises(RuntimeError, match="changed after preview"):
        apply_catalog_import(plan, inbox, catalog, catalog.parent / "backups")


def test_confirmed_import_stops_if_pdf_changed_after_preview(import_files):
    inbox, pdf, catalog, draft = import_files
    plan = plan_catalog_import(draft, inbox, catalog)
    original_catalog = catalog.read_bytes()
    pdf.write_bytes(b"%PDF-1.4\nchanged after preview\n%%EOF")

    with pytest.raises(RuntimeError, match="PDF content changed after preview"):
        apply_catalog_import(plan, inbox, catalog, catalog.parent / "backups")

    assert catalog.read_bytes() == original_catalog
    assert not (catalog.parent / "backups").exists()


def test_import_cli_previews_before_explicit_confirmation(import_files):
    inbox, _, catalog, draft = import_files
    original = catalog.read_bytes()
    command = [
        sys.executable,
        "-m",
        "src.paper_assistant",
        "--catalog",
        str(catalog),
        "import-paper",
        str(draft),
        "--inbox",
        str(inbox),
        "--backup-dir",
        str(catalog.parent / "backups"),
    ]

    preview = subprocess.run(command, capture_output=True, encoding="utf-8", check=False)
    assert preview.returncode == 0
    assert json.loads(preview.stdout)["status"] == "ready"
    assert catalog.read_bytes() == original

    confirmed = subprocess.run(
        [*command, "--confirm"],
        capture_output=True,
        encoding="utf-8",
        check=False,
    )
    assert confirmed.returncode == 0
    assert json.loads(confirmed.stdout)["status"] == "imported"
    assert len(PaperCatalog.from_csv(catalog)) == 2


def test_import_rejects_a_non_object_draft(import_files):
    inbox, _, catalog, draft = import_files
    draft.write_text("[]", encoding="utf-8")

    with pytest.raises(ValueError, match="JSON object"):
        plan_catalog_import(draft, inbox, catalog)
