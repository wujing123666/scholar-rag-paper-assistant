"""Tests for local paper inventory and duplicate detection."""

from __future__ import annotations

import csv
import json
import subprocess
import sys

from src.paper_assistant.inventory import (
    build_paper_inventory,
    sha256_file,
    write_inventory_report,
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


def _catalog_row(paper_id, pdf_file):
    return {
        "paper_id": paper_id,
        "pdf_file": pdf_file,
        "canonical_title": paper_id.title(),
        "title_zh": paper_id,
        "authors": "Author",
        "year": "2026",
        "venue": "Journal",
        "language": "zh",
        "tags": "测试",
        "method_summary": "测试方法",
        "datasets": "",
        "memory_cues": "测试线索",
        "duplicate_group": "",
    }


def _write_catalog(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CATALOG_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def test_inventory_finds_unregistered_missing_duplicates_and_versions(tmp_path):
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    (inbox / "known.pdf").write_bytes(b"same pdf bytes")
    (inbox / "duplicate.pdf").write_bytes(b"same pdf bytes")
    (inbox / "new.PDF").write_bytes(b"new paper")
    (inbox / "notes.txt").write_text("ignored", encoding="utf-8")
    catalog = tmp_path / "catalog.csv"
    _write_catalog(
        catalog,
        [
            _catalog_row("paper-a", "known.pdf"),
            _catalog_row("paper-a", "missing-revision.pdf"),
        ],
    )

    report = build_paper_inventory(inbox, catalog)

    assert report.status == "attention_required"
    summary = report.summary_dict()
    assert "inbox" not in summary
    assert "catalog" not in summary
    assert "files" not in summary
    assert "sha256" not in json.dumps(summary)
    assert report.scanned_pdf_count == 3
    assert report.catalog_paper_count == 1
    assert report.catalog_file_entry_count == 2
    assert report.catalog_unique_path_count == 2
    assert report.registered_pdf_count == 1
    assert report.unregistered_files == ("duplicate.pdf", "new.PDF")
    assert report.missing_files == ("missing-revision.pdf",)
    assert report.exact_duplicate_groups[0].files == (
        "duplicate.pdf",
        "known.pdf",
    )
    assert report.paper_version_groups[0].paper_id == "paper-a"
    assert report.paper_version_groups[0].files == (
        "known.pdf",
        "missing-revision.pdf",
    )
    assert report.files[0].sha256 == sha256_file(inbox / report.files[0].path)


def test_inventory_reports_clean_library_and_writes_json(tmp_path):
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    (inbox / "paper.pdf").write_bytes(b"paper")
    catalog = tmp_path / "catalog.csv"
    _write_catalog(catalog, [_catalog_row("paper", "paper.pdf")])

    report = build_paper_inventory(inbox, catalog)
    output = write_inventory_report(report, tmp_path / "reports" / "inventory.json")

    saved = json.loads(output.read_text(encoding="utf-8"))
    assert report.status == "ok"
    assert saved["status"] == "ok"
    assert saved["files"][0]["paper_ids"] == ["paper"]


def test_inventory_detects_one_catalog_path_assigned_to_two_papers(tmp_path):
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    (inbox / "shared.pdf").write_bytes(b"paper")
    catalog = tmp_path / "catalog.csv"
    _write_catalog(
        catalog,
        [
            _catalog_row("paper-a", "shared.pdf"),
            _catalog_row("paper-b", "shared.pdf"),
        ],
    )

    report = build_paper_inventory(inbox, catalog)

    assert report.status == "attention_required"
    assert report.catalog_path_conflicts[0].path == "shared.pdf"
    assert report.catalog_path_conflicts[0].paper_ids == ("paper-a", "paper-b")
    assert report.catalog_file_entry_count == 2
    assert report.catalog_unique_path_count == 1


def test_inventory_cli_emits_utf8_and_writes_optional_report(tmp_path):
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    (inbox / "中文论文.pdf").write_bytes(b"paper")
    catalog = tmp_path / "catalog.csv"
    _write_catalog(catalog, [_catalog_row("paper", "中文论文.pdf")])
    output = tmp_path / "inventory.json"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.paper_assistant",
            "--catalog",
            str(catalog),
            "inventory",
            "--inbox",
            str(inbox),
            "--output",
            str(output),
        ],
        capture_output=True,
        encoding="utf-8",
        check=False,
    )

    assert result.returncode == 0
    assert "中文论文.pdf" not in result.stdout
    assert str(inbox) not in result.stdout
    assert "sha256" not in result.stdout
    saved = json.loads(output.read_text(encoding="utf-8"))
    assert saved["status"] == "ok"
    assert saved["files"][0]["path"] == "中文论文.pdf"


def test_inventory_cli_returns_one_when_attention_is_required(tmp_path):
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    (inbox / "未登记.pdf").write_bytes(b"paper")
    catalog = tmp_path / "catalog.csv"
    _write_catalog(catalog, [_catalog_row("missing", "缺失.pdf")])

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.paper_assistant",
            "--catalog",
            str(catalog),
            "inventory",
            "--inbox",
            str(inbox),
        ],
        capture_output=True,
        encoding="utf-8",
        check=False,
    )

    output = json.loads(result.stdout)
    assert result.returncode == 1
    assert output["status"] == "attention_required"
    assert output["unregistered_files"] == ["未登记.pdf"]
    assert output["missing_files"] == ["缺失.pdf"]
