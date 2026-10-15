"""Regenerate the original synthetic demonstration PDFs and catalog.

Requires reportlab (only needed to regenerate; PDFs are checked in).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer

ROOT = Path(__file__).resolve().parent


def generate() -> None:
    papers = json.loads((ROOT / "papers.json").read_text(encoding="utf-8"))
    (ROOT / "pdfs").mkdir(exist_ok=True)
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            "Body",
            fontName="Helvetica",
            fontSize=11,
            leading=17,
            spaceAfter=14,
            textColor=colors.HexColor("#25364a"),
        )
    )
    styles["Title"].fontSize = 19
    styles["Title"].leading = 25

    def footer(canvas, document):
        canvas.setFont("Helvetica", 9)
        canvas.setFillColor(colors.HexColor("#566b7d"))
        canvas.drawString(22 * mm, 16 * mm, "ScholarRAG | SYNTHETIC DEMO - NOT A RESEARCH PAPER")
        canvas.drawRightString(188 * mm, 16 * mm, f"PDF page {document.page}")

    for paper in papers:
        story = []
        for index, page in enumerate(paper["pages"]):
            if index:
                story.append(PageBreak())
            story.extend(
                [
                    Paragraph(escape(paper["canonical_title"]), styles["Title"]),
                    Spacer(1, 8 * mm),
                    Paragraph(page["heading"], styles["Heading2"]),
                ]
            )
            story.extend(Paragraph(escape(text), styles["Body"]) for text in page["paragraphs"])
        SimpleDocTemplate(
            str(ROOT / "pdfs" / paper["pdf_file"]),
            pagesize=A4,
            leftMargin=22 * mm,
            rightMargin=22 * mm,
            topMargin=22 * mm,
            bottomMargin=25 * mm,
            title=paper["canonical_title"],
            author="ScholarRAG synthetic demo",
            invariant=1,
        ).build(story, onFirstPage=footer, onLaterPages=footer)
    columns = [
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
    with (ROOT / "paper_catalog.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for paper in papers:
            row = {key: paper.get(key, "") for key in columns}
            row.update(
                authors="Synthetic demo", year="", venue="Synthetic demonstration", language="en"
            )
            writer.writerow(row)


if __name__ == "__main__":
    generate()
