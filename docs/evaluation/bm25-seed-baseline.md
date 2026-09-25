# ScholarRAG BM25 seed baseline

## Purpose

This experiment verifies the first paper-level retrieval loop before adding embeddings. It answers a narrow engineering question: can the system load the paper catalog, merge duplicate PDF versions, rank papers from a fuzzy description, and calculate reproducible retrieval metrics?

## Setup

- Corpus: 10 local PDF files representing 9 unique papers
- Retrieval unit: one `PaperProfile` per `paper_id`
- Duplicate handling: the two TCDI PDF revisions share one result
- Query set: 27 generated seed queries, with 9 easy, 9 medium, and 9 hard cases
- Retriever: weighted BM25 over titles, tags, method summaries, datasets, memory cues, authors, year, and venue
- Metrics: Recall@1, Recall@3, and mean reciprocal rank (MRR)

## Result

| Split | Queries | Recall@1 | Recall@3 | MRR |
|---|---:|---:|---:|---:|
| Development | 18 | 1.000 | 1.000 | 1.000 |
| Test candidate | 9 | 1.000 | 1.000 | 1.000 |
| All seed queries | 27 | 1.000 | 1.000 | 1.000 |

## Interpretation

This is a pipeline baseline, not a trustworthy estimate of real-world accuracy. The catalog fields and seed queries were both derived from the same paper content, so they share distinctive terms and make the task easier. The result proves that the paper identity mapping, duplicate collapse, ranking, and metric calculation work correctly.

A resume-ready evaluation requires a frozen blind set written independently from memory. Those queries must not be edited after inspecting retrieval results. The next experiment will compare BM25, dense retrieval, and reciprocal-rank fusion on that set.

## Reproduce

```powershell
python -m src.paper_assistant evaluate
python -m src.paper_assistant evaluate --split dev
python -m src.paper_assistant evaluate --split test_candidate
```
