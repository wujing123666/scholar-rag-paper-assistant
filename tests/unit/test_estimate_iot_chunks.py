from scripts.estimate_iot_chunks import summarize_chunk_rows


def test_summarize_chunk_rows_excludes_failed_papers() -> None:
    summary = summarize_chunk_rows(
        [
            {"status": "success", "chunk_count": 10, "embedding_characters": 1000},
            {"status": "success", "chunk_count": 20, "embedding_characters": 2000},
            {"status": "failed", "chunk_count": 0, "embedding_characters": 0},
        ]
    )

    assert summary == {
        "papers": 3,
        "successful_papers": 2,
        "failed_papers": 1,
        "chunks": 30,
        "chunks_median_per_paper": 15.0,
        "chunks_p95_per_paper": 20,
        "embedding_characters": 3000,
    }
