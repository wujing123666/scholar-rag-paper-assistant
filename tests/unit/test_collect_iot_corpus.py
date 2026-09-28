from scripts.collect_iot_corpus import (
    balanced_select,
    normalize_doi,
    reconstruct_abstract,
)


def _work(doi: str, title: str, bucket: str) -> dict:
    return {"doi": doi, "title": title, "_topic_bucket": bucket}


def test_reconstruct_abstract_uses_token_positions() -> None:
    assert reconstruct_abstract({"IoT": [0], "works": [2], "retrieval": [1]}) == (
        "IoT retrieval works"
    )


def test_normalize_doi_removes_url_prefix() -> None:
    assert normalize_doi("https://doi.org/10.1000/ABC") == "10.1000/abc"


def test_balanced_select_round_robins_and_deduplicates() -> None:
    duplicate = _work("https://doi.org/10.1/shared", "Shared", "security")
    buckets = {
        "core": [
            _work("https://doi.org/10.1/core-1", "Core 1", "core"),
            _work("https://doi.org/10.1/shared", "Shared", "core"),
        ],
        "security": [
            _work("https://doi.org/10.1/sec-1", "Security 1", "security"),
            duplicate,
        ],
    }

    selected = balanced_select(buckets, 4)

    assert [work["_selected_bucket"] for work in selected] == [
        "core",
        "security",
        "core",
    ]
    assert len(selected) == 3
    shared = next(work for work in selected if work["title"] == "Shared")
    assert shared["_matched_buckets"] == ["core", "security"]


def test_balanced_select_can_require_direct_pdf() -> None:
    without_pdf = _work("10.1/no-pdf", "No PDF", "core")
    with_pdf = _work("10.1/pdf", "PDF", "core")
    with_pdf["best_oa_location"] = {"pdf_url": "https://example.test/paper.pdf"}

    selected = balanced_select(
        {"core": [without_pdf, with_pdf]}, 1, require_direct_pdf=True
    )

    assert [work["title"] for work in selected] == ["PDF"]
