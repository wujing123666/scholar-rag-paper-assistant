from scripts.resolve_iot_fallbacks import alternate_pdf_urls


def test_alternate_pdf_urls_prefers_repository_and_deduplicates() -> None:
    work = {
        "locations": [
            {
                "pdf_url": "https://publisher.test/paper.pdf",
                "source": {"type": "journal"},
            },
            {
                "pdf_url": "https://repo.test/paper.pdf",
                "source": {"type": "repository"},
            },
            {
                "pdf_url": "https://repo.test/paper.pdf",
                "source": {"type": "repository"},
            },
        ]
    }

    assert alternate_pdf_urls(work, "https://failed.test/paper.pdf") == [
        "https://repo.test/paper.pdf",
        "https://publisher.test/paper.pdf",
    ]
