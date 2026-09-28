from scripts.audit_iot_corpus import classify_audit, title_token_coverage


def test_title_token_coverage_ignores_common_words() -> None:
    coverage = title_token_coverage(
        "A Secure Architecture for the Internet of Things",
        "Secure Architecture Internet of Things",
    )

    assert coverage == 1.0


def test_classify_audit_passes_complete_relevant_article() -> None:
    status, flags, core_hit, bucket_hit = classify_audit(
        title="Federated Learning for Internet of Things",
        bucket="federated_iot",
        page_count=12,
        text_char_count=35_000,
        text_page_ratio=1.0,
        title_coverage=0.9,
        opening_text="This paper studies federated learning across IoT devices.",
        encrypted=False,
        repaired=False,
    )

    assert status == "pass"
    assert flags == ()
    assert core_hit
    assert bucket_hit


def test_classify_audit_rejects_correction_title() -> None:
    status, flags, _, _ = classify_audit(
        title="Correction to: Internet of Things Security",
        bucket="iot_security",
        page_count=2,
        text_char_count=1_000,
        text_page_ratio=1.0,
        title_coverage=1.0,
        opening_text="Internet of Things security correction.",
        encrypted=False,
        repaired=False,
    )

    assert status == "reject"
    assert "correction_or_retraction_title" in flags
