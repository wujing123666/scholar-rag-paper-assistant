"""Tests for the production acceptance workload definition."""

from scripts.accept_iot_scale import build_requests


def test_default_acceptance_requests_use_distinct_phrasings() -> None:
    requests = build_requests(20)

    request_ids = [request_id for request_id, _ in requests]
    queries = [query for _, query in requests]
    topic_families = {request_id.rsplit("-", 1)[0] for request_id in request_ids}

    assert len(request_ids) == len(set(request_ids)) == 20
    assert len(queries) == len(set(queries)) == 20
    assert len(topic_families) == 10
