"""Historical public observations must preserve metric denominators."""

import json

import pytest

from scripts.recalculate_benchmarks import DATA, calculate


def test_published_metrics_recalculate_from_observations():
    data = json.loads(DATA.read_text(encoding="utf-8"))
    result = calculate(data)
    assert result["facet_ab"]["baseline"]["covered_aspects"] == 59
    assert result["facet_ab"]["facet"]["covered_aspects"] == 65
    assert result["coverage_gain_percentage_points"] == pytest.approx(100 * 6 / 72)
    assert result["claim_cost"]["baseline_tokens"] == 82148
    assert result["claim_cost"]["risk_cold_tokens"] == 72081
    assert result["claim_cost"]["cold_cache_hits"] == 1
    assert result["holdout"] == {
        "known": 100,
        "known_accepted": 63,
        "negative": 40,
        "negative_rejected": 40,
    }


def test_pairing_cannot_silently_change_aspect_denominator():
    data = json.loads(DATA.read_text(encoding="utf-8"))
    data["facet_ab"]["cases"][0]["facet"]["aspect_ranks"].pop()
    with pytest.raises(ValueError, match="denominators"):
        calculate(data)


def test_claim_audit_denominator_is_validated():
    data = json.loads(DATA.read_text(encoding="utf-8"))
    data["claim_audit"]["claims"] += 1
    with pytest.raises(ValueError, match="denominator"):
        calculate(data)


def test_live_evaluation_help_needs_no_credentials(monkeypatch):
    from scripts.evaluate_paper_demo import main

    monkeypatch.setattr("sys.argv", ["evaluate_paper_demo", "--help"])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 0
