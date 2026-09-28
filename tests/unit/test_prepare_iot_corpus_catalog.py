from scripts.prepare_iot_corpus_catalog import final_decision


def test_final_decision_auto_admits_clean_machine_result() -> None:
    assert final_decision("pass") == "admit"


def test_final_decision_keeps_risk_record_pending_without_review() -> None:
    assert final_decision("needs_review") == "pending"
    assert final_decision("reject") == "pending"


def test_final_decision_honors_separate_manual_review() -> None:
    assert final_decision("reject", "exclude") == "exclude"
    assert final_decision("needs_review", "admit") == "admit"
