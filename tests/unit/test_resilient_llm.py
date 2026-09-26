"""Tests for bounded LLM retries, circuit breaking, and provider fallback."""

from __future__ import annotations

from src.libs.llm.base_llm import ChatResponse, Message
from src.libs.llm.resilient_llm import (
    ResilientChatModel,
    ResilientLLMError,
    RetryPolicy,
    response_resilience_diagnostics,
)


class ProviderError(RuntimeError):
    def __init__(
        self,
        *,
        retryable: bool,
        status_code: int | None = None,
        retry_after_seconds: float | None = None,
    ):
        super().__init__("provider failed")
        self.retryable = retryable
        self.status_code = status_code
        self.retry_after_seconds = retry_after_seconds


class SequenceProvider:
    def __init__(self, actions, *, model="model"):
        self.actions = list(actions)
        self.model = model
        self.calls = []

    def chat(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        action = self.actions.pop(0)
        if isinstance(action, Exception):
            raise action
        return ChatResponse(action, kwargs.get("model", self.model), {"total_tokens": 3})


def _policy(**overrides):
    values = {
        "max_attempts": 2,
        "base_delay_seconds": 0.5,
        "max_delay_seconds": 2.0,
        "circuit_failure_threshold": 2,
        "circuit_cooldown_seconds": 10.0,
    }
    values.update(overrides)
    return RetryPolicy(**values)


def test_retryable_failure_retries_with_retry_after_then_succeeds():
    primary = SequenceProvider(
        [
            ProviderError(
                retryable=True, status_code=429, retry_after_seconds=1.25
            ),
            "ok",
        ]
    )
    delays = []
    llm = ResilientChatModel(
        primary,
        primary_name="deepseek",
        policy=_policy(),
        sleep=delays.append,
    )

    trace = object()
    response = llm.chat([Message("user", "hello")], trace=trace)
    diagnostics = response_resilience_diagnostics(response)

    assert response.content == "ok"
    assert len(primary.calls) == 2
    assert all(call[1]["trace"] is trace for call in primary.calls)
    assert delays == [1.25]
    assert diagnostics["primary_attempts"] == 2
    assert diagnostics["fallback_used"] is False
    assert diagnostics["failures"][0]["status_code"] == 429


def test_exhausted_primary_uses_fallback_default_model():
    primary = SequenceProvider(
        [ProviderError(retryable=True), ProviderError(retryable=True)]
    )
    fallback = SequenceProvider(["fallback"], model="qwen-plus")
    llm = ResilientChatModel(
        primary,
        primary_name="deepseek",
        fallback=fallback,
        fallback_name="qwen",
        fallback_model="qwen-plus",
        policy=_policy(),
        sleep=lambda seconds: None,
    )

    response = llm.chat(
        [Message("user", "hello")], model="deepseek-v4-pro"
    )
    diagnostics = response_resilience_diagnostics(response)

    assert response.content == "fallback"
    assert fallback.calls[0][1]["model"] == "qwen-plus"
    assert diagnostics["fallback_used"] is True
    assert diagnostics["fallback_model"] == "qwen-plus"
    assert diagnostics["fallback_attempts"] == 1


def test_non_retryable_error_does_not_switch_provider():
    primary = SequenceProvider([ProviderError(retryable=False, status_code=401)])
    fallback = SequenceProvider(["must not run"])
    llm = ResilientChatModel(
        primary,
        primary_name="deepseek",
        fallback=fallback,
        fallback_name="qwen",
        policy=_policy(),
    )

    try:
        llm.chat([Message("user", "hello")])
    except ResilientLLMError as error:
        assert error.diagnostics["failures"][0]["status_code"] == 401
    else:
        raise AssertionError("Expected a non-retryable provider error")

    assert len(primary.calls) == 1
    assert fallback.calls == []


def test_open_circuit_skips_primary_and_recovers_after_cooldown():
    clock = [100.0]
    primary = SequenceProvider(
        [
            ProviderError(retryable=True),
            ProviderError(retryable=True),
            "recovered",
        ]
    )
    fallback = SequenceProvider(["first", "while-open"])
    llm = ResilientChatModel(
        primary,
        primary_name="deepseek",
        fallback=fallback,
        fallback_name="qwen",
        fallback_model="qwen-plus",
        policy=_policy(max_attempts=1, circuit_failure_threshold=2),
        sleep=lambda seconds: None,
        monotonic=lambda: clock[0],
    )

    assert llm.chat([Message("user", "one")]).content == "first"
    assert llm.chat([Message("user", "two")]).content == "while-open"
    primary_calls_when_opened = len(primary.calls)
    fallback.actions.append("circuit-open")
    third = llm.chat([Message("user", "three")])

    assert third.content == "circuit-open"
    assert len(primary.calls) == primary_calls_when_opened
    assert response_resilience_diagnostics(third)["circuit_state"] == "open"

    clock[0] += 11.0
    recovered = llm.chat([Message("user", "four")])

    assert recovered.content == "recovered"
    assert response_resilience_diagnostics(recovered)["circuit_state"] == "half_open_probe"


def test_all_providers_failed_returns_sanitized_diagnostics():
    primary = SequenceProvider([ProviderError(retryable=True)])
    fallback = SequenceProvider([ProviderError(retryable=True)])
    llm = ResilientChatModel(
        primary,
        primary_name="deepseek",
        fallback=fallback,
        fallback_name="qwen",
        policy=_policy(max_attempts=1),
    )

    try:
        llm.chat([Message("user", "hello")])
    except ResilientLLMError as error:
        assert error.diagnostics["fallback_used"] is True
        assert [item["provider"] for item in error.diagnostics["failures"]] == [
            "deepseek",
            "qwen",
        ]
        assert "provider failed" not in str(error.diagnostics)
    else:
        raise AssertionError("Expected all providers to fail")
