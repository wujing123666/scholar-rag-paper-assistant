"""Retry, circuit-breaker, and provider-fallback wrapper for chat models."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from src.libs.llm.base_llm import ChatResponse, Message


@dataclass(frozen=True)
class RetryPolicy:
    """Bounded retry and circuit-breaker settings."""

    max_attempts: int = 2
    base_delay_seconds: float = 0.25
    max_delay_seconds: float = 2.0
    circuit_failure_threshold: int = 3
    circuit_cooldown_seconds: float = 30.0

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be at least one")
        if self.base_delay_seconds < 0 or self.max_delay_seconds < 0:
            raise ValueError("retry delays cannot be negative")
        if self.max_delay_seconds < self.base_delay_seconds:
            raise ValueError("max_delay_seconds cannot be smaller than base delay")
        if self.circuit_failure_threshold < 1:
            raise ValueError("circuit_failure_threshold must be at least one")
        if self.circuit_cooldown_seconds < 0:
            raise ValueError("circuit_cooldown_seconds cannot be negative")


class ResilientLLMError(RuntimeError):
    """Raised after the allowed providers and retries are exhausted."""

    def __init__(self, message: str, diagnostics: dict[str, Any]) -> None:
        super().__init__(message)
        self.diagnostics = diagnostics


class ResilientChatModel:
    """Wrap one primary model with bounded retries and an optional fallback."""

    def __init__(
        self,
        primary: Any,
        *,
        primary_name: str,
        fallback: Any | None = None,
        fallback_name: str | None = None,
        fallback_model: str | None = None,
        policy: RetryPolicy | None = None,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        if fallback is not None and not fallback_name:
            raise ValueError("fallback_name is required when fallback is configured")
        self.primary = primary
        self.primary_name = primary_name
        self.fallback = fallback
        self.fallback_name = fallback_name
        self.fallback_model = fallback_model
        self.policy = policy or RetryPolicy()
        self._sleep = sleep
        self._monotonic = monotonic
        self._state_lock = threading.Lock()
        self._consecutive_failures = 0
        self._circuit_opened_at: float | None = None
        self._half_open_probe = False

    @staticmethod
    def _is_retryable(error: Exception) -> bool:
        marker = getattr(error, "retryable", None)
        if isinstance(marker, bool):
            return marker
        return isinstance(error, (TimeoutError, ConnectionError))

    @staticmethod
    def _failure_record(provider: str, error: Exception) -> dict[str, Any]:
        return {
            "provider": provider,
            "error_type": type(error).__name__,
            "status_code": getattr(error, "status_code", None),
            "retryable": ResilientChatModel._is_retryable(error),
        }

    def _circuit_permission(self) -> tuple[bool, str]:
        with self._state_lock:
            if self._circuit_opened_at is None:
                return True, "closed"
            elapsed = self._monotonic() - self._circuit_opened_at
            if elapsed < self.policy.circuit_cooldown_seconds:
                return False, "open"
            if self._half_open_probe:
                return False, "half_open_wait"
            self._half_open_probe = True
            return True, "half_open_probe"

    def _record_primary_success(self) -> None:
        with self._state_lock:
            self._consecutive_failures = 0
            self._circuit_opened_at = None
            self._half_open_probe = False

    def _record_primary_failure(self) -> None:
        with self._state_lock:
            self._consecutive_failures += 1
            if self._consecutive_failures >= self.policy.circuit_failure_threshold:
                self._circuit_opened_at = self._monotonic()
            self._half_open_probe = False

    def _retry_delay(self, error: Exception, attempt: int) -> float:
        exponential = self.policy.base_delay_seconds * (2 ** max(attempt - 1, 0))
        retry_after = getattr(error, "retry_after_seconds", None)
        requested = float(retry_after) if isinstance(retry_after, (int, float)) else 0.0
        return min(max(exponential, requested), self.policy.max_delay_seconds)

    def _call_with_retries(
        self,
        provider: Any,
        provider_name: str,
        messages: list[Message],
        kwargs: dict[str, Any],
        diagnostics: dict[str, Any],
        attempt_field: str,
    ) -> ChatResponse:
        last_error: Exception | None = None
        for attempt in range(1, self.policy.max_attempts + 1):
            diagnostics[attempt_field] = attempt
            try:
                return provider.chat(messages, **kwargs)
            except Exception as error:
                last_error = error
                diagnostics["failures"].append(
                    self._failure_record(provider_name, error)
                )
                if not self._is_retryable(error) or attempt >= self.policy.max_attempts:
                    break
                delay = self._retry_delay(error, attempt)
                diagnostics["retry_delays_seconds"].append(delay)
                self._sleep(delay)
        assert last_error is not None
        raise last_error

    @staticmethod
    def _response_with_diagnostics(
        response: ChatResponse, diagnostics: dict[str, Any]
    ) -> ChatResponse:
        return ChatResponse(
            content=response.content,
            model=response.model,
            usage=response.usage,
            raw_response={
                "provider_response": response.raw_response,
                "resilience": diagnostics,
            },
        )

    def chat(
        self,
        messages: list[Message],
        trace: Any | None = None,
        **kwargs: Any,
    ) -> ChatResponse:
        call_kwargs = dict(kwargs)
        if trace is not None:
            call_kwargs["trace"] = trace
        diagnostics: dict[str, Any] = {
            "primary_provider": self.primary_name,
            "primary_model": getattr(self.primary, "model", None),
            "fallback_provider": self.fallback_name,
            "fallback_model": self.fallback_model
            or getattr(self.fallback, "model", None),
            "primary_attempts": 0,
            "fallback_attempts": 0,
            "fallback_used": False,
            "circuit_state": "closed",
            "failures": [],
            "retry_delays_seconds": [],
        }
        primary_allowed, circuit_state = self._circuit_permission()
        diagnostics["circuit_state"] = circuit_state
        if primary_allowed:
            try:
                response = self._call_with_retries(
                    self.primary,
                    self.primary_name,
                    messages,
                    call_kwargs,
                    diagnostics,
                    "primary_attempts",
                )
                self._record_primary_success()
                return self._response_with_diagnostics(response, diagnostics)
            except Exception as error:
                if not self._is_retryable(error):
                    self._record_primary_success()
                    raise ResilientLLMError(
                        "Primary LLM request failed with a non-retryable error",
                        diagnostics,
                    ) from error
                self._record_primary_failure()

        if self.fallback is None:
            raise ResilientLLMError(
                "Primary LLM is unavailable and no fallback is configured",
                diagnostics,
            )

        diagnostics["fallback_used"] = True
        fallback_kwargs = dict(call_kwargs)
        fallback_model = self.fallback_model or getattr(self.fallback, "model", None)
        if fallback_model:
            fallback_kwargs["model"] = fallback_model
        else:
            fallback_kwargs.pop("model", None)
        try:
            response = self._call_with_retries(
                self.fallback,
                str(self.fallback_name),
                messages,
                fallback_kwargs,
                diagnostics,
                "fallback_attempts",
            )
            return self._response_with_diagnostics(response, diagnostics)
        except Exception as error:
            raise ResilientLLMError(
                "Primary and fallback LLM providers are unavailable",
                diagnostics,
            ) from error


def response_resilience_diagnostics(response: ChatResponse) -> dict[str, Any] | None:
    """Extract sanitized resilience diagnostics from a wrapped response."""
    raw = response.raw_response
    if not isinstance(raw, dict):
        return None
    diagnostics = raw.get("resilience")
    return diagnostics if isinstance(diagnostics, dict) else None
