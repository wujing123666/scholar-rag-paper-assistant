"""Retry metadata exposed by HTTP-based LLM providers."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from src.libs.llm.base_llm import Message
from src.libs.llm.deepseek_llm import DeepSeekLLM, DeepSeekLLMError
from src.libs.llm.openai_llm import OpenAILLM, OpenAILLMError


def _settings():
    return SimpleNamespace(
        llm=SimpleNamespace(
            model="test-model",
            temperature=0.0,
            max_tokens=128,
            api_key="test-key",
            base_url=None,
            azure_endpoint=None,
            api_version=None,
        )
    )


@pytest.mark.parametrize(
    ("provider", "error_type", "retry_after"),
    [
        (OpenAILLM, OpenAILLMError, "1.5"),
        (DeepSeekLLM, DeepSeekLLMError, "2"),
    ],
)
def test_rate_limit_error_exposes_retry_metadata(
    provider, error_type, retry_after
):
    response = MagicMock()
    response.status_code = 429
    response.headers = {"Retry-After": retry_after}
    response.json.return_value = {"error": {"message": "Rate limited"}}
    response.text = "Rate limited"
    llm = provider(_settings())

    with patch("httpx.Client") as client:
        client.return_value.__enter__.return_value.post.return_value = response
        with pytest.raises(error_type) as raised:
            llm.chat([Message(role="user", content="Hello")])

    assert raised.value.status_code == 429
    assert raised.value.retryable is True
    assert raised.value.retry_after_seconds == float(retry_after)
