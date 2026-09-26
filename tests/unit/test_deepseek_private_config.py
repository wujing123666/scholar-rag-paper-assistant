"""Tests for loading DeepSeek connection values from private settings."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

from src.libs.llm.deepseek_llm import DeepSeekLLM


def _settings(*, api_key=None, base_url=None):
    return SimpleNamespace(
        llm=SimpleNamespace(
            model="deepseek-chat",
            temperature=0.0,
            max_tokens=1024,
            api_key=api_key,
            base_url=base_url,
        )
    )


def test_deepseek_reads_key_and_url_from_private_settings():
    settings = _settings(
        api_key="settings-key",
        base_url="https://private.deepseek.example",
    )
    with patch.dict("os.environ", {"DEEPSEEK_API_KEY": "env-key"}):
        llm = DeepSeekLLM(settings)

    assert llm.api_key == "settings-key"
    assert llm.base_url == "https://private.deepseek.example"


def test_explicit_values_override_private_settings():
    llm = DeepSeekLLM(
        _settings(api_key="settings-key", base_url="https://settings.example"),
        api_key="explicit-key",
        base_url="https://explicit.example",
    )

    assert llm.api_key == "explicit-key"
    assert llm.base_url == "https://explicit.example"


def test_environment_remains_a_fallback():
    with patch.dict("os.environ", {"DEEPSEEK_API_KEY": "env-key"}):
        llm = DeepSeekLLM(_settings())

    assert llm.api_key == "env-key"
    assert llm.base_url == DeepSeekLLM.DEFAULT_BASE_URL
