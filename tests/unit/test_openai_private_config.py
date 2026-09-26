"""Tests for selecting a private OpenAI-compatible endpoint from settings."""

from types import SimpleNamespace

from src.libs.llm.openai_llm import OpenAILLM


def _settings(base_url: str):
    return SimpleNamespace(
        llm=SimpleNamespace(
            model="compatible-model",
            temperature=0.0,
            max_tokens=64,
            api_key="settings-key",
            base_url=base_url,
            azure_endpoint=None,
            api_version=None,
        )
    )


def test_openai_compatible_base_url_is_loaded_from_settings():
    llm = OpenAILLM(_settings("https://compatible.api.example/v1"))

    assert llm.base_url == "https://compatible.api.example/v1"
    assert llm._use_azure_auth is False


def test_explicit_base_url_overrides_settings():
    llm = OpenAILLM(
        _settings("https://settings.api.example/v1"),
        base_url="https://explicit.api.example/v1",
    )

    assert llm.base_url == "https://explicit.api.example/v1"
