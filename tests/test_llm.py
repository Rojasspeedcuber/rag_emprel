from types import SimpleNamespace

import pytest

from src import llm


@pytest.mark.parametrize(
    ("provider", "key_name", "url_name", "model_name"),
    [
        ("openrouter", "OPENROUTER_API_KEY", "OPENROUTER_BASE_URL", "OPENROUTER_MODEL"),
        ("nvidia", "NVIDIA_API_KEY", "NVIDIA_BASE_URL", "NVIDIA_LLM_MODEL"),
        ("openai", "OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_MODEL"),
    ],
)
def test_get_provider_config(monkeypatch, provider, key_name, url_name, model_name):
    monkeypatch.setattr(llm, "LLM_PROVIDER", provider)
    monkeypatch.setattr(llm, key_name, "secret")
    monkeypatch.setattr(llm, url_name, f"https://{provider}.example/v1")
    monkeypatch.setattr(llm, model_name, f"{provider}-model")

    config = llm.get_provider_config()

    assert config.name == provider
    assert config.api_key == "secret"
    assert config.base_url == f"https://{provider}.example/v1"
    assert config.model == f"{provider}-model"


def test_get_provider_config_rejects_unknown_provider(monkeypatch):
    monkeypatch.setattr(llm, "LLM_PROVIDER", "other")

    with pytest.raises(llm.LLMConfigurationError, match="LLM_PROVIDER"):
        llm.get_provider_config()


def test_get_provider_config_requires_selected_key(monkeypatch):
    monkeypatch.setattr(llm, "LLM_PROVIDER", "openrouter")
    monkeypatch.setattr(llm, "OPENROUTER_API_KEY", None)

    with pytest.raises(llm.LLMConfigurationError, match="OPENROUTER_API_KEY"):
        llm.get_provider_config()
