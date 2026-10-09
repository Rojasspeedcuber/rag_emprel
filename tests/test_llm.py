from dataclasses import FrozenInstanceError
import importlib

import pytest

from src import config
from src import llm


@pytest.mark.parametrize(
    ("provider", "api_key", "base_url", "model"),
    [
        ("openrouter", "openrouter-key", "https://openrouter.example/v1", "openrouter-model"),
        ("nvidia", "nvidia-key", "https://nvidia.example/v1", "nvidia-model"),
        ("openai", "openai-key", "https://openai.example/v1", "openai-model"),
    ],
)
def test_get_provider_config_returns_selected_provider(
    monkeypatch, provider, api_key, base_url, model
):
    monkeypatch.setattr(llm, "LLM_PROVIDER", provider)
    monkeypatch.setattr(llm, "OPENROUTER_API_KEY", "")
    monkeypatch.setattr(llm, "OPENROUTER_BASE_URL", "https://openrouter.example/v1")
    monkeypatch.setattr(llm, "OPENROUTER_MODEL", "openrouter-model")
    monkeypatch.setattr(llm, "NVIDIA_API_KEY", "")
    monkeypatch.setattr(llm, "NVIDIA_BASE_URL", "https://nvidia.example/v1")
    monkeypatch.setattr(llm, "NVIDIA_LLM_MODEL", "nvidia-model")
    monkeypatch.setattr(llm, "OPENAI_API_KEY", "")
    monkeypatch.setattr(llm, "OPENAI_BASE_URL", "https://openai.example/v1")
    monkeypatch.setattr(llm, "OPENAI_MODEL", "openai-model")
    monkeypatch.setattr(llm, f"{provider.upper()}_API_KEY", api_key)

    config = llm.get_provider_config()

    assert config == llm.ProviderConfig(provider, api_key, base_url, model)


def test_get_provider_config_rejects_unknown_provider(monkeypatch):
    monkeypatch.setattr(llm, "LLM_PROVIDER", "unsupported")

    with pytest.raises(
        llm.LLMConfigurationError, match="LLM_PROVIDER.*unsupported"
    ):
        llm.get_provider_config()


@pytest.mark.parametrize(
    ("provider", "key_name"),
    [
        ("openrouter", "OPENROUTER_API_KEY"),
        ("nvidia", "NVIDIA_API_KEY"),
        ("openai", "OPENAI_API_KEY"),
    ],
)
def test_get_provider_config_names_missing_selected_provider_key(
    monkeypatch, provider, key_name
):
    monkeypatch.setattr(llm, "LLM_PROVIDER", provider)
    monkeypatch.setattr(llm, "OPENROUTER_API_KEY", "")
    monkeypatch.setattr(llm, "NVIDIA_API_KEY", "")
    monkeypatch.setattr(llm, "OPENAI_API_KEY", "")

    with pytest.raises(llm.LLMConfigurationError, match=key_name):
        llm.get_provider_config()


def test_provider_config_is_frozen():
    config = llm.ProviderConfig("openai", "key", "https://openai.example/v1", "model")

    with pytest.raises(FrozenInstanceError):
        config.model = "different-model"


@pytest.mark.parametrize(
    "error_type", [llm.LLMConfigurationError, llm.LLMProviderError]
)
def test_llm_errors_are_runtime_errors(error_type):
    assert issubclass(error_type, RuntimeError)


def test_llm_provider_is_stripped_and_lowercased(monkeypatch):
    with monkeypatch.context() as context:
        context.setenv("LLM_PROVIDER", "  OpenAI  ")
        importlib.reload(config)
        assert config.LLM_PROVIDER == "openai"

    importlib.reload(config)
