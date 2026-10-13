from types import SimpleNamespace
from unittest.mock import Mock

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


def test_get_provider_config_ollama(monkeypatch):
    monkeypatch.setattr(llm, "LLM_PROVIDER", "ollama")
    monkeypatch.setattr(llm, "OLLAMA_BASE_URL", "http://localhost:11434/v1")
    monkeypatch.setattr(llm, "OLLAMA_MODEL", "qwen3:4b")

    config = llm.get_provider_config()

    assert config.name == "ollama"
    assert config.api_key  # chave fictícia aceita pelo endpoint compatível
    assert config.base_url == "http://localhost:11434/v1"
    assert config.model == "qwen3:4b"


def test_get_provider_config_rejects_unknown_provider(monkeypatch):
    monkeypatch.setattr(llm, "LLM_PROVIDER", "other")

    with pytest.raises(llm.LLMConfigurationError, match="LLM_PROVIDER"):
        llm.get_provider_config()


def test_get_provider_config_requires_selected_key(monkeypatch):
    monkeypatch.setattr(llm, "LLM_PROVIDER", "openrouter")
    monkeypatch.setattr(llm, "OPENROUTER_API_KEY", None)

    with pytest.raises(llm.LLMConfigurationError, match="OPENROUTER_API_KEY"):
        llm.get_provider_config()


def test_answer_question_uses_selected_provider(monkeypatch):
    completion = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="Resposta [doc.pdf, p. 1]"))]
    )
    create = Mock(return_value=completion)
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    config = llm.ProviderConfig("openrouter", "secret", "https://router/v1", "model")
    monkeypatch.setattr(llm, "get_provider_config", lambda: config)
    monkeypatch.setattr(llm, "_get_client", lambda selected: client)
    retrieved = [
        SimpleNamespace(
            chunk=SimpleNamespace(text="Evidencia", source="doc.pdf", page=1)
        )
    ]

    answer = llm.answer_question("Pergunta?", retrieved)

    assert answer == "Resposta [doc.pdf, p. 1]"
    kwargs = create.call_args.kwargs
    assert kwargs["model"] == "model"
    assert kwargs["temperature"] == llm.TEMPERATURE
    assert kwargs["top_p"] == llm.TOP_P


def test_answer_question_translates_provider_failure(monkeypatch):
    client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=lambda **kwargs: (_ for _ in ()).throw(RuntimeError("down")))
        )
    )
    config = llm.ProviderConfig("openai", "secret", "https://api.openai.com/v1", "model")
    monkeypatch.setattr(llm, "get_provider_config", lambda: config)
    monkeypatch.setattr(llm, "_get_client", lambda selected: client)

    with pytest.raises(llm.LLMProviderError, match="openai"):
        llm.answer_question("Pergunta?", [])


def test_answer_question_ollama_failure_mentions_server(monkeypatch):
    def fail(**kwargs):
        raise RuntimeError("connection refused")

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=fail)))
    config = llm.ProviderConfig("ollama", "ollama", "http://localhost:11434/v1", "qwen3:4b")
    monkeypatch.setattr(llm, "get_provider_config", lambda: config)
    monkeypatch.setattr(llm, "_get_client", lambda selected: client)

    with pytest.raises(llm.LLMProviderError, match="Ollama"):
        llm.answer_question("Pergunta?", [])
