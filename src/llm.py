from dataclasses import dataclass

from openai import OpenAI, OpenAIError

from src.config import (
    LLM_PROVIDER,
    NVIDIA_API_KEY,
    NVIDIA_BASE_URL,
    NVIDIA_LLM_MODEL,
    OPENAI_API_KEY,
    OPENAI_BASE_URL,
    OPENAI_MODEL,
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    OPENROUTER_MODEL,
    TEMPERATURE,
    TOP_P,
)
from src.prompts import SYSTEM_PROMPT, USER_TEMPLATE, build_context


class LLMConfigurationError(RuntimeError):
    pass


class LLMProviderError(RuntimeError):
    pass


@dataclass(frozen=True)
class ProviderConfig:
    name: str
    api_key: str
    base_url: str
    model: str


def get_provider_config() -> ProviderConfig:
    providers = {
        "openrouter": ProviderConfig(
            "openrouter", OPENROUTER_API_KEY, OPENROUTER_BASE_URL, OPENROUTER_MODEL
        ),
        "nvidia": ProviderConfig(
            "nvidia", NVIDIA_API_KEY, NVIDIA_BASE_URL, NVIDIA_LLM_MODEL
        ),
        "openai": ProviderConfig("openai", OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_MODEL),
    }
    api_key_names = {
        "openrouter": "OPENROUTER_API_KEY",
        "nvidia": "NVIDIA_API_KEY",
        "openai": "OPENAI_API_KEY",
    }

    if LLM_PROVIDER not in providers:
        raise LLMConfigurationError(f"Unknown LLM_PROVIDER: {LLM_PROVIDER}")

    config = providers[LLM_PROVIDER]
    if not config.api_key:
        raise LLMConfigurationError(f"Missing {api_key_names[LLM_PROVIDER]}")
    return config


def _get_client() -> OpenAI:
    return OpenAI(api_key=NVIDIA_API_KEY, base_url=NVIDIA_BASE_URL)


def answer_question(question: str, retrieved_chunks, history: list[dict] | None = None) -> str:
    """
    Gera a resposta com base exclusivamente nos trechos recuperados.
    O histórico é usado apenas como contexto conversacional leve, nunca como fonte.
    """
    context = build_context(retrieved_chunks)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Inclui as últimas 4 mensagens do histórico para contexto conversacional,
    # mas a regra nº 1 do system prompt impede usá-las como fonte factual.
    if history:
        trimmed = [m for m in history if m.get("role") in ("user", "assistant")][-4:]
        messages.extend(trimmed)

    messages.append(
        {"role": "user", "content": USER_TEMPLATE.format(question=question, context=context)}
    )

    client = _get_client()
    try:
        resp = client.chat.completions.create(
            model=NVIDIA_LLM_MODEL,
            messages=messages,
            temperature=TEMPERATURE,
            top_p=TOP_P,
        )
        return resp.choices[0].message.content.strip()
    except OpenAIError as e:
        return f"Erro ao chamar a API do LLM: {e}. Verifique sua NVIDIA_API_KEY e o modelo configurado."
