from dataclasses import dataclass

from openai import OpenAI

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
        "openrouter": (OPENROUTER_API_KEY, OPENROUTER_BASE_URL, OPENROUTER_MODEL, "OPENROUTER_API_KEY"),
        "nvidia": (NVIDIA_API_KEY, NVIDIA_BASE_URL, NVIDIA_LLM_MODEL, "NVIDIA_API_KEY"),
        "openai": (OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_MODEL, "OPENAI_API_KEY"),
    }
    if LLM_PROVIDER not in providers:
        raise LLMConfigurationError(
            "LLM_PROVIDER deve ser openrouter, nvidia ou openai."
        )
    api_key, base_url, model, key_name = providers[LLM_PROVIDER]
    if not api_key:
        raise LLMConfigurationError(f"Configure {key_name} para usar {LLM_PROVIDER}.")
    return ProviderConfig(LLM_PROVIDER, api_key, base_url, model)


def _get_client(config: ProviderConfig) -> OpenAI:
    return OpenAI(api_key=config.api_key, base_url=config.base_url)


def answer_question(question: str, retrieved_chunks, history: list[dict] | None = None) -> str:
    """
    Gera a resposta com base exclusivamente nos trechos recuperados,
    usando o provedor selecionado por LLM_PROVIDER.
    O histórico é usado apenas como contexto conversacional leve, nunca como fonte.
    """
    context = build_context(retrieved_chunks)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Inclui as últimas 4 mensagens do histórico para contexto conversacional,
    # mas a regra do system prompt impede usá-las como fonte factual.
    if history:
        messages.extend(
            message
            for message in history[-4:]
            if message.get("role") in ("user", "assistant")
        )
    messages.append(
        {"role": "user", "content": USER_TEMPLATE.format(question=question, context=context)}
    )

    config = get_provider_config()
    try:
        response = _get_client(config).chat.completions.create(
            model=config.model,
            messages=messages,
            temperature=TEMPERATURE,
            top_p=TOP_P,
        )
        content = response.choices[0].message.content
        if not content or not content.strip():
            raise ValueError("resposta vazia")
        return content.strip()
    except LLMConfigurationError:
        raise
    except Exception as exc:
        raise LLMProviderError(
            f"O provedor {config.name} está indisponível no momento."
        ) from exc
