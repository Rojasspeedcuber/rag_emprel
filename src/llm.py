from openai import OpenAI, OpenAIError

from src.config import (
    NVIDIA_API_KEY,
    NVIDIA_BASE_URL,
    LLM_MODEL,
    TEMPERATURE,
    TOP_P,
)
from src.prompts import SYSTEM_PROMPT, USER_TEMPLATE, build_context


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
            model=LLM_MODEL,
            messages=messages,
            temperature=TEMPERATURE,
            top_p=TOP_P,
        )
        return resp.choices[0].message.content.strip()
    except OpenAIError as e:
        return f"Erro ao chamar a API do LLM: {e}. Verifique sua NVIDIA_API_KEY e o modelo configurado."
