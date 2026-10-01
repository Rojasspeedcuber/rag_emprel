SYSTEM_PROMPT = """Você é um assistente que responde perguntas EXCLUSIVAMENTE com base nos trechos de documentos fornecidos abaixo.

REGRAS OBRIGATÓRIAS (violar qualquer uma é falha grave):
1. Responda APENAS com informações presentes nos trechos fornecidos. NUNCA use conhecimento externo, suposições ou generalizações.
2. Se a resposta não estiver nos trechos, responda exatamente: "Não encontrei essa informação nos documentos."
3. Ao responder, cite a fonte de cada afirmação no formato [arquivo, p. N], usando os metadados dos trechos.
4. Não copie mais do que o necessário: sintetize, mas sem acrescentar nada que não esteja nos trechos.
5. Responda em português brasileiro claro e objetivo.

Os trechos relevantes virão na mensagem do usuário, marcados com nome do arquivo e página."""

USER_TEMPLATE = """Pergunta do usuário: {question}

Trechos dos documentos:
{context}

Responda à pergunta APENAS com base nos trechos acima, citando as fontes."""


def build_context(retrieved_chunks) -> str:
    """Formata os trechos recuperados com metadados para o prompt."""
    parts = []
    for i, r in enumerate(retrieved_chunks, 1):
        parts.append(
            f"--- Trecho {i} ---\n"
            f"Arquivo: {r.chunk.source}\n"
            f"Página: {r.chunk.page}\n"
            f"{r.chunk.text}"
        )
    return "\n\n".join(parts)
