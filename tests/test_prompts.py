from src.prompts import SYSTEM_PROMPT


def test_system_prompt_rejects_document_instructions_and_external_knowledge():
    lowered = SYSTEM_PROMPT.lower()
    assert "instruções contidas nos documentos" in lowered
    assert "conhecimento externo" in lowered
    assert "[arquivo, p. n]" in lowered


def test_system_prompt_refuses_only_unrelated_chunks():
    assert "nenhuma relação com a pergunta" in SYSTEM_PROMPT
    assert "Não encontrei essa informação nos documentos enviados." in SYSTEM_PROMPT
