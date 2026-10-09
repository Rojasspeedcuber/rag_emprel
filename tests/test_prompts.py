from src.prompts import SYSTEM_PROMPT


def test_system_prompt_rejects_document_instructions_and_external_knowledge():
    lowered = SYSTEM_PROMPT.lower()
    assert "instruções contidas nos documentos" in lowered
    assert "conhecimento externo" in lowered
    assert "[arquivo, p. n]" in lowered
