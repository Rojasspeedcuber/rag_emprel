from unittest.mock import Mock

from src.ingest import Chunk, Corpus
from src.retriever import FALLBACK_MESSAGE, RetrievedChunk
from src import service


def test_answer_refuses_without_calling_llm(monkeypatch):
    llm = Mock()
    monkeypatch.setattr(service, "retrieve", lambda corpus, question: ([], FALLBACK_MESSAGE))
    monkeypatch.setattr(service, "answer_question", llm)

    result = service.answer(Corpus(empty=False), "fora do acervo")

    assert result.content == FALLBACK_MESSAGE
    assert result.sources == []
    llm.assert_not_called()


def test_answer_returns_only_sources_sent_to_llm(monkeypatch):
    retrieved = [
        RetrievedChunk(Chunk("evidencia", "doc.pdf", 2), similarity=0.9)
    ]
    monkeypatch.setattr(service, "retrieve", lambda corpus, question: (retrieved, None))
    monkeypatch.setattr(service, "answer_question", lambda *args, **kwargs: "Resposta")

    result = service.answer(Corpus(empty=False), "pergunta")

    assert result.content == "Resposta"
    assert result.sources == [
        {"source": "doc.pdf", "page": 2, "text": "evidencia"}
    ]
