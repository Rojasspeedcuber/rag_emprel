from unittest.mock import Mock

from src.ingest import Chunk, Corpus
from src.retriever import EMPTY_CORPUS_MESSAGE, FALLBACK_MESSAGE, RetrievedChunk
from src import service


def test_answer_without_chunks_does_not_call_llm(monkeypatch):
    llm = Mock()
    monkeypatch.setattr(service, "retrieve", lambda corpus, question: [])
    monkeypatch.setattr(service, "answer_question", llm)

    result = service.answer(Corpus(empty=True), "pergunta")

    assert result.content == EMPTY_CORPUS_MESSAGE
    assert result.sources == []
    llm.assert_not_called()


def test_answer_returns_only_sources_sent_to_llm(monkeypatch):
    retrieved = [
        RetrievedChunk(Chunk("evidencia", "doc.pdf", 2), similarity=0.9)
    ]
    monkeypatch.setattr(service, "retrieve", lambda corpus, question: retrieved)
    monkeypatch.setattr(service, "answer_question", lambda *args, **kwargs: "Resposta")

    result = service.answer(Corpus(empty=False), "pergunta")

    assert result.content == "Resposta"
    assert result.sources == [
        {"source": "doc.pdf", "page": 2, "text": "evidencia"}
    ]


def test_answer_refuses_off_topic_without_calling_llm(monkeypatch):
    weak = [RetrievedChunk(Chunk("receita de bolo", "doc.pdf", 5), similarity=0.2)]
    llm = Mock()
    monkeypatch.setattr(service, "retrieve", lambda corpus, question: weak)
    monkeypatch.setattr(service, "answer_question", llm)

    result = service.answer(Corpus(empty=False), "Como fazer um bolo?")

    assert result.content == FALLBACK_MESSAGE
    assert result.sources == []
    llm.assert_not_called()


def test_answer_accepts_exact_term_match_even_with_low_semantic_score(monkeypatch):
    exact = [RetrievedChunk(Chunk("O ADN é o ambiente", "doc.pdf", 41), similarity=1.0)]
    llm = Mock(return_value="Resposta")
    monkeypatch.setattr(service, "retrieve", lambda corpus, question: exact)
    monkeypatch.setattr(service, "answer_question", llm)

    result = service.answer(Corpus(empty=False), "O que é ADN?")

    assert result.content == "Resposta"
    llm.assert_called_once()
