from unittest.mock import patch

import numpy as np

from src.ingest import Chunk, Corpus
from src.retriever import (
    retrieve,
    FALLBACK_MESSAGE,
    EMPTY_CORPUS_MESSAGE,
    RetrievedChunk,
)


def make_corpus(empty=False, n_chunks=3):
    if empty:
        return Corpus(chunks=[], index=None, empty=True)
    chunks = [Chunk(text=f"trecho {i}", source="doc.pdf", page=i + 1) for i in range(n_chunks)]
    return Corpus(chunks=chunks, index=object(), empty=False)  # index mockado


def fake_search(corpus, query, top_k=5):
    # similaridades controladas para testes de limiar
    return [
        RetrievedChunk(chunk=Chunk(text="t1", source="d.pdf", page=1), similarity=0.9),
        RetrievedChunk(chunk=Chunk(text="t2", source="d.pdf", page=2), similarity=0.5),
    ]


def fake_search_low_similarity(corpus, query, top_k=5):
    return [RetrievedChunk(chunk=Chunk(text="t", source="d.pdf", page=1), similarity=0.1)]


def fake_search_no_results(corpus, query, top_k=5):
    return []


def test_empty_corpus_returns_empty_message():
    chunks, msg = retrieve(make_corpus(empty=True), "pergunta")
    assert chunks == []
    assert msg == EMPTY_CORPUS_MESSAGE


def test_no_results_returns_fallback():
    with patch("src.retriever._search", fake_search_no_results):
        chunks, msg = retrieve(make_corpus(), "pergunta")
    assert chunks == []
    assert msg == FALLBACK_MESSAGE


def test_below_threshold_returns_fallback():
    with patch("src.retriever._search", fake_search_low_similarity):
        chunks, msg = retrieve(make_corpus(), "pergunta")
    assert chunks == []
    assert msg == FALLBACK_MESSAGE


def test_above_threshold_returns_chunks_without_rerank_when_api_fails():
    # rerank falha (requests não configurado) → mantém resultados
    with patch("src.retriever._search", fake_search), \
         patch("src.retriever._rerank", side_effect=lambda q, r: r):
        chunks, msg = retrieve(make_corpus(), "pergunta")
    assert msg is None
    assert len(chunks) == 2
    assert chunks[0].similarity == 0.9
