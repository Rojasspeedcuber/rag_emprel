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


def test_above_threshold_returns_ranked_chunks():
    with patch("src.retriever._search", fake_search):
        chunks, msg = retrieve(make_corpus(), "pergunta")

    assert msg is None
    assert [chunk.similarity for chunk in chunks] == [0.9, 0.5]


def test_code_query_returns_exact_chunk_and_drops_duplicate():
    chunks = [
        Chunk(text="recebo o erro E0312 ao emitir", source="a.pdf", page=119),
        Chunk(text="O erro E3351 significa que o registro já foi inativado", source="faq.pdf", page=21),
        Chunk(text="O erro E3351 significa que o registro já foi inativado", source="faq_copy.pdf", page=21),
    ]
    corpus = Corpus(chunks=chunks, index=object(), empty=False)

    found, msg = retrieve(corpus, "O que significa o erro E3351?")

    assert msg is None
    assert len(found) == 1
    assert found[0].chunk.page == 21
    assert "E3351" in found[0].chunk.text


def test_missing_code_falls_back_to_semantic_threshold():
    with patch("src.retriever._search", fake_search_low_similarity):
        chunks, msg = retrieve(make_corpus(), "O que significa o erro E9999?")

    assert chunks == []
    assert msg == FALLBACK_MESSAGE


def test_semantic_results_drop_duplicate_text():
    duplicated = [
        RetrievedChunk(chunk=Chunk(text="mesmo", source="a.pdf", page=1), similarity=0.8),
        RetrievedChunk(chunk=Chunk(text="mesmo", source="b.pdf", page=1), similarity=0.8),
        RetrievedChunk(chunk=Chunk(text="outro", source="a.pdf", page=2), similarity=0.6),
    ]
    with patch("src.retriever._search", return_value=duplicated):
        found, msg = retrieve(make_corpus(), "pergunta")

    assert msg is None
    assert [item.chunk.text for item in found] == ["mesmo", "outro"]


def test_threshold_accepts_exact_boundary():
    boundary = [
        RetrievedChunk(
            chunk=Chunk(text="t", source="d.pdf", page=1),
            similarity=0.35,
        )
    ]
    with patch("src.retriever._search", return_value=boundary), \
         patch("src.retriever.SIMILARITY_THRESHOLD", 0.35):
        chunks, msg = retrieve(make_corpus(), "pergunta")

    assert chunks == boundary
    assert msg is None
