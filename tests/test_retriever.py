from unittest.mock import patch

from src.ingest import Chunk, Corpus
from src.retriever import retrieve, RetrievedChunk


def make_corpus(empty=False, n_chunks=3):
    if empty:
        return Corpus(chunks=[], store=None, empty=True)
    chunks = [Chunk(text=f"trecho {i}", source="doc.pdf", page=i + 1) for i in range(n_chunks)]
    return Corpus(chunks=chunks, store=object(), empty=False)  # store mockado


def fake_search(corpus, query, top_k=5):
    return [
        RetrievedChunk(chunk=Chunk(text="t1", source="d.pdf", page=1), similarity=0.9),
        RetrievedChunk(chunk=Chunk(text="t2", source="d.pdf", page=2), similarity=0.5),
    ]


def fake_search_low_similarity(corpus, query, top_k=5):
    return [RetrievedChunk(chunk=Chunk(text="t", source="d.pdf", page=1), similarity=0.1)]


def fake_search_no_results(corpus, query, top_k=5):
    return []


def test_empty_corpus_returns_nothing():
    assert retrieve(make_corpus(empty=True), "pergunta") == []


def test_no_semantic_results_returns_empty_list():
    with patch("src.retriever._search", fake_search_no_results):
        assert retrieve(make_corpus(), "pergunta") == []


def test_low_similarity_still_returns_best_chunks():
    with patch("src.retriever._search", fake_search_low_similarity):
        chunks = retrieve(make_corpus(), "pergunta")

    assert [chunk.similarity for chunk in chunks] == [0.1]


def test_returns_ranked_chunks():
    with patch("src.retriever._search", fake_search):
        chunks = retrieve(make_corpus(), "pergunta")

    assert [chunk.similarity for chunk in chunks] == [0.9, 0.5]


def test_code_query_returns_exact_chunk_and_drops_duplicate():
    chunks = [
        Chunk(text="recebo o erro E0312 ao emitir", source="a.pdf", page=119),
        Chunk(text="O erro E3351 significa que o registro já foi inativado", source="faq.pdf", page=21),
        Chunk(text="O erro E3351 significa que o registro já foi inativado", source="faq_copy.pdf", page=21),
    ]
    corpus = Corpus(chunks=chunks, store=object(), empty=False)

    found = retrieve(corpus, "O que significa o erro E3351?")

    assert len(found) == 1
    assert found[0].chunk.page == 21
    assert "E3351" in found[0].chunk.text


def test_missing_code_falls_back_to_semantic_results():
    with patch("src.retriever._search", fake_search_low_similarity):
        chunks = retrieve(make_corpus(), "O que significa o erro E9999?")

    assert [chunk.chunk.text for chunk in chunks] == ["t"]


def test_semantic_results_drop_duplicate_text():
    duplicated = [
        RetrievedChunk(chunk=Chunk(text="mesmo", source="a.pdf", page=1), similarity=0.8),
        RetrievedChunk(chunk=Chunk(text="mesmo", source="b.pdf", page=1), similarity=0.8),
        RetrievedChunk(chunk=Chunk(text="outro", source="a.pdf", page=2), similarity=0.6),
    ]
    with patch("src.retriever._search", return_value=duplicated):
        found = retrieve(make_corpus(), "pergunta")

    assert [item.chunk.text for item in found] == ["mesmo", "outro"]


def test_acronym_question_includes_exact_term_chunk_first():
    chunks = [
        Chunk(text="Texto geral sobre emissão de notas.", source="a.pdf", page=2),
        Chunk(text="O que é o ADN e o que significa ambGer na nota?", source="a.pdf", page=41),
        Chunk(text="CADNASTRO não é a sigla pesquisada", source="a.pdf", page=3),
    ]
    corpus = Corpus(chunks=chunks, store=object(), empty=False)
    semantic = [RetrievedChunk(chunk=chunks[0], similarity=0.2)]

    with patch("src.retriever._search", return_value=semantic):
        found = retrieve(corpus, "O que é ADN?")

    assert [item.chunk.page for item in found] == [41, 2]


def test_acronym_absent_from_documents_still_returns_semantic_results():
    with patch("src.retriever._search", fake_search_low_similarity):
        chunks = retrieve(make_corpus(), "O que é XYZW?")

    assert [chunk.chunk.text for chunk in chunks] == ["t"]
