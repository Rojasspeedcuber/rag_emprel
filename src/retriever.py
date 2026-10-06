from dataclasses import dataclass

import numpy as np

from src.config import (
    TOP_K,
    SIMILARITY_THRESHOLD,
)
from src.embeddings import embed_query
from src.ingest import Corpus, Chunk

FALLBACK_MESSAGE = "Não encontrei essa informação nos documentos enviados."
EMPTY_CORPUS_MESSAGE = "Nenhum documento foi enviado ainda. Envie PDFs pela barra lateral para começar."


@dataclass
class RetrievedChunk:
    chunk: Chunk
    similarity: float


def _search(corpus: Corpus, query: str, top_k: int = TOP_K) -> list[RetrievedChunk]:
    """Busca vetorial por similaridade de cosseno (IndexFlatIP com vetores L2-normalizados)."""
    if corpus.empty or corpus.index is None:
        return []
    q = embed_query(query)
    k = min(top_k, len(corpus.chunks))
    scores, indices = corpus.index.search(q, k)
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx != -1:
            results.append(RetrievedChunk(chunk=corpus.chunks[idx], similarity=float(score)))
    return results


def retrieve(corpus: Corpus, query: str) -> tuple[list[RetrievedChunk], str | None]:
    """
    Retorna (trechos, mensagem_de_fallback).
    - corpus vazio → ([], EMPTY_CORPUS_MESSAGE)
    - abaixo do limiar → ([], FALLBACK_MESSAGE)
    - caso contrário → (trechos, None)
    """
    if corpus.empty:
        return [], EMPTY_CORPUS_MESSAGE

    results = _search(corpus, query)
    if not results:
        return [], FALLBACK_MESSAGE

    best = max(r.similarity for r in results)
    if best < SIMILARITY_THRESHOLD:
        return [], FALLBACK_MESSAGE

    return results, None
