from dataclasses import dataclass

import numpy as np
import requests

from src.config import (
    NVIDIA_API_KEY,
    NVIDIA_BASE_URL,
    RERANKER_MODEL,
    TOP_K,
    SIMILARITY_THRESHOLD,
)
from src.ingest import Corpus, Chunk, embed_query

FALLBACK_MESSAGE = "Não encontrei essa informação nos documentos enviados."
EMPTY_CORPUS_MESSAGE = "Nenhum documento foi enviado ainda. Envie PDFs pela barra lateral para começar."


@dataclass
class RetrievedChunk:
    chunk: Chunk
    similarity: float
    rerank_score: float | None = None


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


def _rerank(query: str, results: list[RetrievedChunk]) -> list[RetrievedChunk]:
    """Reordena os trechos com o reranker da NVIDIA. Se a API falhar, mantém a ordem original."""
    if not results:
        return results
    try:
        resp = requests.post(
            f"{NVIDIA_BASE_URL}/reranking",
            headers={
                "Authorization": f"Bearer {NVIDIA_API_KEY}",
                "Accept": "application/json",
            },
            json={
                "model": RERANKER_MODEL,
                "query": {"text": query},
                "passages": [{"text": r.chunk.text} for r in results],
                "truncate": "END",
            },
            timeout=30,
        )
        resp.raise_for_status()
        rankings = resp.json().get("rankings", [])
        reranked = []
        for item in rankings:
            r = results[item["index"]]
            r.rerank_score = item.get("score")
            reranked.append(r)
        # adiciona eventuais itens não presentes no reranking
        seen = {item["index"] for item in rankings}
        reranked.extend(r for i, r in enumerate(results) if i not in seen)
        return reranked
    except requests.RequestException:
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

    return _rerank(query, results), None