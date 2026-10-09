import os
import re
from dataclasses import dataclass

import numpy as np

from src.config import (
    NVIDIA_API_KEY,
    NVIDIA_BASE_URL,
    TOP_K,
    SIMILARITY_THRESHOLD,
)
from src.embeddings import embed_query
from src.ingest import Corpus, Chunk

RERANKER_MODEL = os.getenv("NVIDIA_RERANKER_MODEL", "")
_CODE = re.compile(r"\b[A-Z]{1,4}\d{2,6}\b", re.IGNORECASE)

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


def _codes(text: str) -> list[str]:
    found = []
    for match in _CODE.finditer(text):
        code = match.group(0).upper()
        if code not in found:
            found.append(code)
    return found


def _dedupe(results: list[RetrievedChunk]) -> list[RetrievedChunk]:
    seen = set()
    unique = []
    for item in results:
        key = re.sub(r"\s+", " ", item.chunk.text).strip().lower()
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
    return unique


def _lexical(corpus: Corpus, codes: list[str]) -> list[RetrievedChunk]:
    scored = []
    for chunk in corpus.chunks:
        text = chunk.text.upper()
        count = sum(text.count(code) for code in codes)
        if count:
            scored.append((count, RetrievedChunk(chunk=chunk, similarity=1.0)))
    scored.sort(key=lambda item: item[0], reverse=True)
    return _dedupe([item[1] for item in scored])


def retrieve(corpus: Corpus, query: str) -> tuple[list[RetrievedChunk], str | None]:
    """
    Retorna (trechos, mensagem_de_fallback).
    - corpus vazio → ([], EMPTY_CORPUS_MESSAGE)
    - código exato no texto → trechos que contêm o código, sem limiar vetorial
    - abaixo do limiar → ([], FALLBACK_MESSAGE)
    - caso contrário → (trechos, None)
    """
    if corpus.empty:
        return [], EMPTY_CORPUS_MESSAGE

    codes = _codes(query)
    if codes:
        lexical = _lexical(corpus, codes)[:TOP_K]
        if lexical:
            return lexical, None

    results = _dedupe(_search(corpus, query, top_k=TOP_K * 3))[:TOP_K]
    if not results:
        return [], FALLBACK_MESSAGE

    best = max(r.similarity for r in results)
    if best < SIMILARITY_THRESHOLD:
        return [], FALLBACK_MESSAGE

    return _rerank(query, results), None
    return results, None
