import re
from dataclasses import dataclass

import numpy as np

from src.config import TOP_K
from src.embeddings import embed_query
from src.ingest import Corpus, Chunk

_CODE = re.compile(r"\b[A-Z]{1,4}\d{2,6}\b", re.IGNORECASE)
_ACRONYM = re.compile(r"\b[A-Z]{2,6}\b")
_COMMON_CAPS = {
    "DE", "DO", "DA", "EM", "NO", "NA", "UM", "UMA", "SE", "SIM", "NAO", "QUE", "COMO",
    "PARA", "POR", "QUAL", "QUAIS", "ONDE", "QUANDO", "MEU", "MINHA", "ME",
}
_ACRONYM_HITS = 2

EMPTY_CORPUS_MESSAGE = "Nenhum documento foi enviado ainda. Envie PDFs pela barra lateral para começar."
FALLBACK_MESSAGE = "Não encontrei essa informação nos documentos enviados."


@dataclass
class RetrievedChunk:
    chunk: Chunk
    similarity: float


def _search(corpus: Corpus, query: str, top_k: int = TOP_K) -> list[RetrievedChunk]:
    """Busca vetorial por similaridade de cosseno no banco vetorial (vetores L2-normalizados)."""
    if corpus.empty or corpus.store is None:
        return []
    q = embed_query(query)[0].tolist()
    hits = corpus.store.search(q, limit=min(top_k, len(corpus.chunks)))
    return [
        RetrievedChunk(
            chunk=Chunk(text=payload["text"], source=payload["source"], page=payload["page"]),
            similarity=float(score),
        )
        for score, payload in hits
    ]


def _codes(text: str) -> list[str]:
    found = []
    for match in _CODE.finditer(text):
        code = match.group(0).upper()
        if code not in found:
            found.append(code)
    return found


def _acronyms(text: str) -> list[str]:
    found = []
    for match in _ACRONYM.finditer(text):
        term = match.group(0)
        if term not in _COMMON_CAPS and term not in found:
            found.append(term)
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


def _lexical(corpus: Corpus, terms: list[str]) -> list[RetrievedChunk]:
    patterns = [re.compile(rf"(?<![A-Z0-9]){re.escape(term)}(?![A-Z0-9])") for term in terms]
    scored = []
    for chunk in corpus.chunks:
        text = chunk.text.upper()
        count = sum(len(pattern.findall(text)) for pattern in patterns)
        if count:
            scored.append((count, RetrievedChunk(chunk=chunk, similarity=1.0)))
    scored.sort(key=lambda item: item[0], reverse=True)
    return _dedupe([item[1] for item in scored])


def retrieve(corpus: Corpus, query: str) -> list[RetrievedChunk]:
    """
    Retorna sempre os trechos mais relevantes para a pergunta, sem recusar por limiar.
    - código exato no texto → trechos que contêm o código
    - sigla da pergunta no texto → até 2 trechos com a sigla, seguidos da busca vetorial
    - caso contrário → busca vetorial
    """
    if corpus.empty:
        return []

    codes = _codes(query)
    if codes:
        lexical = _lexical(corpus, codes)[:TOP_K]
        if lexical:
            return lexical

    results = _dedupe(_search(corpus, query, top_k=TOP_K * 3))

    acronyms = _acronyms(query)
    if acronyms:
        results = _dedupe(_lexical(corpus, acronyms)[:_ACRONYM_HITS] + results)

    return results[:TOP_K]
