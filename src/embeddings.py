from functools import lru_cache

import numpy as np

from src.config import EMBEDDING_MODEL


@lru_cache(maxsize=1)
def _get_model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(EMBEDDING_MODEL)


def _encode(prefixed_texts: list[str]) -> np.ndarray:
    if not prefixed_texts:
        return np.empty((0, 0), dtype=np.float32)

    vectors = np.asarray(
        _get_model().encode(prefixed_texts, convert_to_numpy=True),
        dtype=np.float32,
    )
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norms, 1e-12)


def embed_documents(texts: list[str]) -> np.ndarray:
    return _encode([f"passage: {text}" for text in texts])


def embed_query(query: str) -> np.ndarray:
    return _encode([f"query: {query}"])
