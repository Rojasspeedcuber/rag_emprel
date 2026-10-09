import importlib.machinery
import sys
import types
from functools import lru_cache

import numpy as np
from openai import OpenAI

from src.config import EMBEDDING_MODEL, OLLAMA_BASE_URL


def _is_ollama() -> bool:
    return EMBEDDING_MODEL.strip().lower().startswith("ollama:")


def _ollama_model_name() -> str:
    return EMBEDDING_MODEL.split(":", 1)[1]


class _BlockedSklearnLoader:
    def create_module(self, spec):
        module = types.ModuleType(spec.name)
        module.__path__ = []
        module.__package__ = spec.name
        return module

    def exec_module(self, module):
        if module.__name__ == "sklearn":
            module.__version__ = "0"
        if module.__name__ == "sklearn.metrics":

            def _blocked(*args, **kwargs):
                raise RuntimeError(
                    "scikit-learn bloqueado pela politica de Controle de Aplicativo"
                )

            module.pairwise_distances = _blocked
            module.roc_curve = _blocked


class _BlockedSklearnFinder:
    def find_spec(self, fullname, path, target=None):
        if fullname == "sklearn" or fullname.startswith("sklearn."):
            return importlib.machinery.ModuleSpec(
                fullname, _BlockedSklearnLoader(), is_package=True
            )
        return None


def _sklearn_is_blocked() -> bool:
    try:
        import sklearn  # noqa: F401
    except ImportError as exc:
        text = str(exc).lower()
        return (
            "dll load failed" in text
            or "application control" in text
            or "controle de aplicativo" in text
        )
    return False


def _bypass_blocked_sklearn() -> None:
    if any(isinstance(finder, _BlockedSklearnFinder) for finder in sys.meta_path):
        return
    if not _sklearn_is_blocked():
        return
    for name in list(sys.modules):
        if name == "sklearn" or name.startswith("sklearn."):
            del sys.modules[name]
    sys.meta_path.insert(0, _BlockedSklearnFinder())


@lru_cache(maxsize=1)
def _load_model():
    _bypass_blocked_sklearn()
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(EMBEDDING_MODEL)


@lru_cache(maxsize=1)
def _get_ollama_client() -> OpenAI:
    return OpenAI(api_key="ollama", base_url=OLLAMA_BASE_URL)


def _ollama_encode(texts: list[str]) -> np.ndarray:
    response = _get_ollama_client().embeddings.create(
        model=_ollama_model_name(),
        input=texts,
    )
    matrix = np.asarray([item.embedding for item in response.data], dtype="float32")
    return _normalize(matrix)


def _normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, 1e-12)


def _encode(texts: list[str]) -> np.ndarray:
    if not texts:
        return np.empty((0, 0), dtype="float32")
    if _is_ollama():
        return _ollama_encode(texts)
    vectors = _load_model().encode(texts, convert_to_numpy=True)
    return _normalize(np.asarray(vectors, dtype="float32"))


def embed_documents(texts: list[str]) -> np.ndarray:
    return _encode(list(texts))


def embed_query(query: str) -> np.ndarray:
    return _encode([query])
