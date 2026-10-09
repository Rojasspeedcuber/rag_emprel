import numpy as np
from types import SimpleNamespace
from unittest.mock import Mock

from src import embeddings


class FakeModel:
    def __init__(self):
        self.inputs = []

    def encode(self, texts, **kwargs):
        self.inputs.append((texts, kwargs))
        return np.array([[3.0, 4.0] for _ in texts], dtype="float32")


def test_embed_documents_passes_texts_directly(monkeypatch):
    model = FakeModel()
    monkeypatch.setattr(embeddings, "_load_model", lambda: model)

    vectors = embeddings.embed_documents(["texto"])

    assert model.inputs[0][0] == ["texto"]
    assert vectors.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), [1.0])


def test_embed_query_returns_single_normalized_vector(monkeypatch):
    model = FakeModel()
    monkeypatch.setattr(embeddings, "_load_model", lambda: model)

    vector = embeddings.embed_query("pergunta")

    assert model.inputs[0][0] == ["pergunta"]
    assert vector.shape == (1, 2)
    np.testing.assert_allclose(np.linalg.norm(vector, axis=1), [1.0])


def test_embed_documents_handles_empty_input():
    assert embeddings.embed_documents([]).shape == (0, 0)


def test_blocked_sklearn_finder_exposes_import_hooks():
    spec = embeddings._BlockedSklearnFinder().find_spec("sklearn.metrics", None)
    module = spec.loader.create_module(spec)
    spec.loader.exec_module(module)
    assert callable(module.pairwise_distances)
    assert callable(module.roc_curve)


def test_bypass_installs_finder_when_dll_blocked(monkeypatch):
    import sys

    monkeypatch.setattr(embeddings, "_sklearn_is_blocked", lambda: True)
    sys.meta_path[:] = [
        finder
        for finder in sys.meta_path
        if not isinstance(finder, embeddings._BlockedSklearnFinder)
    ]
    try:
        embeddings._bypass_blocked_sklearn()
        assert any(
            isinstance(finder, embeddings._BlockedSklearnFinder) for finder in sys.meta_path
        )
    finally:
        sys.meta_path[:] = [
            finder
            for finder in sys.meta_path
            if not isinstance(finder, embeddings._BlockedSklearnFinder)
        ]


def test_is_ollama_model_detects_prefix(monkeypatch):
    monkeypatch.setattr(embeddings, "EMBEDDING_MODEL", "ollama:nomic-embed-text")
    assert embeddings._is_ollama()
    monkeypatch.setattr(embeddings, "EMBEDDING_MODEL", "paraphrase-multilingual-MiniLM-L12-v2")
    assert not embeddings._is_ollama()


def test_ollama_embed_uses_openai_compatible_api(monkeypatch):
    monkeypatch.setattr(embeddings, "EMBEDDING_MODEL", "ollama:nomic-embed-text")
    monkeypatch.setattr(embeddings, "OLLAMA_BASE_URL", "http://localhost:11434/v1")
    create = Mock(
        return_value=SimpleNamespace(
            data=[
                SimpleNamespace(embedding=[3.0, 4.0]),
                SimpleNamespace(embedding=[1.0, 0.0]),
            ]
        )
    )
    client = SimpleNamespace(embeddings=SimpleNamespace(create=create))
    monkeypatch.setattr(embeddings, "_get_ollama_client", lambda: client)

    vectors = embeddings.embed_documents(["a", "b"])

    assert create.call_args.kwargs["model"] == "nomic-embed-text"
    assert create.call_args.kwargs["input"] == ["a", "b"]
    assert vectors.shape == (2, 2)
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), [1.0, 1.0])
