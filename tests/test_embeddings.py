import sys
from types import SimpleNamespace

import numpy as np
import pytest

import src.embeddings as embeddings


class FakeModel:
    def __init__(self):
        self.calls = []

    def encode(self, texts, **kwargs):
        self.calls.append((texts, kwargs))
        vectors = {
            "passage: primeiro": [3.0, 4.0],
            "passage: segundo": [0.0, 5.0],
            "query: pergunta": [5.0, 12.0],
        }
        return np.array([vectors[text] for text in texts], dtype=np.float64)


@pytest.fixture
def fake_model(monkeypatch):
    model = FakeModel()
    constructor_calls = []

    def fake_sentence_transformer(model_name):
        constructor_calls.append(model_name)
        return model

    embeddings._get_model.cache_clear()
    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        SimpleNamespace(SentenceTransformer=fake_sentence_transformer),
    )
    yield model, constructor_calls
    embeddings._get_model.cache_clear()


def test_embed_documents_prefixes_normalizes_and_returns_float32(fake_model):
    model, constructor_calls = fake_model

    result = embeddings.embed_documents(["primeiro", "segundo"])

    assert model.calls == [
        (
            ["passage: primeiro", "passage: segundo"],
            {"convert_to_numpy": True},
        )
    ]
    assert constructor_calls == [embeddings.EMBEDDING_MODEL]
    assert result.dtype == np.float32
    np.testing.assert_allclose(result, [[0.6, 0.8], [0.0, 1.0]])
    np.testing.assert_allclose(np.linalg.norm(result, axis=1), [1.0, 1.0])


def test_embed_query_prefixes_and_returns_single_normalized_row(fake_model):
    model, constructor_calls = fake_model

    result = embeddings.embed_query("pergunta")
    embeddings.embed_query("pergunta")

    assert model.calls[0] == (
        ["query: pergunta"],
        {"convert_to_numpy": True},
    )
    assert constructor_calls == [embeddings.EMBEDDING_MODEL]
    assert result.shape == (1, 2)
    assert result.dtype == np.float32
    np.testing.assert_allclose(result, [[5.0 / 13.0, 12.0 / 13.0]])
    np.testing.assert_allclose(np.linalg.norm(result, axis=1), [1.0])


def test_embed_documents_empty_returns_empty_float32_matrix(fake_model):
    model, constructor_calls = fake_model

    result = embeddings.embed_documents([])

    assert result.shape == (0, 0)
    assert result.dtype == np.float32
    assert model.calls == []
    assert constructor_calls == []
