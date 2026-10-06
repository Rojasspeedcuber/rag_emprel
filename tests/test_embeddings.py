import numpy as np

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
