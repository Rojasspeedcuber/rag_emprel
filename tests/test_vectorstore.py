import numpy as np
import pytest

from src import vectorstore
from src.vectorstore import VectorStore


@pytest.fixture
def store(tmp_path):
    vectorstore._get_client.cache_clear()
    instance = VectorStore(tmp_path, collection="test_chunks")
    yield instance
    instance.client.close()
    vectorstore._get_client.cache_clear()


def test_local_lock_error_suggests_server_mode(monkeypatch, tmp_path):
    vectorstore._get_client.cache_clear()
    monkeypatch.setattr(vectorstore, "QDRANT_URL", None)

    def locked(path):
        raise RuntimeError("Storage folder is already accessed by another instance of Qdrant client.")

    monkeypatch.setattr(vectorstore, "QdrantClient", locked)
    try:
        with pytest.raises(RuntimeError, match="QDRANT_URL"):
            vectorstore._get_client(str(tmp_path))
    finally:
        vectorstore._get_client.cache_clear()


def test_upsert_and_search_returns_payload_with_score(store):
    vectors = np.array([[1.0, 0.0], [0.0, 1.0]], dtype="float32")
    store.upsert(
        ["00000000-0000-0000-0000-000000000001", "00000000-0000-0000-0000-000000000002"],
        vectors,
        [
            {"text": "a", "source": "a.pdf", "page": 1},
            {"text": "b", "source": "b.pdf", "page": 2},
        ],
    )

    hits = store.search([1.0, 0.0], limit=1)

    assert len(hits) == 1
    score, payload = hits[0]
    assert score == pytest.approx(1.0)
    assert payload == {"text": "a", "source": "a.pdf", "page": 1}


def test_existing_ids_only_returns_stored_points(store):
    store.upsert(
        ["00000000-0000-0000-0000-000000000001"],
        np.array([[1.0, 0.0]], dtype="float32"),
        [{"text": "a", "source": "a.pdf", "page": 1}],
    )

    found = store.existing_ids(
        ["00000000-0000-0000-0000-000000000001", "00000000-0000-0000-0000-000000000009"]
    )

    assert found == {"00000000-0000-0000-0000-000000000001"}


def test_stale_ids_lists_points_outside_keep_set(store):
    ids = [
        "00000000-0000-0000-0000-000000000001",
        "00000000-0000-0000-0000-000000000002",
    ]
    store.upsert(ids, np.eye(2, dtype="float32"), [{"text": "a", "source": "a", "page": 1}] * 2)

    assert store.stale_ids({ids[0]}) == [ids[1]]


def test_stale_ids_is_empty_without_collection(store):
    assert store.stale_ids(set()) == []
    assert store.is_populated() is False


def test_delete_removes_points(store):
    ids = ["00000000-0000-0000-0000-000000000001"]
    store.upsert(ids, np.array([[1.0, 0.0]], dtype="float32"), [{"text": "a", "source": "a", "page": 1}])

    store.delete(ids)

    assert store.is_populated() is False
