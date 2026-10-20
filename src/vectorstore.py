import atexit
import uuid
from functools import lru_cache
from pathlib import Path

import numpy as np
from qdrant_client import QdrantClient, models

from src.config import QDRANT_API_KEY, QDRANT_COLLECTION, QDRANT_URL

_SCROLL_BATCH = 500


@lru_cache(maxsize=4)
def _get_client(local_path: str) -> QdrantClient:
    if QDRANT_URL:
        client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    else:
        try:
            client = QdrantClient(path=local_path)
        except RuntimeError as exc:
            if "already accessed" not in str(exc):
                raise
            raise RuntimeError(
                "O Qdrant embutido já está em uso por outro processo. "
                "Feche a outra instância do app ou defina QDRANT_URL para usar um servidor Qdrant."
            ) from exc
    atexit.register(client.close)
    return client


class VectorStore:
    """Coleção de vetores normalizados com similaridade de cosseno e metadados (source, page, text)."""

    def __init__(self, local_dir: Path, collection: str = QDRANT_COLLECTION):
        self.collection = collection
        self.client = _get_client(str(local_dir / "qdrant"))

    def _exists(self) -> bool:
        return self.client.collection_exists(self.collection)

    def is_populated(self) -> bool:
        return self._exists() and self.client.count(self.collection, exact=True).count > 0

    def existing_ids(self, ids: list[str]) -> set[str]:
        if not ids or not self._exists():
            return set()
        records = self.client.retrieve(self.collection, ids=ids, with_payload=False, with_vectors=False)
        return {str(record.id) for record in records}

    def stale_ids(self, keep: set[str]) -> list[str]:
        if not self._exists():
            return []
        stale, offset = [], None
        while True:
            points, offset = self.client.scroll(
                self.collection,
                limit=_SCROLL_BATCH,
                offset=offset,
                with_payload=False,
                with_vectors=False,
            )
            stale.extend(str(point.id) for point in points if str(point.id) not in keep)
            if offset is None:
                return stale

    def upsert(self, ids: list[str], vectors: np.ndarray, payloads: list[dict]) -> None:
        self._ensure_collection(vectors.shape[1])
        points = [
            models.PointStruct(id=point_id, vector=vector.tolist(), payload=payload)
            for point_id, vector, payload in zip(ids, vectors, payloads)
        ]
        self.client.upsert(self.collection, points=points)

    def delete(self, ids: list[str]) -> None:
        if ids:
            self.client.delete(self.collection, points_selector=models.PointIdsList(points=ids))

    def search(self, vector: list[float], limit: int) -> list[tuple[float, dict]]:
        result = self.client.query_points(
            self.collection,
            query=vector,
            limit=limit,
            with_payload=True,
        )
        return [(point.score, point.payload) for point in result.points]

    def _ensure_collection(self, dim: int) -> None:
        if self._exists():
            current = self.client.get_collection(self.collection).config.params.vectors.size
            if current == dim:
                return
            self.client.delete_collection(self.collection)
        self.client.create_collection(
            self.collection,
            vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE),
        )
