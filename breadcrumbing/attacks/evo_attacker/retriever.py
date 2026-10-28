"""Deterministic MiniLM normalized-cosine retrieval."""

from __future__ import annotations

import threading
from typing import Any

import numpy as np

from .memory_io import FrozenMemory
from .schema import RetrievedMemory


_MODEL_LOCK = threading.Lock()
_MODELS: dict[str, Any] = {}


def _encoder(model_name: str) -> Any:
    with _MODEL_LOCK:
        if model_name not in _MODELS:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise RuntimeError(
                    "sentence-transformers is required for frozen Evo memory"
                ) from exc
            _MODELS[model_name] = SentenceTransformer(model_name)
        return _MODELS[model_name]


class EmptyMemoryRetriever:
    """Zero-shot exploration mode used for all train rollouts."""

    memory_id = "empty"
    memory_sha256 = ""

    def retrieve(self, query: str, *, k: int = 5) -> list[RetrievedMemory]:
        del query, k
        return []


class FrozenMemoryRetriever:
    """Read-only all-MiniLM-L6-v2 normalized cosine top-k."""

    def __init__(self, memory: FrozenMemory):
        self.memory = memory
        self.memory_id = memory.manifest.memory_id
        self.memory_sha256 = memory.manifest.memory_sha256

    @classmethod
    def load(
        cls, path: str, *, verify_checksum: bool = True
    ) -> "FrozenMemoryRetriever":
        return cls(FrozenMemory.load(path, verify_checksum=verify_checksum))

    def retrieve(self, query: str, *, k: int = 5) -> list[RetrievedMemory]:
        if k <= 0 or not self.memory.entries:
            return []
        vector = _encoder(self.memory.manifest.embedding_model).encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0]
        scores = self.memory.embeddings @ np.asarray(vector, dtype=np.float32)
        count = min(int(k), len(self.memory.entries))
        indices = np.argsort(-scores, kind="stable")[:count]
        return [
            RetrievedMemory(
                entry_id=self.memory.entries[int(index)].entry_id,
                score=float(scores[int(index)]),
                retrieval_view=self.memory.entries[int(index)].retrieval_view,
            )
            for index in indices
        ]
