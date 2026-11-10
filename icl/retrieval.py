"""Embedding-based retrieval of candidate demonstrations (stage 1 of selection).

Default embedder: sentence-transformers ``all-MiniLM-L6-v2``. ``hash`` is a
dependency-free bag-of-words fallback (used by the offline tests).
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import numpy as np


def example_text(ex: dict, fields: list[str]) -> str:
    return " ".join(str(ex[f]) for f in fields)


class HashEmbedder:
    """Hashed bag-of-words + bigrams, L2-normalised. No downloads needed."""

    name = "hash"

    def __init__(self, dim: int = 2048):
        self.dim = dim

    def encode(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            toks = re.findall(r"\w+", t.lower())
            for g in toks + [a + "_" + b for a, b in zip(toks, toks[1:])]:
                h = int(hashlib.md5(g.encode()).hexdigest()[:8], 16)
                out[i, h % self.dim] += 1.0
        return _l2(out)


class STEmbedder:
    def __init__(self, name: str = "sentence-transformers/all-MiniLM-L6-v2", device: str | None = None):
        from sentence_transformers import SentenceTransformer

        self.name = name
        self.model = SentenceTransformer(name, device=device)

    def encode(self, texts: list[str]) -> np.ndarray:
        emb = self.model.encode(texts, batch_size=256, show_progress_bar=len(texts) > 2000,
                                convert_to_numpy=True, normalize_embeddings=True)
        return emb.astype(np.float32)


def load_embedder(name: str, device: str | None = None):
    return HashEmbedder() if name == "hash" else STEmbedder(name, device)


def _l2(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-8)


class Retriever:
    """Cosine top-M retrieval from a task's demonstration pool, with an on-disk
    cache of pool embeddings (keyed by task, pool content and embedder)."""

    def __init__(self, embedder, pool: list[dict], fields: list[str], cache_dir: str | Path | None = None):
        self.embedder = embedder
        self.pool = pool
        self.fields = fields
        texts = [example_text(e, fields) for e in pool]
        key = hashlib.md5(("\n".join(texts) + embedder.name).encode()).hexdigest()[:16]
        path = Path(cache_dir) / f"emb_{key}.npy" if cache_dir else None
        if path is not None and path.exists():
            self.emb = np.load(path)
        else:
            self.emb = embedder.encode(texts)
            if path is not None:
                path.parent.mkdir(parents=True, exist_ok=True)
                np.save(path, self.emb)

    def top_m(self, queries: list[dict], m: int) -> tuple[np.ndarray, np.ndarray]:
        """Return (indices [N, m], similarities [N, m]), most similar first."""
        q = self.embedder.encode([example_text(x, self.fields) for x in queries])
        sims = q @ self.emb.T
        m = min(m, sims.shape[1])
        idx = np.argpartition(-sims, m - 1, axis=1)[:, :m]
        part = np.take_along_axis(sims, idx, 1)
        order = np.argsort(-part, axis=1)
        return np.take_along_axis(idx, order, 1), np.take_along_axis(part, order, 1)
