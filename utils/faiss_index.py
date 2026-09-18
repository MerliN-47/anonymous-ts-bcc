"""
[Anonymous] TS-RAG / CAVIR FAISS Index Builder & Memory Partition Manager
Double-Blind Compliant Modular Architecture
"""

import os
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Union

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False


class TimeSeriesFAISSIndex:
    """
    Unified index wrapper supporting Flat (L2 / Inner Product), IVF-PQ, and HNSW backends.
    Includes a robust vectorized NumPy fallback for environments without FAISS binary installed.
    """
    def __init__(
        self,
        dimension: int = 768,
        index_type: str = "flat",
        metric: str = "cosine",
        use_gpu: bool = False
    ):
        self.dimension = dimension
        self.index_type = index_type.lower()
        self.metric = metric.lower()
        self.use_gpu = use_gpu and torch.cuda.is_available() and FAISS_AVAILABLE
        self.is_trained = False
        self.ntotal = 0
        self.index = None
        self.vectors = None  # Storage for NumPy fallback or metadata retrieval

        self._initialize_index()

    def _initialize_index(self):
        if not FAISS_AVAILABLE:
            self.vectors = np.empty((0, self.dimension), dtype=np.float32)
            self.is_trained = True
            return

        faiss_metric = faiss.METRIC_INNER_PRODUCT if self.metric == "cosine" else faiss.METRIC_L2

        if self.index_type == "flat":
            self.index = faiss.IndexFlatIP(self.dimension) if self.metric == "cosine" else faiss.IndexFlatL2(self.dimension)
            self.is_trained = True
        elif self.index_type == "ivf_pq":
            nlist = 100
            m = 8  # number of subquantizers
            nbits = 8
            quantizer = faiss.IndexFlatIP(self.dimension) if self.metric == "cosine" else faiss.IndexFlatL2(self.dimension)
            self.index = faiss.IndexIVFPQ(quantizer, self.dimension, nlist, m, nbits, faiss_metric)
            self.is_trained = False
        elif self.index_type == "hnsw":
            M = 32  # Number of neighbors per node
            self.index = faiss.IndexHNSWFlat(self.dimension, M, faiss_metric)
            self.is_trained = True
        else:
            self.index = faiss.IndexFlatIP(self.dimension) if self.metric == "cosine" else faiss.IndexFlatL2(self.dimension)
            self.is_trained = True

        if self.use_gpu:
            try:
                res = faiss.StandardGpuResources()
                self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
            except Exception:
                pass  # Fall back to CPU gracefully

    def _normalize(self, x: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8
        return (x / norms).astype(np.float32)

    def train(self, vectors: np.ndarray):
        """Train quantization codebooks if required by index type."""
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        if self.metric == "cosine":
            vectors = self._normalize(vectors)

        if FAISS_AVAILABLE and self.index is not None:
            if not self.index.is_trained:
                self.index.train(vectors)
            self.is_trained = True
        else:
            self.is_trained = True

    def add(self, vectors: np.ndarray):
        """Add embedding vectors to index."""
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        if self.metric == "cosine":
            vectors = self._normalize(vectors)

        if FAISS_AVAILABLE and self.index is not None:
            if not self.index.is_trained:
                self.train(vectors)
            self.index.add(vectors)
            self.ntotal = self.index.ntotal
        else:
            if self.vectors is None or len(self.vectors) == 0:
                self.vectors = vectors
            else:
                self.vectors = np.vstack([self.vectors, vectors])
            self.ntotal = len(self.vectors)

    def search(self, queries: np.ndarray, top_k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for nearest neighbors.
        Returns:
            distances: (N, top_k) array of similarities or distances
            indices: (N, top_k) array of integer indices
        """
        queries = np.ascontiguousarray(queries, dtype=np.float32)
        if self.metric == "cosine":
            queries = self._normalize(queries)

        if FAISS_AVAILABLE and self.index is not None:
            distances, indices = self.index.search(queries, top_k)
            return distances, indices
        else:
            # Vectorized NumPy fallback
            if self.ntotal == 0 or self.vectors is None:
                return np.zeros((len(queries), top_k), dtype=np.float32), np.full((len(queries), top_k), -1, dtype=np.int64)

            top_k = min(top_k, self.ntotal)
            if self.metric == "cosine":
                # Higher inner product = more similar
                sims = np.dot(queries, self.vectors.T)
                indices = np.argsort(-sims, axis=1)[:, :top_k]
                distances = np.take_along_axis(sims, indices, axis=1)
            else:
                # Lower L2 distance = closer
                dists = np.linalg.norm(queries[:, None, :] - self.vectors[None, :, :], axis=-1)
                indices = np.argsort(dists, axis=1)[:, :top_k]
                distances = np.take_along_axis(dists, indices, axis=1)

            return distances.astype(np.float32), indices.astype(np.int64)

    def save(self, filepath: str):
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        if FAISS_AVAILABLE and self.index is not None:
            cpu_index = faiss.index_gpu_to_cpu(self.index) if self.use_gpu else self.index
            faiss.write_index(cpu_index, filepath)
        else:
            np.save(filepath + ".npy", self.vectors)

    def load(self, filepath: str):
        if FAISS_AVAILABLE and os.path.exists(filepath):
            self.index = faiss.read_index(filepath)
            self.ntotal = self.index.ntotal
            self.is_trained = self.index.is_trained
            if self.use_gpu:
                res = faiss.StandardGpuResources()
                self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
        elif os.path.exists(filepath + ".npy"):
            self.vectors = np.load(filepath + ".npy")
            self.ntotal = len(self.vectors)
            self.is_trained = True


def partition_memory(
    data: np.ndarray,
    regime: str = "strict_zeroshot",
    train_ratio: float = 0.7,
    kb_fraction: float = 1.0,
    poison_ratio: float = 0.0,
    seed: int = 2021
) -> Dict[str, np.ndarray]:
    """
    Partitions raw series or memory embeddings according to benchmark protocol.
    Guarantees strict temporal causality without future leakage.
    """
    rng = np.random.RandomState(seed)
    total_len = len(data)
    train_end = int(total_len * train_ratio)

    if regime == "strict_zeroshot":
        # Memory strictly drawn from historical training split only
        memory_pool = data[:train_end]
        test_pool = data[train_end:]
    elif regime == "cross_domain":
        # Cross-domain scenario
        memory_pool = data[:train_end]
        test_pool = data[train_end:]
    elif regime == "retrieval_shot":
        memory_pool = data[:train_end]
        test_pool = data[train_end:]
    else:
        memory_pool = data[:train_end]
        test_pool = data[train_end:]

    # Subsample memory if kb_fraction < 1.0
    if kb_fraction < 1.0:
        sub_n = max(10, int(len(memory_pool) * kb_fraction))
        sub_indices = rng.choice(len(memory_pool), size=sub_n, replace=False)
        memory_pool = memory_pool[sub_indices]

    # Inject adversarial / poison noise if requested
    if poison_ratio > 0.0:
        num_poison = int(len(memory_pool) * poison_ratio)
        poison_idx = rng.choice(len(memory_pool), size=num_poison, replace=False)
        noise = rng.normal(0, 2.0, size=memory_pool[poison_idx].shape)
        memory_pool[poison_idx] = memory_pool[poison_idx] + noise

    return {
        "memory": memory_pool,
        "test": test_pool
    }
