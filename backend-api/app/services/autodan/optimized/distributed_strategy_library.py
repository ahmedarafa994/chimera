"""
Distributed Strategy Library with FAISS Integration.

Implements efficient strategy storage and retrieval using:
- FAISS for fast similarity search
- Hierarchical clustering for organization
- Distributed storage support
- Incremental index updates
"""

import hashlib
import json
import logging
import os
import pickle
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class IndexType(Enum):
    """FAISS index types."""

    FLAT = "flat"  # Exact search
    IVF = "ivf"  # Inverted file index
    HNSW = "hnsw"  # Hierarchical navigable small world
    PQ = "pq"  # Product quantization


@dataclass
class Strategy:
    """Represents a jailbreak strategy."""

    id: str
    content: str
    embedding: np.ndarray
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    usage_count: int = 0
    success_rate: float = 0.0


@dataclass
class SearchResult:
    """Search result from strategy library."""

    strategy: Strategy
    similarity: float
    rank: int


class FAISSIndex:
    """
    FAISS-based vector index for strategy embeddings.

    Supports multiple index types for different performance/accuracy tradeoffs.
    """

    def __init__(
        self,
        dimension: int = 384,
        index_type: IndexType = IndexType.HNSW,
        nlist: int = 100,  # For IVF
        m: int = 32,  # For HNSW
        ef_construction: int = 200,  # For HNSW
        ef_search: int = 50,  # For HNSW
    ):
        self.dimension = dimension
        self.index_type = index_type
        self.nlist = nlist
        self.m = m
        self.ef_construction = ef_construction
        self.ef_search = ef_search

        # Index will be created lazily
        self.index = None
        self.is_trained = False

        # ID mapping (FAISS uses sequential IDs)
        self.id_to_idx: dict[str, int] = {}
        self.idx_to_id: dict[int, str] = {}
        self.next_idx = 0

        # Thread safety
        self.lock = threading.RLock()

    def _create_index(self):
        """Create FAISS index based on type."""
        try:
            import faiss

            if self.index_type == IndexType.FLAT:
                self.index = faiss.IndexFlatIP(self.dimension)
                self.is_trained = True

            elif self.index_type == IndexType.IVF:
                quantizer = faiss.IndexFlatIP(self.dimension)
                self.index = faiss.IndexIVFFlat(quantizer, self.dimension, self.nlist)

            elif self.index_type == IndexType.HNSW:
                self.index = faiss.IndexHNSWFlat(self.dimension, self.m)
                self.index.hnsw.efConstruction = self.ef_construction
                self.index.hnsw.efSearch = self.ef_search
                self.is_trained = True

            elif self.index_type == IndexType.PQ:
                self.index = faiss.IndexPQ(
                    self.dimension,
                    8,
                    8,  # m, nbits
                )

            logger.info(f"Created FAISS index: {self.index_type.value}")

        except ImportError:
            logger.warning("FAISS not available, using numpy fallback")
            self.index = None

    def _ensure_index(self):
        """Ensure index is created."""
        if self.index is None:
            self._create_index()

    def train(self, embeddings: np.ndarray):
        """
        Train index on embeddings (required for some index types).

        Args:
            embeddings: Training embeddings (n_samples, dimension)
        """
        self._ensure_index()

        if self.index is None:
            return

        with self.lock:
            if not self.is_trained:
                # Normalize for inner product
                faiss = self._get_faiss()
                if faiss:
                    faiss.normalize_L2(embeddings)
                    self.index.train(embeddings)
                    self.is_trained = True
                    logger.info(f"Trained index on {len(embeddings)} vectors")

    def add(self, id: str, embedding: np.ndarray):
        """
        Add embedding to index.

        Args:
            id: Strategy ID
            embedding: Embedding vector
        """
        self._ensure_index()

        with self.lock:
            # Normalize
            embedding = embedding.astype(np.float32).reshape(1, -1)
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm

            if self.index is not None:
                # Add to FAISS
                self.index.add(embedding)

            # Update mappings
            self.id_to_idx[id] = self.next_idx
            self.idx_to_id[self.next_idx] = id
            self.next_idx += 1

    def search(
        self,
        query: np.ndarray,
        k: int = 10,
    ) -> list[tuple[str, float]]:
        """
        Search for similar embeddings.

        Args:
            query: Query embedding
            k: Number of results

        Returns:
            List of (id, similarity) tuples
        """
        self._ensure_index()

        with self.lock:
            # Normalize query
            query = query.astype(np.float32).reshape(1, -1)
            norm = np.linalg.norm(query)
            if norm > 0:
                query = query / norm

            if self.index is not None and self.index.ntotal > 0:
                # FAISS search
                k = min(k, self.index.ntotal)
                distances, indices = self.index.search(query, k)

                results = []
                for dist, idx in zip(distances[0], indices[0], strict=False):
                    if idx >= 0 and idx in self.idx_to_id:
                        results.append((self.idx_to_id[idx], float(dist)))
                return results

            return []

    def remove(self, id: str):
        """Remove embedding from index (marks as deleted)."""
        with self.lock:
            if id in self.id_to_idx:
                # FAISS doesn't support true deletion
                # Mark as deleted in mapping
                idx = self.id_to_idx.pop(id)
                self.idx_to_id.pop(idx, None)

    def save(self, path: str):
        """Save index to file."""
        self._ensure_index()

        with self.lock:
            data = {
                "id_to_idx": self.id_to_idx,
                "idx_to_id": self.idx_to_id,
                "next_idx": self.next_idx,
                "dimension": self.dimension,
                "index_type": self.index_type.value,
            }

            # Save mappings
            with open(f"{path}.meta", "wb") as f:
                pickle.dump(data, f)

            # Save FAISS index
            if self.index is not None:
                faiss = self._get_faiss()
                if faiss:
                    faiss.write_index(self.index, f"{path}.faiss")

    def load(self, path: str):
        """Load index from file."""
        with self.lock:
            # Load mappings
            with open(f"{path}.meta", "rb") as f:
                data = pickle.load(f)

            self.id_to_idx = data["id_to_idx"]
            self.idx_to_id = {int(k): v for k, v in data["idx_to_id"].items()}
            self.next_idx = data["next_idx"]
            self.dimension = data["dimension"]
            self.index_type = IndexType(data["index_type"])

            # Load FAISS index
            faiss = self._get_faiss()
            if faiss and os.path.exists(f"{path}.faiss"):
                self.index = faiss.read_index(f"{path}.faiss")
                self.is_trained = True

    def _get_faiss(self):
        """Get FAISS module if available."""
        try:
            import faiss

            return faiss
        except ImportError:
            return None

    def __len__(self) -> int:
        return len(self.id_to_idx)


class HierarchicalCluster:
    """
    Hierarchical clustering for strategy organization.

    Groups similar strategies for efficient browsing and retrieval.
    """

    def __init__(
        self,
        n_clusters: int = 10,
        min_cluster_size: int = 5,
    ):
        self.n_clusters = n_clusters
        self.min_cluster_size = min_cluster_size

        # Cluster assignments
        self.clusters: dict[int, list[str]] = {}
        self.strategy_to_cluster: dict[str, int] = {}

        # Cluster centroids
        self.centroids: dict[int, np.ndarray] = {}

    def fit(self, strategies: list[Strategy]):
        """
        Fit clustering on strategies.

        Args:
            strategies: List of strategies with embeddings
        """
        if len(strategies) < self.n_clusters:
            # Not enough for clustering
            self.clusters[0] = [s.id for s in strategies]
            for s in strategies:
                self.strategy_to_cluster[s.id] = 0
            return

        # Extract embeddings
        embeddings = np.array([s.embedding for s in strategies])

        # K-means clustering
        centroids, assignments = self._kmeans(embeddings, self.n_clusters)

        # Build clusters
        self.clusters = {i: [] for i in range(self.n_clusters)}
        for i, strategy in enumerate(strategies):
            cluster_id = int(assignments[i])
            self.clusters[cluster_id].append(strategy.id)
            self.strategy_to_cluster[strategy.id] = cluster_id

        # Store centroids
        for i, centroid in enumerate(centroids):
            self.centroids[i] = centroid

    def _kmeans(
        self,
        data: np.ndarray,
        k: int,
        max_iters: int = 100,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Simple k-means implementation."""
        n_samples = len(data)

        # Initialize centroids randomly
        indices = np.random.choice(n_samples, k, replace=False)
        centroids = data[indices].copy()

        for _ in range(max_iters):
            # Assign to nearest centroid
            distances = np.zeros((n_samples, k))
            for i, centroid in enumerate(centroids):
                distances[:, i] = np.linalg.norm(data - centroid, axis=1)
            assignments = np.argmin(distances, axis=1)

            # Update centroids
            new_centroids = np.zeros_like(centroids)
            for i in range(k):
                mask = assignments == i
                if mask.sum() > 0:
                    new_centroids[i] = data[mask].mean(axis=0)
                else:
                    new_centroids[i] = centroids[i]

            # Check convergence
            if np.allclose(centroids, new_centroids):
                break
            centroids = new_centroids

        return centroids, assignments

    def get_cluster(self, cluster_id: int) -> list[str]:
        """Get strategy IDs in cluster."""
        return self.clusters.get(cluster_id, [])

    def get_nearest_cluster(self, embedding: np.ndarray) -> int:
        """Find nearest cluster for embedding."""
        if not self.centroids:
            return 0

        min_dist = float("inf")
        nearest = 0

        for cluster_id, centroid in self.centroids.items():
            dist = np.linalg.norm(embedding - centroid)
            if dist < min_dist:
                min_dist = dist
                nearest = cluster_id

        return nearest


class DistributedStrategyLibrary:
    """
    Distributed strategy library with FAISS integration.

    Provides efficient storage, retrieval, and management of
    jailbreak strategies across distributed systems.
    """

    def __init__(
        self,
        dimension: int = 384,
        index_type: IndexType = IndexType.HNSW,
        storage_path: str | None = None,
        enable_clustering: bool = True,
        n_clusters: int = 10,
    ):
        self.dimension = dimension
        self.storage_path = storage_path
        self.enable_clustering = enable_clustering

        # FAISS index
        self.index = FAISSIndex(
            dimension=dimension,
            index_type=index_type,
        )

        # Strategy storage
        self.strategies: dict[str, Strategy] = {}

        # Clustering
        self.clustering = HierarchicalCluster(n_clusters=n_clusters) if enable_clustering else None

        # Statistics
        self.stats = {
            "total_strategies": 0,
            "total_searches": 0,
            "avg_search_time": 0.0,
        }

        # Thread safety
        self.lock = threading.RLock()

        # Load existing data
        if storage_path and os.path.exists(f"{storage_path}.meta"):
            self.load()

    def add_strategy(
        self,
        content: str,
        embedding: np.ndarray,
        score: float = 0.0,
        metadata: dict[str, Any] | None = None,
    ) -> Strategy:
        """
        Add a new strategy to the library.

        Args:
            content: Strategy content/prompt
            embedding: Strategy embedding
            score: Initial score
            metadata: Additional metadata

        Returns:
            Created strategy
        """
        # Generate ID
        strategy_id = self._generate_id(content)

        with self.lock:
            # Check for duplicate
            if strategy_id in self.strategies:
                # Update existing
                existing = self.strategies[strategy_id]
                existing.usage_count += 1
                return existing

            # Create strategy
            strategy = Strategy(
                id=strategy_id,
                content=content,
                embedding=embedding,
                score=score,
                metadata=metadata or {},
            )

            # Store
            self.strategies[strategy_id] = strategy
            self.index.add(strategy_id, embedding)

            self.stats["total_strategies"] += 1

            return strategy

    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 10,
        min_score: float = 0.0,
        filter_metadata: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """
        Search for similar strategies.

        Args:
            query_embedding: Query embedding
            k: Number of results
            min_score: Minimum strategy score
            filter_metadata: Metadata filters

        Returns:
            List of search results
        """
        start_time = time.time()

        with self.lock:
            # FAISS search
            results = self.index.search(query_embedding, k * 2)

            # Filter and rank
            search_results = []
            for strategy_id, similarity in results:
                if strategy_id not in self.strategies:
                    continue

                strategy = self.strategies[strategy_id]

                # Apply filters
                if strategy.score < min_score:
                    continue

                if filter_metadata:
                    match = all(strategy.metadata.get(k) == v for k, v in filter_metadata.items())
                    if not match:
                        continue

                search_results.append(
                    SearchResult(
                        strategy=strategy,
                        similarity=similarity,
                        rank=len(search_results),
                    )
                )

                if len(search_results) >= k:
                    break

            # Update stats
            search_time = time.time() - start_time
            self.stats["total_searches"] += 1
            self.stats["avg_search_time"] = self.stats["avg_search_time"] * 0.9 + search_time * 0.1

            return search_results

    def get_strategy(self, strategy_id: str) -> Strategy | None:
        """Get strategy by ID."""
        return self.strategies.get(strategy_id)

    def update_strategy(
        self,
        strategy_id: str,
        score: float | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        """Update strategy score or metadata."""
        with self.lock:
            if strategy_id in self.strategies:
                strategy = self.strategies[strategy_id]

                if score is not None:
                    strategy.score = score

                if metadata is not None:
                    strategy.metadata.update(metadata)

    def record_usage(
        self,
        strategy_id: str,
        success: bool,
    ):
        """Record strategy usage and update success rate."""
        with self.lock:
            if strategy_id in self.strategies:
                strategy = self.strategies[strategy_id]
                strategy.usage_count += 1

                # Update success rate with exponential moving average
                alpha = 0.1
                strategy.success_rate = (
                    strategy.success_rate * (1 - alpha) + (1.0 if success else 0.0) * alpha
                )

    def remove_strategy(self, strategy_id: str):
        """Remove strategy from library."""
        with self.lock:
            if strategy_id in self.strategies:
                del self.strategies[strategy_id]
                self.index.remove(strategy_id)
                self.stats["total_strategies"] -= 1

    def get_top_strategies(
        self,
        n: int = 10,
        by: str = "score",
    ) -> list[Strategy]:
        """
        Get top strategies by metric.

        Args:
            n: Number of strategies
            by: Metric to sort by (score, success_rate, usage_count)

        Returns:
            List of top strategies
        """
        with self.lock:
            strategies = list(self.strategies.values())

            if by == "score":
                strategies.sort(key=lambda s: s.score, reverse=True)
            elif by == "success_rate":
                strategies.sort(key=lambda s: s.success_rate, reverse=True)
            elif by == "usage_count":
                strategies.sort(key=lambda s: s.usage_count, reverse=True)

            return strategies[:n]

    def get_cluster_strategies(
        self,
        query_embedding: np.ndarray,
    ) -> list[Strategy]:
        """Get strategies from nearest cluster."""
        if self.clustering is None:
            return []

        with self.lock:
            cluster_id = self.clustering.get_nearest_cluster(query_embedding)
            strategy_ids = self.clustering.get_cluster(cluster_id)

            return [self.strategies[sid] for sid in strategy_ids if sid in self.strategies]

    def rebuild_clusters(self):
        """Rebuild hierarchical clusters."""
        if self.clustering is None:
            return

        with self.lock:
            strategies = list(self.strategies.values())
            if strategies:
                self.clustering.fit(strategies)

    def save(self):
        """Save library to storage."""
        if self.storage_path is None:
            return

        with self.lock:
            # Save index
            self.index.save(self.storage_path)

            # Save strategies
            strategies_data = {}
            for sid, strategy in self.strategies.items():
                strategies_data[sid] = {
                    "id": strategy.id,
                    "content": strategy.content,
                    "embedding": strategy.embedding.tolist(),
                    "score": strategy.score,
                    "metadata": strategy.metadata,
                    "created_at": strategy.created_at,
                    "usage_count": strategy.usage_count,
                    "success_rate": strategy.success_rate,
                }

            with open(f"{self.storage_path}.strategies", "w") as f:
                json.dump(strategies_data, f)

            logger.info(f"Saved {len(self.strategies)} strategies to {self.storage_path}")

    def load(self):
        """Load library from storage."""
        if self.storage_path is None:
            return

        with self.lock:
            # Load index
            try:
                self.index.load(self.storage_path)
            except FileNotFoundError:
                logger.warning("Index file not found, starting fresh")

            # Load strategies
            strategies_path = f"{self.storage_path}.strategies"
            if os.path.exists(strategies_path):
                with open(strategies_path) as f:
                    strategies_data = json.load(f)

                for sid, data in strategies_data.items():
                    self.strategies[sid] = Strategy(
                        id=data["id"],
                        content=data["content"],
                        embedding=np.array(data["embedding"]),
                        score=data["score"],
                        metadata=data["metadata"],
                        created_at=data["created_at"],
                        usage_count=data["usage_count"],
                        success_rate=data["success_rate"],
                    )

                self.stats["total_strategies"] = len(self.strategies)
                logger.info(f"Loaded {len(self.strategies)} strategies")

            # Rebuild clusters
            if self.enable_clustering:
                self.rebuild_clusters()

    def get_stats(self) -> dict[str, Any]:
        """Get library statistics."""
        with self.lock:
            return {
                **self.stats,
                "index_size": len(self.index),
                "n_clusters": (len(self.clustering.clusters) if self.clustering else 0),
            }

    def _generate_id(self, content: str) -> str:
        """Generate unique ID for content."""
        return hashlib.sha256(content.encode()).hexdigest()[:16]
