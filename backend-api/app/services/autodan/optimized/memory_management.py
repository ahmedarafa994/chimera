"""
Memory Management Optimizations for AutoDAN.

Implements efficient memory handling with:
- Adaptive memory pooling
- Gradient checkpointing
- Embedding compression
- Memory-efficient data structures
"""

import gc
import logging
import threading
import weakref
from dataclasses import dataclass
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class MemoryStats:
    """Memory usage statistics."""

    allocated_bytes: int
    peak_bytes: int
    pool_size: int
    active_allocations: int
    cache_hits: int
    cache_misses: int


class AdaptiveMemoryPool:
    """
    Adaptive memory pool for efficient allocation.

    Pre-allocates memory blocks and reuses them to reduce
    allocation overhead and memory fragmentation.
    """

    def __init__(
        self,
        initial_size: int = 1024 * 1024,  # 1MB
        max_size: int = 1024 * 1024 * 1024,  # 1GB
        growth_factor: float = 1.5,
        shrink_threshold: float = 0.3,
    ):
        self.initial_size = initial_size
        self.max_size = max_size
        self.growth_factor = growth_factor
        self.shrink_threshold = shrink_threshold

        # Memory pools by size class
        self.pools: dict[int, list[np.ndarray]] = {}
        self.size_classes = [64, 256, 1024, 4096, 16384, 65536, 262144, 1048576]

        # Statistics
        self.stats = MemoryStats(
            allocated_bytes=0,
            peak_bytes=0,
            pool_size=0,
            active_allocations=0,
            cache_hits=0,
            cache_misses=0,
        )

        # Thread safety
        self.lock = threading.Lock()

        # Initialize pools
        self._initialize_pools()

    def _initialize_pools(self):
        """Initialize memory pools for each size class."""
        for size in self.size_classes:
            self.pools[size] = []

    def _get_size_class(self, size: int) -> int:
        """Get appropriate size class for requested size."""
        for sc in self.size_classes:
            if size <= sc:
                return sc
        return size  # Larger than any class

    def allocate(self, shape: tuple[int, ...], dtype: np.dtype = np.float32) -> np.ndarray:
        """
        Allocate array from pool.

        Args:
            shape: Array shape
            dtype: Data type

        Returns:
            Allocated numpy array
        """
        size = int(np.prod(shape) * np.dtype(dtype).itemsize)
        size_class = self._get_size_class(size)

        with self.lock:
            # Try to get from pool
            if self.pools.get(size_class):
                arr = self.pools[size_class].pop()
                self.stats.cache_hits += 1

                # Reshape if needed
                if arr.size >= np.prod(shape):
                    return arr.ravel()[: np.prod(shape)].reshape(shape)

            # Allocate new
            self.stats.cache_misses += 1
            arr = np.zeros(shape, dtype=dtype)

            self.stats.allocated_bytes += arr.nbytes
            self.stats.peak_bytes = max(
                self.stats.peak_bytes,
                self.stats.allocated_bytes,
            )
            self.stats.active_allocations += 1

            return arr

    def release(self, arr: np.ndarray):
        """
        Release array back to pool.

        Args:
            arr: Array to release
        """
        size = arr.nbytes
        size_class = self._get_size_class(size)

        with self.lock:
            # Add to pool if not too large
            if size_class in self.pools:
                if len(self.pools[size_class]) < 100:  # Limit pool size
                    self.pools[size_class].append(arr.ravel())
                    self.stats.pool_size += size

            self.stats.active_allocations -= 1

    def shrink(self):
        """Shrink pools if usage is low."""
        with self.lock:
            for size_class in self.pools:
                pool = self.pools[size_class]
                if len(pool) > 10:
                    # Keep only half
                    to_remove = len(pool) // 2
                    for _ in range(to_remove):
                        arr = pool.pop()
                        self.stats.pool_size -= arr.nbytes

    def clear(self):
        """Clear all pools."""
        with self.lock:
            for size_class in self.pools:
                self.pools[size_class].clear()
            self.stats.pool_size = 0
            gc.collect()

    def get_stats(self) -> dict[str, Any]:
        """Get memory pool statistics."""
        return {
            "allocated_bytes": self.stats.allocated_bytes,
            "peak_bytes": self.stats.peak_bytes,
            "pool_size": self.stats.pool_size,
            "active_allocations": self.stats.active_allocations,
            "cache_hits": self.stats.cache_hits,
            "cache_misses": self.stats.cache_misses,
            "hit_rate": (
                self.stats.cache_hits / (self.stats.cache_hits + self.stats.cache_misses)
                if (self.stats.cache_hits + self.stats.cache_misses) > 0
                else 0.0
            ),
        }


class GradientCheckpointing:
    """
    Gradient checkpointing for memory-efficient backpropagation.

    Trades computation for memory by recomputing activations
    during backward pass instead of storing them.
    """

    def __init__(
        self,
        checkpoint_ratio: float = 0.5,
        max_checkpoints: int = 10,
    ):
        self.checkpoint_ratio = checkpoint_ratio
        self.max_checkpoints = max_checkpoints

        # Checkpoints storage
        self.checkpoints: dict[int, np.ndarray] = {}
        self.checkpoint_indices: list[int] = []

        # Recomputation functions
        self.recompute_fns: dict[int, Any] = {}

    def should_checkpoint(self, layer_idx: int, total_layers: int) -> bool:
        """Determine if layer should be checkpointed."""
        if len(self.checkpoint_indices) >= self.max_checkpoints:
            return False

        # Checkpoint every N layers
        checkpoint_interval = max(
            1,
            int(total_layers * (1 - self.checkpoint_ratio)),
        )
        return layer_idx % checkpoint_interval == 0

    def save_checkpoint(
        self,
        layer_idx: int,
        activation: np.ndarray,
        recompute_fn: Any | None = None,
    ):
        """
        Save activation checkpoint.

        Args:
            layer_idx: Layer index
            activation: Activation to checkpoint
            recompute_fn: Function to recompute activation
        """
        self.checkpoints[layer_idx] = activation.copy()
        self.checkpoint_indices.append(layer_idx)

        if recompute_fn is not None:
            self.recompute_fns[layer_idx] = recompute_fn

    def get_checkpoint(self, layer_idx: int) -> np.ndarray | None:
        """Get checkpointed activation."""
        return self.checkpoints.get(layer_idx)

    def recompute(
        self,
        layer_idx: int,
        input_activation: np.ndarray,
    ) -> np.ndarray | None:
        """Recompute activation from checkpoint."""
        if layer_idx in self.recompute_fns:
            return self.recompute_fns[layer_idx](input_activation)
        return None

    def clear(self):
        """Clear all checkpoints."""
        self.checkpoints.clear()
        self.checkpoint_indices.clear()
        self.recompute_fns.clear()

    def get_memory_saved(self) -> int:
        """Estimate memory saved by checkpointing."""
        # Estimate based on what wasn't stored
        if not self.checkpoints:
            return 0

        avg_size = np.mean([c.nbytes for c in self.checkpoints.values()])
        # Assume we saved storing ~50% of activations
        return int(avg_size * len(self.checkpoints))


class EmbeddingCompressor:
    """
    Embedding compression using quantization and dimensionality reduction.

    Reduces memory footprint of embedding storage.
    """

    def __init__(
        self,
        target_dim: int | None = None,
        quantization_bits: int = 8,
        use_pca: bool = True,
    ):
        self.target_dim = target_dim
        self.quantization_bits = quantization_bits
        self.use_pca = use_pca

        # PCA components
        self.pca_components: np.ndarray | None = None
        self.pca_mean: np.ndarray | None = None

        # Quantization parameters
        self.quant_min: float = 0.0
        self.quant_max: float = 1.0
        self.quant_scale: float = 1.0

    def fit(self, embeddings: np.ndarray):
        """
        Fit compressor on embeddings.

        Args:
            embeddings: Training embeddings (n_samples, dim)
        """
        if self.use_pca and self.target_dim:
            self._fit_pca(embeddings)

        # Fit quantization
        self.quant_min = float(embeddings.min())
        self.quant_max = float(embeddings.max())
        self.quant_scale = (self.quant_max - self.quant_min) / (2**self.quantization_bits - 1)

    def _fit_pca(self, embeddings: np.ndarray):
        """Fit PCA for dimensionality reduction."""
        # Center data
        self.pca_mean = embeddings.mean(axis=0)
        centered = embeddings - self.pca_mean

        # Compute covariance and eigenvectors
        cov = np.cov(centered.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        # Sort by eigenvalue (descending)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvectors = eigenvectors[:, idx]

        # Keep top components
        self.pca_components = eigenvectors[:, : self.target_dim]

    def compress(self, embedding: np.ndarray) -> np.ndarray:
        """
        Compress an embedding.

        Args:
            embedding: Original embedding

        Returns:
            Compressed embedding
        """
        result = embedding.copy()

        # Apply PCA
        if self.use_pca and self.pca_components is not None:
            centered = result - self.pca_mean
            result = np.dot(centered, self.pca_components)

        # Quantize
        result = self._quantize(result)

        return result

    def decompress(self, compressed: np.ndarray) -> np.ndarray:
        """
        Decompress an embedding.

        Args:
            compressed: Compressed embedding

        Returns:
            Decompressed embedding
        """
        # Dequantize
        result = self._dequantize(compressed)

        # Inverse PCA
        if self.use_pca and self.pca_components is not None:
            result = np.dot(result, self.pca_components.T) + self.pca_mean

        return result

    def _quantize(self, data: np.ndarray) -> np.ndarray:
        """Quantize to lower precision."""
        # Normalize to [0, 1]
        normalized = (data - self.quant_min) / (self.quant_max - self.quant_min + 1e-10)
        normalized = np.clip(normalized, 0, 1)

        # Quantize
        max_val = 2**self.quantization_bits - 1
        quantized = np.round(normalized * max_val)

        if self.quantization_bits <= 8:
            return quantized.astype(np.uint8)
        elif self.quantization_bits <= 16:
            return quantized.astype(np.uint16)
        else:
            return quantized.astype(np.uint32)

    def _dequantize(self, data: np.ndarray) -> np.ndarray:
        """Dequantize back to float."""
        max_val = 2**self.quantization_bits - 1
        normalized = data.astype(np.float32) / max_val
        return normalized * (self.quant_max - self.quant_min) + self.quant_min

    def get_compression_ratio(self, original_dim: int) -> float:
        """Get compression ratio."""
        compressed_dim = self.target_dim or original_dim
        bits_original = original_dim * 32  # float32
        bits_compressed = compressed_dim * self.quantization_bits
        return bits_original / bits_compressed


class MemoryEfficientBuffer:
    """
    Memory-efficient circular buffer for storing history.

    Uses weak references and automatic cleanup.
    """

    def __init__(
        self,
        max_size: int = 1000,
        dtype: np.dtype = np.float32,
    ):
        self.max_size = max_size
        self.dtype = dtype

        # Circular buffer
        self.buffer: list[np.ndarray | None] = [None] * max_size
        self.write_idx = 0
        self.size = 0

        # Weak references for large objects
        self.weak_refs: dict[int, weakref.ref] = {}

    def append(self, item: np.ndarray):
        """Append item to buffer."""
        # Store in circular buffer
        self.buffer[self.write_idx] = item

        # Update indices
        self.write_idx = (self.write_idx + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def get(self, idx: int) -> np.ndarray | None:
        """Get item by index (0 = oldest)."""
        if idx >= self.size:
            return None

        actual_idx = (self.write_idx - self.size + idx) % self.max_size
        return self.buffer[actual_idx]

    def get_recent(self, n: int) -> list[np.ndarray]:
        """Get n most recent items."""
        n = min(n, self.size)
        result = []
        for i in range(n):
            idx = (self.write_idx - 1 - i) % self.max_size
            if self.buffer[idx] is not None:
                result.append(self.buffer[idx])
        return result[::-1]  # Oldest first

    def clear(self):
        """Clear buffer."""
        self.buffer = [None] * self.max_size
        self.write_idx = 0
        self.size = 0
        self.weak_refs.clear()

    def __len__(self) -> int:
        return self.size


class MemoryManager:
    """
    Centralized memory management for AutoDAN.

    Coordinates all memory optimization components.
    """

    def __init__(
        self,
        pool_size: int = 100 * 1024 * 1024,  # 100MB
        enable_checkpointing: bool = True,
        enable_compression: bool = True,
        target_embedding_dim: int = 128,
    ):
        self.enable_checkpointing = enable_checkpointing
        self.enable_compression = enable_compression

        # Components
        self.pool = AdaptiveMemoryPool(initial_size=pool_size)
        self.checkpointing = GradientCheckpointing() if enable_checkpointing else None
        self.compressor = (
            EmbeddingCompressor(
                target_dim=target_embedding_dim,
            )
            if enable_compression
            else None
        )

        # Buffers
        self.embedding_buffer = MemoryEfficientBuffer(max_size=10000)
        self.gradient_buffer = MemoryEfficientBuffer(max_size=1000)

        # Memory tracking
        self.memory_history: list[int] = []

    def allocate(
        self,
        shape: tuple[int, ...],
        dtype: np.dtype = np.float32,
    ) -> np.ndarray:
        """Allocate array from pool."""
        return self.pool.allocate(shape, dtype)

    def release(self, arr: np.ndarray):
        """Release array back to pool."""
        self.pool.release(arr)

    def store_embedding(
        self,
        embedding: np.ndarray,
        compress: bool = True,
    ) -> np.ndarray:
        """
        Store embedding with optional compression.

        Args:
            embedding: Embedding to store
            compress: Whether to compress

        Returns:
            Stored (possibly compressed) embedding
        """
        if compress and self.compressor is not None:
            embedding = self.compressor.compress(embedding)

        self.embedding_buffer.append(embedding)
        return embedding

    def retrieve_embedding(
        self,
        idx: int,
        decompress: bool = True,
    ) -> np.ndarray | None:
        """
        Retrieve stored embedding.

        Args:
            idx: Buffer index
            decompress: Whether to decompress

        Returns:
            Retrieved embedding
        """
        embedding = self.embedding_buffer.get(idx)

        if embedding is not None and decompress and self.compressor is not None:
            embedding = self.compressor.decompress(embedding)

        return embedding

    def checkpoint_activation(
        self,
        layer_idx: int,
        activation: np.ndarray,
    ):
        """Checkpoint activation for gradient computation."""
        if self.checkpointing is not None:
            self.checkpointing.save_checkpoint(layer_idx, activation)

    def get_checkpoint(self, layer_idx: int) -> np.ndarray | None:
        """Get checkpointed activation."""
        if self.checkpointing is not None:
            return self.checkpointing.get_checkpoint(layer_idx)
        return None

    def track_memory(self):
        """Track current memory usage."""
        import psutil

        process = psutil.Process()
        memory_bytes = process.memory_info().rss
        self.memory_history.append(memory_bytes)

        # Keep limited history
        if len(self.memory_history) > 1000:
            self.memory_history = self.memory_history[-1000:]

    def optimize_memory(self):
        """Run memory optimization."""
        # Shrink pools
        self.pool.shrink()

        # Clear checkpoints if not needed
        if self.checkpointing is not None:
            self.checkpointing.clear()

        # Force garbage collection
        gc.collect()

    def get_stats(self) -> dict[str, Any]:
        """Get memory management statistics."""
        stats = {
            "pool": self.pool.get_stats(),
            "embedding_buffer_size": len(self.embedding_buffer),
            "gradient_buffer_size": len(self.gradient_buffer),
        }

        if self.memory_history:
            stats["current_memory_mb"] = self.memory_history[-1] / (1024 * 1024)
            stats["peak_memory_mb"] = max(self.memory_history) / (1024 * 1024)

        if self.checkpointing is not None:
            stats["checkpointing"] = {
                "n_checkpoints": len(self.checkpointing.checkpoints),
                "memory_saved": self.checkpointing.get_memory_saved(),
            }

        if self.compressor is not None:
            stats["compression_ratio"] = self.compressor.get_compression_ratio(
                384  # Default embedding dim
            )

        return stats

    def reset(self):
        """Reset memory manager."""
        self.pool.clear()
        if self.checkpointing is not None:
            self.checkpointing.clear()
        self.embedding_buffer.clear()
        self.gradient_buffer.clear()
        self.memory_history.clear()
        gc.collect()
