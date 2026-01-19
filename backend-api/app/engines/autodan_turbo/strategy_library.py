"""
Strategy Library for AutoDAN-Turbo.

Provides persistent storage and similarity-based retrieval of jailbreak strategies.
Based on the paper's formalization of strategy libraries.

Enhanced with:
- Config-driven embedding provider selection
- Integration with centralized EmbeddingService
- Support for API-based embeddings via config
- Automatic fallback to local embeddings
"""

import logging
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml

try:
    import faiss

    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False

from .models import JailbreakStrategy, StrategyMetadata, StrategySource

logger = logging.getLogger(__name__)


# =============================================================================
# Config-Driven Embedding Integration
# =============================================================================


class EmbeddingProvider:
    """
    Config-driven embedding provider for strategy library.

    Provides embeddings using:
    1. Backend EmbeddingService (preferred when available)
    2. Local sentence-transformers (fallback)

    The provider selection is driven by the AI config manager.
    """

    _embedding_service = None
    _local_model = None
    _initialized = False
    _use_api = False

    @classmethod
    def _initialize(cls) -> None:
        """Initialize embedding provider lazily."""
        if cls._initialized:
            return

        cls._initialized = True

        # Try to use backend embedding service
        try:
            from app.services.embedding_service import get_embedding_service

            cls._embedding_service = get_embedding_service()
            cls._use_api = True
            logger.info("StrategyLibrary using backend EmbeddingService " "(config-driven)")
        except ImportError:
            logger.debug("Backend EmbeddingService not available, " "will use local embeddings")
            cls._use_api = False
        except Exception as e:
            logger.warning(
                f"Failed to initialize EmbeddingService: {e}, " f"falling back to local embeddings"
            )
            cls._use_api = False

    @classmethod
    def get_dimension(cls, provider: str | None = None) -> int:
        """Get embedding dimension from config or default."""
        cls._initialize()

        if cls._use_api and cls._embedding_service:
            return cls._embedding_service.get_embedding_dimensions(provider or "openai")

        # Default for sentence-transformers all-MiniLM-L6-v2
        return 384

    @classmethod
    def encode(
        cls,
        text: str,
        provider: str | None = None,
    ) -> list[float]:
        """
        Encode text to embedding vector.

        Uses config-driven provider selection when backend available.
        """
        cls._initialize()

        if cls._use_api and cls._embedding_service:
            return cls._encode_via_api(text, provider)
        else:
            return cls._encode_local(text)

    @classmethod
    def _encode_via_api(
        cls,
        text: str,
        provider: str | None = None,
    ) -> list[float]:
        """Encode using backend embedding service."""
        import asyncio

        try:
            # Handle async in sync context
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None

            if loop and loop.is_running():
                # Already in async context - use thread pool
                import concurrent.futures

                with concurrent.futures.ThreadPoolExecutor() as pool:
                    future = pool.submit(
                        asyncio.run,
                        cls._embedding_service.generate_embedding(text, provider=provider),
                    )
                    return future.result()
            else:
                # Not in async context
                return asyncio.run(
                    cls._embedding_service.generate_embedding(text, provider=provider)
                )
        except Exception as e:
            logger.error(f"API embedding failed: {e}, falling back to local")
            return cls._encode_local(text)

    @classmethod
    def _encode_local(cls, text: str) -> list[float]:
        """Encode using local sentence-transformers."""
        if cls._local_model is None:
            try:
                from sentence_transformers import SentenceTransformer

                cls._local_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
                logger.info("Loaded local sentence-transformer model")
            except ImportError:
                logger.error("sentence-transformers not installed, " "returning empty embedding")
                return []

        try:
            embedding = cls._local_model.encode(
                text,
                normalize_embeddings=True,
                convert_to_numpy=True,
            )
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Local embedding failed: {e}")
            return []

    @classmethod
    def get_provider_info(cls) -> dict[str, Any]:
        """Get information about the current embedding provider."""
        cls._initialize()

        if cls._use_api and cls._embedding_service:
            providers = cls._embedding_service.get_embedding_providers()
            return {
                "type": "api",
                "service": "EmbeddingService",
                "providers": providers,
                "config_driven": True,
            }
        else:
            return {
                "type": "local",
                "model": "sentence-transformers/all-MiniLM-L6-v2",
                "config_driven": False,
            }


# Constants for validation
MAX_TEMPLATE_SIZE = 50000  # 50KB max template size
MAX_STRATEGY_NAME_LENGTH = 200
MAX_DESCRIPTION_LENGTH = 5000
MAX_LIBRARY_SIZE = 10000  # Maximum strategies in library

# complex YAML patterns that could indicate injection attempts
complex_YAML_PATTERNS = ["!!python", "!!ruby", "!include", "<<:", "!!exec", "!!map"]


class StrategyLibrary:
    """
    Persistent storage and retrieval system for jailbreak strategies.

    Features:
    - YAML-based persistence for human-readable storage
    - Embedding-based similarity search (accelerated by FAISS)
    - Usage statistics tracking for adaptive strategy selection
    - Deduplication based on semantic similarity
    - Config-driven embedding provider selection
    - Integration with centralized EmbeddingService
    """

    def __init__(
        self,
        storage_dir: str | Path | None = None,
        similarity_threshold: float = 0.85,
        embedding_model=None,
        embedding_provider: str | None = None,
        use_config_embeddings: bool = True,
    ):
        """
        Initialize the strategy library.

        Args:
            storage_dir: Directory for YAML storage (default: data/strategies/)
            similarity_threshold: Threshold for deduplication (0-1)
            embedding_model: Model for computing embeddings (optional)
            embedding_provider: Override provider for embeddings
            use_config_embeddings: Use config-driven embedding service
        """
        if storage_dir is None:
            # Default to data/strategies/ relative to backend-api
            base_dir = Path(__file__).parent.parent.parent.parent
            storage_dir = base_dir / "data" / "strategies"

        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        # Alias for compatibility with AutoDANTurboLifelongEngine
        self.storage_path = self.storage_dir

        self.similarity_threshold = similarity_threshold
        self.embedding_model = embedding_model
        self.embedding_provider = embedding_provider
        self.use_config_embeddings = use_config_embeddings

        # Thread safety lock for concurrent access
        self._lock = threading.RLock()

        # In-memory strategy cache
        self._strategies: dict[str, JailbreakStrategy] = {}

        # FAISS Index
        self.index = None
        self.index_ids = []  # Map FAISS row index to strategy ID

        # Load existing strategies
        self._load_all()

        logger.info(
            f"StrategyLibrary initialized with {len(self._strategies)} strategies. FAISS available: {HAS_FAISS}"
        )

    # ========================
    # Core CRUD Operations
    # ========================

    def add_strategy(
        self, strategy: JailbreakStrategy, check_duplicate: bool = True
    ) -> tuple[bool, str]:
        """
        Add a new strategy to the library.

        Args:
            strategy: The strategy to add
            check_duplicate: Whether to check for duplicates

        Returns:
            Tuple of (success, message/strategy_id)
        """
        # Validate strategy before adding
        is_valid, validation_msg = self._validate_strategy(strategy)
        if not is_valid:
            logger.warning(f"Strategy validation failed: {validation_msg}")
            return False, validation_msg

        with self._lock:
            # Check library size limit
            if len(self._strategies) >= MAX_LIBRARY_SIZE:
                return False, f"Library size limit reached ({MAX_LIBRARY_SIZE})"

            # Compute embedding if model available and missing
            if self.embedding_model and strategy.embedding is None:
                strategy.embedding = self._compute_embedding(strategy.template)

            if check_duplicate:
                duplicate = self._find_duplicate(strategy)
                if duplicate:
                    logger.info(f"Strategy '{strategy.name}' is a duplicate of '{duplicate.name}'")
                    return False, f"Duplicate of: {duplicate.id}"

            self._strategies[strategy.id] = strategy
            self._save_strategy(strategy)

            # Update FAISS index
            if HAS_FAISS and self.embedding_model and strategy.embedding:
                self._add_to_index(strategy)

        logger.info(f"Added strategy: {strategy.name} ({strategy.id})")
        return True, strategy.id

    def _validate_strategy(self, strategy: JailbreakStrategy) -> tuple[bool, str]:
        """
        Validate a strategy before adding to the library.

        Checks for:
        - Template size limits
        - Name and description length limits
        - complex YAML patterns

        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check template size
        if len(strategy.template) > MAX_TEMPLATE_SIZE:
            return False, f"Template exceeds maximum size ({MAX_TEMPLATE_SIZE} bytes)"

        # Check name length
        if len(strategy.name) > MAX_STRATEGY_NAME_LENGTH:
            return False, f"Strategy name exceeds maximum length ({MAX_STRATEGY_NAME_LENGTH})"

        # Check description length
        if len(strategy.description) > MAX_DESCRIPTION_LENGTH:
            return False, f"Description exceeds maximum length ({MAX_DESCRIPTION_LENGTH})"

        # Check for complex YAML patterns (potential injection)
        template_lower = strategy.template.lower()
        for pattern in complex_YAML_PATTERNS:
            if pattern.lower() in template_lower:
                return False, f"Template contains potentially complex YAML construct: {pattern}"

        # Check description for complex patterns too
        desc_lower = strategy.description.lower()
        for pattern in complex_YAML_PATTERNS:
            if pattern.lower() in desc_lower:
                return (
                    False,
                    f"Description contains potentially complex YAML construct: {pattern}",
                )

        return True, ""

    def get_strategy(self, strategy_id: str) -> JailbreakStrategy | None:
        """Get a strategy by ID."""
        return self._strategies.get(strategy_id)

    def get_all_strategies(self) -> list[JailbreakStrategy]:
        """Get all strategies in the library."""
        return list(self._strategies.values())

    def update_strategy(self, strategy: JailbreakStrategy) -> bool:
        """Update an existing strategy."""
        with self._lock:
            if strategy.id not in self._strategies:
                return False

            # If embedding changed, we need to handle that, but typically updates are status/usage
            # Re-saving strategy
            strategy.metadata.updated_at = datetime.utcnow()
            self._strategies[strategy.id] = strategy
            self._save_strategy(strategy)

            # If embedding changed or is new, naive approach: rebuild index if needed or just update
            # For simplicity, if we were doing complex updates we might rebuild.
            # Assuming updates don't change embeddings often.

            return True

    def remove_strategy(self, strategy_id: str) -> bool:
        """Remove a strategy from the library."""
        with self._lock:
            if strategy_id not in self._strategies:
                return False

            strategy = self._strategies.pop(strategy_id)

            # Remove from file
            file_path = self._get_strategy_file_path(strategy)
            if file_path.exists():
                file_path.unlink()

            # Rebuild FAISS index to remove vector
            if HAS_FAISS and self.index is not None:
                self._build_index()

            logger.info(f"Removed strategy: {strategy.name} ({strategy_id})")
            return True

    # ========================
    # Retrieval Methods
    # ========================

    def search(
        self, query: str, top_k: int = 5, min_score: float = 0.0
    ) -> list[tuple[JailbreakStrategy, float]]:
        """
        Search for strategies similar to a query.

        Args:
            query: The search query (typically the complex request)
            top_k: Maximum number of results
            min_score: Minimum similarity score (0-1)

        Returns:
            List of (strategy, similarity_score) tuples, sorted by score descending
        """
        if not self._strategies:
            return []

        if self.embedding_model:
            return self._embedding_search(query, top_k, min_score)
        else:
            return self._keyword_search(query, top_k)

    def get_top_strategies(
        self, top_k: int = 5, by: str = "success_rate"
    ) -> list[JailbreakStrategy]:
        """
        Get top strategies by performance metric.

        Args:
            top_k: Number of strategies to return
            by: Metric to sort by ("success_rate", "average_score", "usage_count")
        """
        strategies = list(self._strategies.values())

        if by == "success_rate":
            strategies.sort(key=lambda s: s.metadata.success_rate, reverse=True)
        elif by == "average_score":
            strategies.sort(key=lambda s: s.metadata.average_score, reverse=True)
        elif by == "usage_count":
            strategies.sort(key=lambda s: s.metadata.usage_count, reverse=True)
        else:
            raise ValueError(f"Unknown sort metric: {by}")

        return strategies[:top_k]

    def get_strategies_by_source(self, source: StrategySource) -> list[JailbreakStrategy]:
        """Get all strategies from a specific source."""
        return [s for s in self._strategies.values() if s.metadata.source == source]

    def get_strategies_by_tag(self, tag: str) -> list[JailbreakStrategy]:
        """Get all strategies with a specific tag."""
        return [s for s in self._strategies.values() if tag in s.tags]

    # ========================
    # Statistics Updates
    # ========================

    def update_statistics(self, strategy_id: str, score: float, success: bool) -> bool:
        """
        Update usage statistics for a strategy after an attack attempt.

        Thread-safe implementation using lock.

        Args:
            strategy_id: ID of the strategy used
            score: Attack score (1-10)
            success: Whether the attack succeeded
        """
        with self._lock:
            strategy = self._strategies.get(strategy_id)
            if not strategy:
                return False

            strategy.update_statistics(score, success)
            self._save_strategy(strategy)

        logger.debug(
            f"Updated stats for {strategy.name}: "
            f"usage={strategy.metadata.usage_count}, "
            f"success_rate={strategy.metadata.success_rate:.2%}"
        )
        return True

    # ========================
    # Performance Decay (OPT-KT-2)
    # ========================

    def apply_performance_decay(
        self, decay_factor: float = 0.95, min_usage_for_decay: int = 5
    ) -> int:
        """
        Apply performance decay to strategy statistics.

        This implements OPT-KT-2 from the optimization report:
        - Older performance data is weighted less over time
        - Helps strategies adapt to changing target model behavior
        - Prevents stale strategies from dominating selection

        Expected Impact: Better strategy selection over time.

        Args:
            decay_factor: Multiplier for decaying scores (0-1, default 0.95)
            min_usage_for_decay: Minimum usage count before decay applies

        Returns:
            Number of strategies that had decay applied
        """
        decayed_count = 0

        with self._lock:
            for strategy in self._strategies.values():
                if strategy.metadata.usage_count < min_usage_for_decay:
                    continue

                # Apply decay to total score (affects average)
                old_total = strategy.metadata.total_score
                strategy.metadata.total_score *= decay_factor

                # Apply decay to success count (affects success rate)
                # Use floor to ensure integer
                old_success = strategy.metadata.success_count
                strategy.metadata.success_count = int(
                    strategy.metadata.success_count * decay_factor
                )

                # Apply decay to usage count proportionally
                old_usage = strategy.metadata.usage_count
                strategy.metadata.usage_count = max(
                    min_usage_for_decay, int(strategy.metadata.usage_count * decay_factor)
                )

                decayed_count += 1

                logger.debug(
                    f"Applied decay to {strategy.name}: "
                    f"score {old_total:.1f}->{strategy.metadata.total_score:.1f}, "
                    f"success {old_success}->{strategy.metadata.success_count}, "
                    f"usage {old_usage}->{strategy.metadata.usage_count}"
                )

        if decayed_count > 0:
            logger.info(f"Applied performance decay to {decayed_count} strategies")

        return decayed_count

    def get_strategies_with_decay_scores(
        self, top_k: int = 10, recency_weight: float = 0.3
    ) -> list[tuple[JailbreakStrategy, float]]:
        """
        Get top strategies with recency-weighted scores.

        Combines historical performance with recency to favor
        recently successful strategies.

        Args:
            top_k: Number of strategies to return
            recency_weight: Weight for recency factor (0-1)

        Returns:
            List of (strategy, weighted_score) tuples
        """
        now = datetime.utcnow()
        scored_strategies = []

        for strategy in self._strategies.values():
            # Base score from success rate and average score
            base_score = (
                strategy.metadata.success_rate * 0.5
                + (strategy.metadata.average_score / 10.0) * 0.5
            )

            # Recency factor (days since last update)
            days_since_update = (now - strategy.metadata.updated_at).days
            recency_factor = 1.0 / (1.0 + days_since_update * 0.1)

            # Combined score
            weighted_score = base_score * (1 - recency_weight) + recency_factor * recency_weight

            scored_strategies.append((strategy, weighted_score))

        # Sort by weighted score descending
        scored_strategies.sort(key=lambda x: x[1], reverse=True)

        return scored_strategies[:top_k]

    def prune_underperforming_strategies(
        self, min_usage: int = 10, min_success_rate: float = 0.1, max_age_days: int = 90
    ) -> int:
        """
        Remove strategies that consistently underperform.

        Args:
            min_usage: Minimum usage count before pruning eligible
            min_success_rate: Minimum success rate to keep
            max_age_days: Maximum days without update before pruning

        Returns:
            Number of strategies pruned
        """
        now = datetime.utcnow()
        to_remove = []

        with self._lock:
            for strategy_id, strategy in self._strategies.items():
                # Skip strategies without enough usage data
                if strategy.metadata.usage_count < min_usage:
                    continue

                # Check success rate
                if strategy.metadata.success_rate < min_success_rate:
                    days_since_update = (now - strategy.metadata.updated_at).days
                    if days_since_update > max_age_days:
                        to_remove.append(strategy_id)
                        logger.info(
                            f"Pruning underperforming strategy: {strategy.name} "
                            f"(success_rate={strategy.metadata.success_rate:.1%}, "
                            f"age={days_since_update} days)"
                        )

        # Remove strategies outside the lock to avoid deadlock
        for strategy_id in to_remove:
            self.remove_strategy(strategy_id)

        if to_remove:
            logger.info(f"Pruned {len(to_remove)} underperforming strategies")

        return len(to_remove)

    # ========================
    # Persistence
    # ========================

    def _load_all(self) -> None:
        """Load all strategies from YAML files."""
        yaml_files = list(self.storage_dir.glob("*.yaml")) + list(self.storage_dir.glob("*.yml"))

        for file_path in yaml_files:
            try:
                with open(file_path, encoding="utf-8") as f:
                    data = yaml.safe_load(f)

                if data is None:
                    continue

                # Handle single strategy or list of strategies
                if isinstance(data, list):
                    for item in data:
                        self._load_strategy_from_dict(item)
                elif isinstance(data, dict):
                    # Check if it's a direct strategy or wrapped
                    if "strategies" in data:
                        for item in data["strategies"]:
                            self._load_strategy_from_dict(item)
                    else:
                        self._load_strategy_from_dict(data)

            except Exception as e:
                logger.error(f"Failed to load {file_path}: {e}")

        # Build FAISS index after loading all
        if HAS_FAISS and self.embedding_model:
            self._build_index()

    def _load_strategy_from_dict(self, data: dict) -> None:
        """Load a single strategy from dictionary data."""
        try:
            # Skip non-strategy data (e.g., failure_tracking.yaml)
            # A valid strategy must have at least 'name', 'description', and 'template'
            required_fields = {"name", "description", "template"}
            if not required_fields.issubset(data.keys()):
                logger.debug(f"Skipping non-strategy data with keys: {list(data.keys())}")
                return

            # Handle metadata separately
            if "metadata" in data:
                meta_data = data["metadata"]
                if isinstance(meta_data.get("source"), str):
                    meta_data["source"] = StrategySource(meta_data["source"])
                if isinstance(meta_data.get("created_at"), str):
                    meta_data["created_at"] = datetime.fromisoformat(meta_data["created_at"])
                if isinstance(meta_data.get("updated_at"), str):
                    meta_data["updated_at"] = datetime.fromisoformat(meta_data["updated_at"])
                data["metadata"] = StrategyMetadata(**meta_data)

            strategy = JailbreakStrategy(**data)
            self._strategies[strategy.id] = strategy

        except Exception as e:
            logger.error(f"Failed to parse strategy: {e}")

    def _save_strategy(self, strategy: JailbreakStrategy) -> None:
        """Save a single strategy to YAML."""
        file_path = self._get_strategy_file_path(strategy)

        # Convert to dict, handling special types
        data = strategy.model_dump(exclude={"embedding"})
        data["metadata"]["source"] = strategy.metadata.source.value
        data["metadata"]["created_at"] = strategy.metadata.created_at.isoformat()
        data["metadata"]["updated_at"] = strategy.metadata.updated_at.isoformat()

        with open(file_path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    def _get_strategy_file_path(self, strategy: JailbreakStrategy) -> Path:
        """Get the file path for a strategy."""
        # Use sanitized name + ID for filename
        safe_name = "".join(c if c.isalnum() else "_" for c in strategy.name.lower())
        return self.storage_dir / f"{safe_name}_{strategy.id[:8]}.yaml"

    def save_all(self) -> None:
        """Save all strategies to disk."""
        for strategy in self._strategies.values():
            self._save_strategy(strategy)
        logger.info(f"Saved {len(self._strategies)} strategies")

    # ========================
    # Similarity & Deduplication
    # ========================

    def _build_index(self):
        """Build or rebuild FAISS index from current strategies."""
        if not HAS_FAISS or not self._strategies:
            return

        embeddings = []
        ids = []
        for strategy_id, strategy in self._strategies.items():
            if strategy.embedding is not None:
                embeddings.append(strategy.embedding)
                ids.append(strategy_id)

        if not embeddings:
            return

        # Initialize index
        dimension = len(embeddings[0])
        # Inner Product for cosine similarity (normalized vectors)
        self.index = faiss.IndexFlatIP(dimension)

        # Add vectors
        vectors = np.array(embeddings).astype("float32")
        # Normalize if not already (IndexFlatIP requires normalized vectors for cosine sim)
        faiss.normalize_L2(vectors)
        self.index.add(vectors)
        self.index_ids = ids
        logger.debug(f"Built FAISS index with {len(ids)} vectors, dimension {dimension}")

    def _add_to_index(self, strategy: JailbreakStrategy):
        """Add a single strategy to the FAISS index."""
        if not HAS_FAISS or strategy.embedding is None:
            return

        embedding = np.array([strategy.embedding]).astype("float32")
        faiss.normalize_L2(embedding)

        if self.index is None:
            self.index = faiss.IndexFlatIP(embedding.shape[1])

        self.index.add(embedding)
        self.index_ids.append(strategy.id)

    def _find_duplicate(self, strategy: JailbreakStrategy) -> JailbreakStrategy | None:
        """Find a duplicate strategy based on similarity."""
        if not self._strategies:
            return None

        if self.embedding_model:
            # Use search to find most similar
            # Since _embedding_search now uses FAISS or fallback, this is consistent
            similar = self.search(strategy.template, top_k=1, min_score=self.similarity_threshold)
            if similar:
                return similar[0][0]
        else:
            # Fallback: exact template match
            for existing in self._strategies.values():
                if strategy.template.strip() == existing.template.strip():
                    return existing

        return None

    def _embedding_search(
        self, query: str, top_k: int, min_score: float
    ) -> list[tuple[JailbreakStrategy, float]]:
        """Search using embedding similarity (FAISS if available)."""
        query_embedding_list = self._compute_embedding(query)

        if not query_embedding_list:
            return []

        if HAS_FAISS and self.index is not None and self.index.ntotal > 0:
            # FAISS Search
            query_vec = np.array([query_embedding_list]).astype("float32")
            faiss.normalize_L2(query_vec)

            # Since distances is cosine similarity (IndexFlatIP), and range is -1 to 1
            distances, indices = self.index.search(query_vec, top_k)

            results = []
            for i, score in zip(indices[0], distances[0], strict=False):
                if i < 0 or i >= len(self.index_ids):
                    continue
                if score < min_score:
                    continue

                strategy_id = self.index_ids[i]
                strategy = self._strategies.get(strategy_id)
                if strategy:
                    results.append((strategy, float(score)))
            return results

        else:
            # Fallback linear search
            scores = []
            for strategy in self._strategies.values():
                if strategy.embedding:
                    similarity = self._cosine_similarity(query_embedding_list, strategy.embedding)
                    if similarity >= min_score:
                        scores.append((strategy, similarity))

            scores.sort(key=lambda x: x[1], reverse=True)
            return scores[:top_k]

    def _keyword_search(self, query: str, top_k: int) -> list[tuple[JailbreakStrategy, float]]:
        """Fallback keyword-based search."""
        query_words = set(query.lower().split())

        scores = []
        for strategy in self._strategies.values():
            # Combine template, description, and tags for matching
            text = f"{strategy.template} {strategy.description} {' '.join(strategy.tags)}"
            text_words = set(text.lower().split())

            # Jaccard similarity
            if not query_words or not text_words:
                similarity = 0.0
            else:
                intersection = len(query_words & text_words)
                union = len(query_words | text_words)
                similarity = intersection / union if union > 0 else 0.0

            scores.append((strategy, similarity))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def _compute_embedding(self, text: str) -> list[float]:
        """
        Compute embedding for text.

        Uses config-driven embedding service when available,
        falls back to provided embedding_model or local embeddings.
        """
        # Priority 1: Use config-driven embedding service
        if self.use_config_embeddings:
            try:
                embedding = EmbeddingProvider.encode(text, provider=self.embedding_provider)
                if embedding:
                    return embedding
            except Exception as e:
                logger.warning(f"Config-driven embedding failed: {e}, " f"trying fallback")

        # Priority 2: Use provided embedding model
        if self.embedding_model is not None:
            try:
                # Handle different embedding model interfaces
                if hasattr(self.embedding_model, "encode"):
                    # SentenceTransformer-style
                    embedding = self.embedding_model.encode(text)
                    if hasattr(embedding, "tolist"):
                        return embedding.tolist()
                    return list(embedding)
                elif hasattr(self.embedding_model, "embed"):
                    # Custom interface
                    return self.embedding_model.embed(text)
                else:
                    logger.warning("Unknown embedding model interface")
            except Exception as e:
                logger.error(f"Embedding model failed: {e}")

        # Priority 3: Fall back to local sentence-transformers
        try:
            return EmbeddingProvider._encode_local(text)
        except Exception as e:
            logger.error(f"All embedding methods failed: {e}")
            return []

    def get_embedding_info(self) -> dict[str, Any]:
        """Get information about the embedding configuration."""
        return {
            "provider_override": self.embedding_provider,
            "use_config_embeddings": self.use_config_embeddings,
            "has_embedding_model": self.embedding_model is not None,
            "provider_info": EmbeddingProvider.get_provider_info(),
        }

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        """Compute cosine similarity between two vectors."""
        if not a or not b or len(a) != len(b):
            return 0.0

        a_arr = np.array(a)
        b_arr = np.array(b)

        dot = np.dot(a_arr, b_arr)
        norm_a = np.linalg.norm(a_arr)
        norm_b = np.linalg.norm(b_arr)

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return float(dot / (norm_a * norm_b))

    # ========================
    # Utility Methods
    # ========================

    def __len__(self) -> int:
        """Return number of strategies in library."""
        return len(self._strategies)

    def __contains__(self, strategy_id: str) -> bool:
        """Check if strategy exists."""
        return strategy_id in self._strategies

    def clear(self) -> None:
        """Clear all strategies (use with caution)."""
        self._strategies.clear()
        # Remove all YAML files
        for file_path in self.storage_dir.glob("*.yaml"):
            file_path.unlink()

        # Clear index
        self.index = None
        self.index_ids = []

        logger.warning("Cleared all strategies from library")
