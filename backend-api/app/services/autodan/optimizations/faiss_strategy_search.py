"""
FAISS-based Strategy Search for AutoDAN

This module provides intelligent semantic search for AutoDAN strategies using FAISS
vector similarity search. It enables:
- Fast similarity-based strategy retrieval
- Semantic strategy clustering and organization
- Dynamic strategy recommendation based on context
- Efficient strategy library management with vector indexing
"""

import asyncio
import hashlib
import logging
import pickle
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Try to import FAISS
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logger.warning("FAISS not available, falling back to linear search")

# Try to import embedding model
try:
    from ..llm import embedding_models as _embedding_models
    EMBEDDING_AVAILABLE = True
    del _embedding_models
except ImportError:
    EMBEDDING_AVAILABLE = False
    logger.warning("Embedding model not available")


@dataclass
class StrategyEntry:
    """Represents a strategy in the vector index"""
    strategy_id: str
    name: str
    definition: str
    examples: list[str] = field(default_factory=list)
    embedding: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    creation_time: float = field(default_factory=time.time)
    usage_count: int = 0
    effectiveness_scores: list[float] = field(default_factory=list)
    tags: set[str] = field(default_factory=set)


@dataclass
class StrategySearchResult:
    """Result from strategy search"""
    strategy: StrategyEntry
    similarity_score: float
    rank: int
    distance: float


@dataclass
class StrategyCluster:
    """Cluster of semantically similar strategies"""
    cluster_id: str
    centroid: np.ndarray
    strategies: list[StrategyEntry]
    cluster_name: str = ""
    coherence_score: float = 0.0


class FAISSStrategyIndex:
    """FAISS-based vector index for strategy search"""

    def __init__(
        self,
        dimension: int = 768,
        index_type: str = "IVF",
        nlist: int = 100,
        use_gpu: bool = False,
        similarity_threshold: float = 0.7
    ):
        self.dimension = dimension
        self.index_type = index_type
        self.nlist = nlist
        self.use_gpu = use_gpu
        self.similarity_threshold = similarity_threshold

        # FAISS index
        self.index: faiss.Index | None = None
        self.is_trained = False

        # Strategy mappings
        self.id_to_strategy: dict[int, StrategyEntry] = {}
        self.strategy_id_to_faiss_id: dict[str, int] = {}
        self.next_faiss_id = 0

        # Thread safety
        self._lock = asyncio.Lock()

        logger.info(f"FAISSStrategyIndex initialized: dim={dimension}, type={index_type}")

    async def initialize(self):
        """Initialize the FAISS index"""
        if not FAISS_AVAILABLE:
            logger.warning("FAISS not available, using fallback index")
            return

        async with self._lock:
            if self.index is not None:
                return

            logger.info("Initializing FAISS index...")

            if self.index_type == "IVF":
                # IVF index for larger datasets
                quantizer = faiss.IndexFlatIP(self.dimension)
                self.index = faiss.IndexIVFFlat(quantizer, self.dimension, self.nlist)
            elif self.index_type == "HNSW":
                # HNSW index for fast approximate search
                self.index = faiss.IndexHNSWFlat(self.dimension, 32)
                self.index.hnsw.efConstruction = 200
                self.index.hnsw.efSearch = 100
            else:
                # Flat index for exact search
                self.index = faiss.IndexFlatIP(self.dimension)

            if self.use_gpu and faiss.get_num_gpus() > 0:
                logger.info("Using GPU acceleration for FAISS")
                self.index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, self.index)

            logger.info(f"FAISS index initialized: {type(self.index).__name__}")

    async def add_strategy(self, strategy: StrategyEntry) -> bool:
        """Add a strategy to the index"""
        if not FAISS_AVAILABLE or self.index is None:
            return False

        if strategy.embedding is None:
            logger.warning(f"Strategy {strategy.strategy_id} has no embedding, skipping")
            return False

        async with self._lock:
            # Check if strategy already exists
            if strategy.strategy_id in self.strategy_id_to_faiss_id:
                logger.debug(f"Strategy {strategy.strategy_id} already in index, updating")
                await self.remove_strategy(strategy.strategy_id)

            # Normalize embedding for cosine similarity
            embedding = strategy.embedding.copy()
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm

            # Train index if needed (for IVF)
            if self.index_type == "IVF" and not self.is_trained and self.next_faiss_id >= self.nlist:
                await self._train_index()

            # Add to FAISS index
            faiss_id = self.next_faiss_id
            self.index.add(embedding.reshape(1, -1).astype('float32'))

            # Update mappings
            self.id_to_strategy[faiss_id] = strategy
            self.strategy_id_to_faiss_id[strategy.strategy_id] = faiss_id
            self.next_faiss_id += 1

            logger.debug(f"Added strategy {strategy.strategy_id} to index (FAISS ID: {faiss_id})")
            return True

    async def remove_strategy(self, strategy_id: str) -> bool:
        """Remove a strategy from the index"""
        async with self._lock:
            if strategy_id not in self.strategy_id_to_faiss_id:
                return False

            faiss_id = self.strategy_id_to_faiss_id[strategy_id]

            # Remove from mappings
            del self.id_to_strategy[faiss_id]
            del self.strategy_id_to_faiss_id[strategy_id]

            # Note: FAISS doesn't support efficient deletion, so we mark as deleted
            # In production, consider rebuilding the index periodically

            logger.debug(f"Removed strategy {strategy_id} from index")
            return True

    async def search_similar(
        self,
        query_embedding: np.ndarray,
        k: int = 10,
        min_similarity: float | None = None
    ) -> list[StrategySearchResult]:
        """Search for similar strategies"""
        if not FAISS_AVAILABLE or self.index is None or self.next_faiss_id == 0:
            return []

        min_similarity = min_similarity or self.similarity_threshold

        async with self._lock:
            # Normalize query embedding
            query = query_embedding.copy()
            norm = np.linalg.norm(query)
            if norm > 0:
                query = query / norm

            # Search in FAISS
            k_search = min(k * 2, self.next_faiss_id)  # Over-fetch for filtering
            similarities, indices = self.index.search(
                query.reshape(1, -1).astype('float32'),
                k_search
            )

            results = []
            for rank, (similarity, idx) in enumerate(zip(similarities[0], indices[0], strict=False)):
                if idx == -1:  # No more results
                    break

                if similarity < min_similarity:
                    continue

                if idx not in self.id_to_strategy:
                    continue  # Deleted strategy

                strategy = self.id_to_strategy[idx]
                distance = 1.0 - similarity  # Convert similarity to distance

                result = StrategySearchResult(
                    strategy=strategy,
                    similarity_score=float(similarity),
                    rank=rank,
                    distance=float(distance)
                )
                results.append(result)

                if len(results) >= k:
                    break

            return results

    async def search_by_text(
        self,
        query_text: str,
        embedding_model,
        k: int = 10,
        min_similarity: float | None = None
    ) -> list[StrategySearchResult]:
        """Search for strategies using text query"""
        if not embedding_model:
            logger.warning("No embedding model available for text search")
            return []

        # Generate embedding for query text
        query_embedding = await self._get_text_embedding_async(query_text, embedding_model)
        if query_embedding is None:
            return []

        return await self.search_similar(query_embedding, k, min_similarity)

    async def get_strategy_clusters(
        self,
        n_clusters: int = 5,
        min_cluster_size: int = 2
    ) -> list[StrategyCluster]:
        """Cluster strategies by semantic similarity"""
        if not FAISS_AVAILABLE or len(self.id_to_strategy) < n_clusters:
            return []

        try:
            from sklearn.cluster import KMeans
        except ImportError:
            logger.warning("sklearn not available for clustering")
            return []

        # Collect all embeddings
        embeddings = []
        strategies = []

        for strategy in self.id_to_strategy.values():
            if strategy.embedding is not None:
                embeddings.append(strategy.embedding)
                strategies.append(strategy)

        if len(embeddings) < n_clusters:
            return []

        embeddings_array = np.vstack(embeddings)

        # Perform K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(embeddings_array)
        centroids = kmeans.cluster_centers_

        # Create clusters
        clusters = []
        for cluster_id in range(n_clusters):
            cluster_strategies = [
                strategies[i] for i, label in enumerate(cluster_labels)
                if label == cluster_id
            ]

            if len(cluster_strategies) < min_cluster_size:
                continue

            # Calculate cluster coherence (average intra-cluster similarity)
            cluster_embeddings = [s.embedding for s in cluster_strategies if s.embedding is not None]
            coherence_score = 0.0
            if len(cluster_embeddings) > 1:
                similarities = []
                for i in range(len(cluster_embeddings)):
                    for j in range(i + 1, len(cluster_embeddings)):
                        sim = np.dot(cluster_embeddings[i], cluster_embeddings[j])
                        similarities.append(sim)
                coherence_score = np.mean(similarities)

            # Generate cluster name based on most common words
            cluster_name = self._generate_cluster_name(cluster_strategies)

            cluster = StrategyCluster(
                cluster_id=f"cluster_{cluster_id}",
                centroid=centroids[cluster_id],
                strategies=cluster_strategies,
                cluster_name=cluster_name,
                coherence_score=coherence_score
            )
            clusters.append(cluster)

        # Sort by coherence score
        clusters.sort(key=lambda c: c.coherence_score, reverse=True)

        return clusters

    async def _train_index(self):
        """Train the FAISS index (for IVF indices)"""
        if self.index_type != "IVF" or self.is_trained:
            return

        # Collect training data
        embeddings = []
        for strategy in self.id_to_strategy.values():
            if strategy.embedding is not None:
                normalized = strategy.embedding / (np.linalg.norm(strategy.embedding) + 1e-8)
                embeddings.append(normalized)

        if len(embeddings) >= self.nlist:
            training_data = np.vstack(embeddings).astype('float32')
            self.index.train(training_data)
            self.is_trained = True
            logger.info(f"FAISS index trained with {len(embeddings)} strategies")

    async def _get_text_embedding_async(self, text: str, embedding_model) -> np.ndarray | None:
        """Get embedding for text asynchronously"""
        try:
            loop = asyncio.get_event_loop()
            with ThreadPoolExecutor(max_workers=1) as executor:
                embedding = await loop.run_in_executor(executor, embedding_model.embed, text)
                return embedding
        except Exception as e:
            logger.error(f"Failed to generate embedding for text: {e}")
            return None

    def _generate_cluster_name(self, strategies: list[StrategyEntry]) -> str:
        """Generate a name for a strategy cluster"""
        # Extract key words from strategy names and definitions
        words = []
        for strategy in strategies:
            words.extend(strategy.name.lower().split())
            words.extend(strategy.definition.lower().split()[:10])  # First 10 words

        # Count word frequency
        word_counts = {}
        for word in words:
            if len(word) > 3 and word.isalpha():  # Filter short/non-alpha words
                word_counts[word] = word_counts.get(word, 0) + 1

        # Get top words
        top_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:3]

        if top_words:
            return " + ".join([word.title() for word, _ in top_words])
        else:
            return f"Cluster ({len(strategies)} strategies)"

    async def save_index(self, filepath: Path):
        """Save the FAISS index and mappings"""
        if not FAISS_AVAILABLE or self.index is None:
            return

        async with self._lock:
            try:
                # Save FAISS index
                index_path = filepath.with_suffix('.faiss')
                faiss.write_index(self.index, str(index_path))

                # Save mappings and metadata
                metadata = {
                    'id_to_strategy': self.id_to_strategy,
                    'strategy_id_to_faiss_id': self.strategy_id_to_faiss_id,
                    'next_faiss_id': self.next_faiss_id,
                    'is_trained': self.is_trained,
                    'dimension': self.dimension,
                    'index_type': self.index_type
                }

                with open(filepath.with_suffix('.pkl'), 'wb') as f:
                    pickle.dump(metadata, f)

                logger.info(f"FAISS index saved to {filepath}")
            except Exception as e:
                logger.error(f"Failed to save FAISS index: {e}")

    async def load_index(self, filepath: Path):
        """Load the FAISS index and mappings"""
        if not FAISS_AVAILABLE:
            return

        async with self._lock:
            try:
                # Load FAISS index
                index_path = filepath.with_suffix('.faiss')
                if index_path.exists():
                    self.index = faiss.read_index(str(index_path))

                    # Load mappings and metadata
                    metadata_path = filepath.with_suffix('.pkl')
                    if metadata_path.exists():
                        with open(metadata_path, 'rb') as f:
                            metadata = pickle.load(f)

                        self.id_to_strategy = metadata['id_to_strategy']
                        self.strategy_id_to_faiss_id = metadata['strategy_id_to_faiss_id']
                        self.next_faiss_id = metadata['next_faiss_id']
                        self.is_trained = metadata['is_trained']

                        logger.info(f"FAISS index loaded: {len(self.id_to_strategy)} strategies")
                    else:
                        logger.warning("FAISS index found but metadata missing")
                else:
                    logger.info("No existing FAISS index found, will create new")
            except Exception as e:
                logger.error(f"Failed to load FAISS index: {e}")

    @property
    def stats(self) -> dict[str, Any]:
        """Get index statistics"""
        return {
            'total_strategies': len(self.id_to_strategy),
            'is_trained': self.is_trained,
            'faiss_available': FAISS_AVAILABLE,
            'index_type': self.index_type,
            'dimension': self.dimension,
            'next_faiss_id': self.next_faiss_id
        }


class StrategySearchEngine:
    """High-level strategy search engine using FAISS"""

    def __init__(
        self,
        embedding_model=None,
        index_config: dict[str, Any] | None = None,
        cache_dir: Path | None = None
    ):
        self.embedding_model = embedding_model
        self.cache_dir = cache_dir or Path("cache/strategy_search")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Initialize FAISS index
        config = index_config or {}
        self.faiss_index = FAISSStrategyIndex(
            dimension=config.get('dimension', 768),
            index_type=config.get('index_type', 'IVF'),
            nlist=config.get('nlist', 100),
            use_gpu=config.get('use_gpu', False),
            similarity_threshold=config.get('similarity_threshold', 0.7)
        )

        # Strategy cache
        self.strategy_cache: dict[str, StrategyEntry] = {}
        self._last_save_time = time.time()
        self._save_interval = 300  # Save every 5 minutes

        logger.info("StrategySearchEngine initialized")

    async def initialize(self, embedding_model=None):
        """Initialize the search engine"""
        if embedding_model:
            self.embedding_model = embedding_model

        await self.faiss_index.initialize()

        # Try to load existing index
        index_path = self.cache_dir / "strategy_index"
        await self.faiss_index.load_index(index_path)

        logger.info("StrategySearchEngine initialized and ready")

    async def add_strategies_from_library(
        self,
        strategy_library: dict[str, Any],
        recompute_embeddings: bool = False
    ) -> int:
        """Add strategies from AutoDAN strategy library"""
        added_count = 0

        for strategy_name, strategy_data in strategy_library.items():
            try:
                # Create strategy entry
                strategy_entry = StrategyEntry(
                    strategy_id=self._generate_strategy_id(strategy_name),
                    name=strategy_name,
                    definition=strategy_data.get('Definition', ''),
                    examples=strategy_data.get('Example', []),
                    metadata={
                        'scores': strategy_data.get('Score', []),
                        'source': 'autodan_library'
                    },
                    tags=set(self._extract_tags(strategy_name, strategy_data))
                )

                # Get or compute embedding
                if not recompute_embeddings and 'Embeddings' in strategy_data:
                    embeddings = strategy_data['Embeddings']
                    if embeddings and len(embeddings) > 0:
                        strategy_entry.embedding = np.array(embeddings[0])

                if strategy_entry.embedding is None and self.embedding_model:
                    # Combine definition and examples for embedding
                    text_for_embedding = strategy_entry.definition
                    if strategy_entry.examples:
                        text_for_embedding += " " + " ".join(strategy_entry.examples[:2])

                    embedding = await self.faiss_index._get_text_embedding_async(
                        text_for_embedding, self.embedding_model
                    )
                    strategy_entry.embedding = embedding

                if strategy_entry.embedding is not None:
                    success = await self.faiss_index.add_strategy(strategy_entry)
                    if success:
                        self.strategy_cache[strategy_entry.strategy_id] = strategy_entry
                        added_count += 1

            except Exception as e:
                logger.error(f"Failed to add strategy {strategy_name}: {e}")

        logger.info(f"Added {added_count} strategies to search index")

        # Auto-save if enough time has passed
        if time.time() - self._last_save_time > self._save_interval:
            await self.save_index()

        return added_count

    async def search_strategies(
        self,
        query: str,
        k: int = 10,
        min_similarity: float | None = None,
        filter_tags: set[str] | None = None
    ) -> list[StrategySearchResult]:
        """Search for strategies using text query"""
        if not self.embedding_model:
            logger.warning("No embedding model available for search")
            return []

        results = await self.faiss_index.search_by_text(
            query, self.embedding_model, k * 2, min_similarity  # Over-fetch for filtering
        )

        # Apply tag filtering if specified
        if filter_tags:
            filtered_results = []
            for result in results:
                if filter_tags.intersection(result.strategy.tags):
                    filtered_results.append(result)
                if len(filtered_results) >= k:
                    break
            results = filtered_results
        else:
            results = results[:k]

        # Update usage counts
        for result in results:
            if result.strategy.strategy_id in self.strategy_cache:
                self.strategy_cache[result.strategy.strategy_id].usage_count += 1

        return results

    async def get_strategy_recommendations(
        self,
        context: str,
        previous_strategies: list[str] | None = None,
        k: int = 5
    ) -> list[StrategySearchResult]:
        """Get strategy recommendations based on context"""
        previous_strategies = previous_strategies or []

        # Search for strategies similar to context
        results = await self.search_strategies(context, k * 3)  # Over-fetch

        # Filter out previously used strategies
        filtered_results = []
        for result in results:
            if result.strategy.name not in previous_strategies:
                filtered_results.append(result)
            if len(filtered_results) >= k:
                break

        return filtered_results

    async def update_strategy_effectiveness(
        self,
        strategy_id: str,
        effectiveness_score: float
    ):
        """Update the effectiveness score for a strategy"""
        if strategy_id in self.strategy_cache:
            strategy = self.strategy_cache[strategy_id]
            strategy.effectiveness_scores.append(effectiveness_score)

            # Keep only recent scores
            if len(strategy.effectiveness_scores) > 100:
                strategy.effectiveness_scores = strategy.effectiveness_scores[-100:]

    async def get_strategy_clusters(self, n_clusters: int = 5) -> list[StrategyCluster]:
        """Get strategy clusters for analysis"""
        return await self.faiss_index.get_strategy_clusters(n_clusters)

    async def save_index(self):
        """Save the search index"""
        index_path = self.cache_dir / "strategy_index"
        await self.faiss_index.save_index(index_path)
        self._last_save_time = time.time()

    def _generate_strategy_id(self, strategy_name: str) -> str:
        """Generate a unique ID for a strategy"""
        return hashlib.md5(strategy_name.encode()).hexdigest()

    def _extract_tags(self, strategy_name: str, strategy_data: dict[str, Any]) -> list[str]:
        """Extract tags from strategy name and data"""
        tags = []

        # Extract from name
        name_lower = strategy_name.lower()
        if 'persona' in name_lower or 'role' in name_lower:
            tags.append('persona')
        if 'logic' in name_lower or 'reason' in name_lower:
            tags.append('logical')
        if 'emotion' in name_lower or 'feel' in name_lower:
            tags.append('emotional')
        if 'context' in name_lower or 'scenario' in name_lower:
            tags.append('contextual')
        if 'example' in name_lower or 'demonstrate' in name_lower:
            tags.append('example-based')

        # Extract from definition
        definition = strategy_data.get('Definition', '').lower()
        if 'persuasive' in definition or 'convince' in definition:
            tags.append('persuasive')
        if 'technical' in definition or 'expert' in definition:
            tags.append('technical')
        if 'creative' in definition or 'imaginative' in definition:
            tags.append('creative')

        return tags

    @property
    def stats(self) -> dict[str, Any]:
        """Get search engine statistics"""
        return {
            'total_strategies': len(self.strategy_cache),
            'embedding_model_available': self.embedding_model is not None,
            'faiss_stats': self.faiss_index.stats,
            'cache_dir': str(self.cache_dir)
        }


# Context manager for managed search engine lifecycle
@asynccontextmanager
async def managed_strategy_search_engine(embedding_model=None, **kwargs):
    """Context manager for automatic search engine lifecycle management"""
    engine = StrategySearchEngine(embedding_model=embedding_model, **kwargs)
    try:
        await engine.initialize(embedding_model)
        yield engine
    finally:
        await engine.save_index()
