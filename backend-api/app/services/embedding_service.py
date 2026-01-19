"""
Embedding Service with Config-Driven Provider Selection

Provides centralized embedding generation with:
- Config-driven provider/model selection
- Embedding caching with TTL
- Batch processing with configurable batch sizes
- Cost estimation using config pricing
- Similarity computation utilities
- Fallback to embedding-capable providers
"""

import asyncio
import hashlib
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Embedding model configurations per provider
EMBEDDING_MODELS = {
    "openai": {
        "text-embedding-3-small": {
            "dimensions": 1536,
            "max_tokens": 8191,
            "price_per_1k_tokens": 0.00002,
        },
        "text-embedding-3-large": {
            "dimensions": 3072,
            "max_tokens": 8191,
            "price_per_1k_tokens": 0.00013,
        },
        "text-embedding-ada-002": {
            "dimensions": 1536,
            "max_tokens": 8191,
            "price_per_1k_tokens": 0.0001,
        },
        "default": "text-embedding-3-small",
    },
    "gemini": {
        "text-embedding-004": {
            "dimensions": 768,
            "max_tokens": 2048,
            "price_per_1k_tokens": 0.00001,
        },
        "embedding-001": {
            "dimensions": 768,
            "max_tokens": 2048,
            "price_per_1k_tokens": 0.00001,
        },
        "default": "text-embedding-004",
    },
    "qwen": {
        "text-embedding-v1": {
            "dimensions": 1536,
            "max_tokens": 2048,
            "price_per_1k_tokens": 0.0001,
        },
        "default": "text-embedding-v1",
    },
    "bigmodel": {
        "embedding-2": {
            "dimensions": 1024,
            "max_tokens": 512,
            "price_per_1k_tokens": 0.0005,
        },
        "default": "embedding-2",
    },
}


@dataclass
class CacheEntry:
    """Cache entry for embeddings."""

    embedding: list[float]
    created_at: float
    provider: str
    model: str


@dataclass
class EmbeddingCostEstimate:
    """Cost estimate for embedding operation."""

    provider: str
    model: str
    text_count: int
    estimated_tokens: int
    price_per_1k: float
    estimated_cost: float


@dataclass
class EmbeddingStats:
    """Statistics for embedding service."""

    total_requests: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    total_embeddings_generated: int = 0
    total_tokens_used: int = 0
    total_cost: float = 0.0
    errors: int = 0
    providers_used: dict[str, int] = field(default_factory=dict)


class EmbeddingService:
    """
    Config-driven embedding generation service.

    Features:
    - Config-driven provider selection for embeddings
    - Capability validation (check `embeddings` capability)
    - Cost estimation using config pricing
    - Embedding caching with configurable TTL
    - Batch processing with config-driven batch sizes
    - Fallback to embedding-capable providers

    Usage:
        from app.services.embedding_service import embedding_service

        # Generate single embedding
        embedding = await embedding_service.generate_embedding("Hello world")

        # Generate batch embeddings
        embeddings = await embedding_service.generate_batch_embeddings(["text1", "text2"])

        # Compute similarity
        similarity = await embedding_service.compute_similarity("text1", "text2")
    """

    def __init__(
        self,
        cache_ttl_seconds: int = 3600,
        default_batch_size: int = 100,
        enable_cache: bool = True,
    ):
        """
        Initialize the embedding service.

        Args:
            cache_ttl_seconds: TTL for cached embeddings (default: 1 hour)
            default_batch_size: Default batch size for batch operations
            enable_cache: Whether to enable embedding caching
        """
        self._config_manager = None
        self._embedding_cache: dict[str, CacheEntry] = {}
        self._cache_ttl = cache_ttl_seconds
        self._default_batch_size = default_batch_size
        self._enable_cache = enable_cache
        self._stats = EmbeddingStats()
        self._clients: dict[str, Any] = {}
        self._lock = asyncio.Lock()

        logger.info(
            f"EmbeddingService initialized: cache_ttl={cache_ttl_seconds}s, "
            f"batch_size={default_batch_size}, cache_enabled={enable_cache}"
        )

    # =========================================================================
    # Config Manager Access
    # =========================================================================

    def _get_config_manager(self):
        """Get the AI config manager lazily."""
        if self._config_manager is None:
            try:
                from app.core.service_registry import get_ai_config_manager

                self._config_manager = get_ai_config_manager()
            except Exception as e:
                logger.warning(f"Failed to get AI config manager: {e}")
                return None
        return self._config_manager

    def _get_embedding_provider(self) -> str | None:
        """
        Get best provider with embedding capability.

        Returns provider ID or None if no embedding-capable providers.
        """
        config_manager = self._get_config_manager()
        if not config_manager:
            logger.debug("No config manager available, using default 'openai'")
            return "openai"  # Default fallback

        try:
            if not config_manager.is_loaded():
                logger.warning("Config not loaded, using default 'openai'")
                return "openai"

            config = config_manager.get_config()

            # Find providers with embedding capability, sorted by priority
            embedding_providers = []
            for name, provider in config.providers.items():
                if provider.enabled and provider.capabilities.supports_embeddings:
                    embedding_providers.append((name, provider.priority))

            if not embedding_providers:
                logger.warning("No embedding-capable providers found in config")
                return None

            # Sort by priority (higher first) and return best
            embedding_providers.sort(key=lambda x: x[1], reverse=True)
            return embedding_providers[0][0]

        except Exception as e:
            logger.error(f"Error getting embedding provider: {e}")
            return "openai"  # Fallback

    def get_embedding_providers(self) -> list[str]:
        """
        Get list of providers supporting embeddings from config.

        Returns:
            List of provider IDs that support embeddings
        """
        config_manager = self._get_config_manager()
        if not config_manager:
            return list(EMBEDDING_MODELS.keys())

        try:
            if not config_manager.is_loaded():
                return list(EMBEDDING_MODELS.keys())

            config = config_manager.get_config()

            providers = []
            for name, provider in config.providers.items():
                if provider.enabled and provider.capabilities.supports_embeddings:
                    providers.append(name)

            return providers

        except Exception as e:
            logger.error(f"Error getting embedding providers: {e}")
            return list(EMBEDDING_MODELS.keys())

    def get_embedding_model(self, provider: str) -> str | None:
        """
        Get default embedding model for provider from config.

        Args:
            provider: Provider ID

        Returns:
            Default embedding model name or None
        """
        # Check static config first
        if provider in EMBEDDING_MODELS:
            return EMBEDDING_MODELS[provider].get("default")

        return None

    def get_embedding_dimensions(self, provider: str, model: str | None = None) -> int:
        """
        Get embedding dimensions for a provider/model combination.

        Args:
            provider: Provider ID
            model: Optional model name (uses default if not specified)

        Returns:
            Embedding dimensions
        """
        if provider not in EMBEDDING_MODELS:
            return 1536  # Common default

        provider_models = EMBEDDING_MODELS[provider]
        model_name = model or provider_models.get("default")

        if model_name and model_name in provider_models:
            return provider_models[model_name].get("dimensions", 1536)

        return 1536

    # =========================================================================
    # Core Embedding Methods
    # =========================================================================

    async def generate_embedding(
        self,
        text: str,
        provider: str | None = None,
        model: str | None = None,
        use_cache: bool | None = None,
    ) -> list[float]:
        """
        Generate embedding using config-driven provider selection.

        Args:
            text: Text to embed
            provider: Optional provider override
            model: Optional model override
            use_cache: Whether to use cache (defaults to service setting)

        Returns:
            Embedding vector as list of floats

        Raises:
            ValueError: If no embedding-capable providers available
            RuntimeError: If embedding generation fails
        """
        self._stats.total_requests += 1
        use_cache = use_cache if use_cache is not None else self._enable_cache

        # Resolve provider
        actual_provider = provider or self._get_embedding_provider()
        if not actual_provider:
            raise ValueError("No embedding-capable providers available")

        # Resolve model
        actual_model = model or self.get_embedding_model(actual_provider)

        # Check cache
        if use_cache:
            cache_key = self._get_cache_key(text, actual_provider, actual_model)
            cached = self._get_from_cache(cache_key)
            if cached is not None:
                self._stats.cache_hits += 1
                return cached
            self._stats.cache_misses += 1

        # Generate embedding
        try:
            embedding = await self._generate_embedding_impl(text, actual_provider, actual_model)

            # Update stats
            self._stats.total_embeddings_generated += 1
            self._stats.providers_used[actual_provider] = (
                self._stats.providers_used.get(actual_provider, 0) + 1
            )

            # Estimate and track cost
            tokens = len(text.split()) + 1  # Rough estimate
            self._stats.total_tokens_used += tokens
            if actual_provider in EMBEDDING_MODELS:
                model_info = EMBEDDING_MODELS[actual_provider].get(
                    actual_model or EMBEDDING_MODELS[actual_provider].get("default", ""), {}
                )
                price = model_info.get("price_per_1k_tokens", 0)
                self._stats.total_cost += (tokens / 1000) * price

            # Cache result
            if use_cache:
                self._add_to_cache(cache_key, embedding, actual_provider, actual_model)

            return embedding

        except Exception as e:
            self._stats.errors += 1
            logger.error(f"Embedding generation failed: {e}")

            # Try fallback provider
            fallback = self._get_fallback_provider(actual_provider)
            if fallback:
                logger.info(f"Trying fallback provider: {fallback}")
                return await self.generate_embedding(text, provider=fallback, model=None)

            raise RuntimeError(f"Embedding generation failed: {e}")

    async def generate_batch_embeddings(
        self,
        texts: list[str],
        provider: str | None = None,
        batch_size: int | None = None,
        model: str | None = None,
        use_cache: bool | None = None,
    ) -> list[list[float]]:
        """
        Batch embedding generation with config-driven batching.

        Args:
            texts: List of texts to embed
            provider: Optional provider override
            batch_size: Batch size (defaults to service setting)
            model: Optional model override
            use_cache: Whether to use cache

        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        batch_size = batch_size or self._default_batch_size
        actual_provider = provider or self._get_embedding_provider()
        actual_model = model or self.get_embedding_model(actual_provider)
        use_cache = use_cache if use_cache is not None else self._enable_cache

        results = [None] * len(texts)
        texts_to_generate = []
        indices_to_generate = []

        # Check cache first
        if use_cache:
            for i, text in enumerate(texts):
                cache_key = self._get_cache_key(text, actual_provider, actual_model)
                cached = self._get_from_cache(cache_key)
                if cached is not None:
                    results[i] = cached
                    self._stats.cache_hits += 1
                else:
                    texts_to_generate.append(text)
                    indices_to_generate.append(i)
                    self._stats.cache_misses += 1
        else:
            texts_to_generate = texts
            indices_to_generate = list(range(len(texts)))

        # Generate in batches
        for batch_start in range(0, len(texts_to_generate), batch_size):
            batch_end = min(batch_start + batch_size, len(texts_to_generate))
            batch_texts = texts_to_generate[batch_start:batch_end]
            batch_indices = indices_to_generate[batch_start:batch_end]

            try:
                batch_embeddings = await self._generate_batch_impl(
                    batch_texts, actual_provider, actual_model
                )

                for i, embedding in zip(batch_indices, batch_embeddings, strict=False):
                    results[i] = embedding

                    if use_cache:
                        cache_key = self._get_cache_key(
                            texts_to_generate[indices_to_generate.index(i)],
                            actual_provider,
                            actual_model,
                        )
                        self._add_to_cache(cache_key, embedding, actual_provider, actual_model)

                self._stats.total_embeddings_generated += len(batch_embeddings)

            except Exception as e:
                logger.error(f"Batch embedding failed: {e}")
                self._stats.errors += 1

                # Fall back to individual generation
                for idx, text in zip(batch_indices, batch_texts, strict=False):
                    try:
                        results[idx] = await self.generate_embedding(
                            text, provider=actual_provider, model=actual_model
                        )
                    except Exception as inner_e:
                        logger.error(f"Individual fallback failed: {inner_e}")
                        # Return zero vector as last resort
                        dims = self.get_embedding_dimensions(actual_provider, actual_model)
                        results[idx] = [0.0] * dims

        return results

    async def compute_similarity(
        self,
        text1: str,
        text2: str,
        provider: str | None = None,
    ) -> float:
        """
        Compute cosine similarity between two texts.

        Args:
            text1: First text
            text2: Second text
            provider: Optional provider override

        Returns:
            Cosine similarity score (-1 to 1)
        """
        embedding1 = await self.generate_embedding(text1, provider=provider)
        embedding2 = await self.generate_embedding(text2, provider=provider)

        return self._cosine_similarity(embedding1, embedding2)

    # =========================================================================
    # Cost Estimation
    # =========================================================================

    def estimate_embedding_cost(
        self,
        texts: list[str],
        provider: str,
        model: str | None = None,
    ) -> EmbeddingCostEstimate:
        """
        Estimate cost for embedding texts using config pricing.

        Args:
            texts: List of texts to estimate for
            provider: Provider to use
            model: Optional model override

        Returns:
            EmbeddingCostEstimate with cost details
        """
        actual_model = model or self.get_embedding_model(provider)

        # Estimate tokens (rough: words + 1 per text)
        estimated_tokens = sum(len(text.split()) + 1 for text in texts)

        # Get price
        price_per_1k = 0.0
        if provider in EMBEDDING_MODELS:
            model_info = EMBEDDING_MODELS[provider].get(actual_model or "", {})
            price_per_1k = model_info.get("price_per_1k_tokens", 0.0)

        estimated_cost = (estimated_tokens / 1000) * price_per_1k

        return EmbeddingCostEstimate(
            provider=provider,
            model=actual_model or "unknown",
            text_count=len(texts),
            estimated_tokens=estimated_tokens,
            price_per_1k=price_per_1k,
            estimated_cost=estimated_cost,
        )

    # =========================================================================
    # Provider Implementation
    # =========================================================================

    async def _generate_embedding_impl(
        self,
        text: str,
        provider: str,
        model: str | None,
    ) -> list[float]:
        """Generate embedding using the specified provider."""
        if provider == "openai":
            return await self._openai_embedding(text, model)
        elif provider in ("gemini", "google"):
            return await self._gemini_embedding(text, model)
        elif provider == "qwen":
            return await self._qwen_embedding(text, model)
        elif provider == "bigmodel":
            return await self._bigmodel_embedding(text, model)
        else:
            # Try OpenAI-compatible endpoint
            return await self._openai_compatible_embedding(text, model, provider)

    async def _generate_batch_impl(
        self,
        texts: list[str],
        provider: str,
        model: str | None,
    ) -> list[list[float]]:
        """Generate batch embeddings using the specified provider."""
        if provider == "openai":
            return await self._openai_batch_embedding(texts, model)
        elif provider in ("gemini", "google"):
            # Gemini doesn't support batch, generate individually
            return [await self._gemini_embedding(text, model) for text in texts]
        else:
            # Generate individually as fallback
            return [await self._generate_embedding_impl(text, provider, model) for text in texts]

    async def _openai_embedding(self, text: str, model: str | None) -> list[float]:
        """Generate embedding using OpenAI."""
        model = model or "text-embedding-3-small"

        try:
            import openai

            client = openai.AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

            response = await client.embeddings.create(
                input=text,
                model=model,
            )

            return response.data[0].embedding

        except ImportError:
            logger.error("OpenAI library not installed")
            raise
        except Exception as e:
            logger.error(f"OpenAI embedding failed: {e}")
            raise

    async def _openai_batch_embedding(
        self,
        texts: list[str],
        model: str | None,
    ) -> list[list[float]]:
        """Generate batch embeddings using OpenAI."""
        model = model or "text-embedding-3-small"

        try:
            import openai

            client = openai.AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

            response = await client.embeddings.create(
                input=texts,
                model=model,
            )

            # Sort by index to maintain order
            sorted_data = sorted(response.data, key=lambda x: x.index)
            return [item.embedding for item in sorted_data]

        except ImportError:
            logger.error("OpenAI library not installed")
            raise
        except Exception as e:
            logger.error(f"OpenAI batch embedding failed: {e}")
            raise

    async def _gemini_embedding(self, text: str, model: str | None) -> list[float]:
        """Generate embedding using Gemini."""
        model = model or "text-embedding-004"

        try:
            import google.generativeai as genai

            genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

            result = await asyncio.to_thread(
                genai.embed_content,
                model=f"models/{model}",
                content=text,
                task_type="retrieval_document",
            )

            return result["embedding"]

        except ImportError:
            logger.error("Google Generative AI library not installed")
            raise
        except Exception as e:
            logger.error(f"Gemini embedding failed: {e}")
            raise

    async def _qwen_embedding(self, text: str, model: str | None) -> list[float]:
        """Generate embedding using Qwen/DashScope."""
        model = model or "text-embedding-v1"

        try:
            import dashscope
            from dashscope import TextEmbedding

            dashscope.api_key = os.getenv("QWEN_API_KEY")

            response = await asyncio.to_thread(
                TextEmbedding.call,
                model=model,
                input=text,
            )

            if response.status_code == 200:
                return response.output["embeddings"][0]["embedding"]
            else:
                raise RuntimeError(f"Qwen embedding failed: {response.message}")

        except ImportError:
            logger.error("DashScope library not installed")
            raise
        except Exception as e:
            logger.error(f"Qwen embedding failed: {e}")
            raise

    async def _bigmodel_embedding(self, text: str, model: str | None) -> list[float]:
        """Generate embedding using ZhiPu BigModel."""
        model = model or "embedding-2"

        try:
            import zhipuai

            client = zhipuai.ZhipuAI(api_key=os.getenv("BIGMODEL_API_KEY"))

            response = await asyncio.to_thread(
                client.embeddings.create,
                model=model,
                input=text,
            )

            return response.data[0].embedding

        except ImportError:
            logger.error("ZhipuAI library not installed")
            raise
        except Exception as e:
            logger.error(f"BigModel embedding failed: {e}")
            raise

    async def _openai_compatible_embedding(
        self,
        text: str,
        model: str | None,
        provider: str,
    ) -> list[float]:
        """Generate embedding using OpenAI-compatible endpoint."""
        config_manager = self._get_config_manager()
        if not config_manager:
            raise RuntimeError(f"Cannot get config for provider {provider}")

        try:
            provider_config = config_manager.get_provider(provider)
            if not provider_config:
                raise ValueError(f"Provider {provider} not found in config")

            import openai

            client = openai.AsyncOpenAI(
                api_key=os.getenv(provider_config.api.key_env_var),
                base_url=provider_config.api.base_url,
            )

            response = await client.embeddings.create(
                input=text,
                model=model or "default",
            )

            return response.data[0].embedding

        except Exception as e:
            logger.error(f"OpenAI-compatible embedding failed for {provider}: {e}")
            raise

    # =========================================================================
    # Cache Management
    # =========================================================================

    def _get_cache_key(self, text: str, provider: str, model: str | None) -> str:
        """Generate cache key for embedding."""
        content = f"{provider}:{model or 'default'}:{text}"
        return hashlib.sha256(content.encode()).hexdigest()

    def _get_from_cache(self, key: str) -> list[float] | None:
        """Get embedding from cache if not expired."""
        if key not in self._embedding_cache:
            return None

        entry = self._embedding_cache[key]
        if time.time() - entry.created_at > self._cache_ttl:
            # Expired
            del self._embedding_cache[key]
            return None

        return entry.embedding

    def _add_to_cache(
        self,
        key: str,
        embedding: list[float],
        provider: str,
        model: str | None,
    ) -> None:
        """Add embedding to cache."""
        self._embedding_cache[key] = CacheEntry(
            embedding=embedding,
            created_at=time.time(),
            provider=provider,
            model=model or "default",
        )

    def clear_cache(self) -> int:
        """Clear all cached embeddings. Returns count of cleared entries."""
        count = len(self._embedding_cache)
        self._embedding_cache.clear()
        logger.info(f"Cleared {count} cached embeddings")
        return count

    def cleanup_expired_cache(self) -> int:
        """Remove expired cache entries. Returns count of removed entries."""
        current_time = time.time()
        expired_keys = [
            key
            for key, entry in self._embedding_cache.items()
            if current_time - entry.created_at > self._cache_ttl
        ]

        for key in expired_keys:
            del self._embedding_cache[key]

        if expired_keys:
            logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")

        return len(expired_keys)

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def _get_fallback_provider(self, failed_provider: str) -> str | None:
        """Get fallback provider when primary fails."""
        providers = self.get_embedding_providers()
        for provider in providers:
            if provider != failed_provider:
                return provider
        return None

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

    @staticmethod
    def batch_cosine_similarity(
        query: list[float],
        corpus: list[list[float]],
    ) -> list[float]:
        """Compute cosine similarity between query and multiple corpus vectors."""
        if not query or not corpus:
            return []

        query_arr = np.array(query)
        corpus_arr = np.array(corpus)

        # Normalize
        query_norm = query_arr / (np.linalg.norm(query_arr) + 1e-10)
        corpus_norms = corpus_arr / (np.linalg.norm(corpus_arr, axis=1, keepdims=True) + 1e-10)

        # Compute similarities
        similarities = np.dot(corpus_norms, query_norm)

        return similarities.tolist()

    def get_stats(self) -> dict:
        """Get service statistics."""
        return {
            "total_requests": self._stats.total_requests,
            "cache_hits": self._stats.cache_hits,
            "cache_misses": self._stats.cache_misses,
            "cache_hit_rate": (
                self._stats.cache_hits / self._stats.total_requests
                if self._stats.total_requests > 0
                else 0.0
            ),
            "total_embeddings_generated": self._stats.total_embeddings_generated,
            "total_tokens_used": self._stats.total_tokens_used,
            "total_cost": self._stats.total_cost,
            "errors": self._stats.errors,
            "providers_used": self._stats.providers_used,
            "cache_size": len(self._embedding_cache),
            "cache_ttl_seconds": self._cache_ttl,
        }

    def reset_stats(self) -> None:
        """Reset service statistics."""
        self._stats = EmbeddingStats()
        logger.info("Embedding service stats reset")


# Global singleton instance
_embedding_service: EmbeddingService | None = None


def get_embedding_service() -> EmbeddingService:
    """Get the global embedding service instance."""
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
    return _embedding_service


# Convenience alias
embedding_service = get_embedding_service()
