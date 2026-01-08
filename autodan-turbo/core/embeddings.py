"""
AutoDAN-Turbo Embedding Module

Provides text embedding functionality for strategy retrieval.
Uses sentence-transformers for local embeddings with optional
integration with backend config manager for API-based embeddings.

Enhanced with:
- Optional integration with backend config manager
- Support config-driven model selection
- Fallback when backend config not available
- Maintains standalone functionality for AutoDAN-Turbo module
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)


class ConfigIntegration:
    """
    Optional integration with backend AI config manager.

    This class provides a bridge to the centralized embedding service
    when running as part of the full Chimera backend. When running
    standalone, it gracefully falls back to local embeddings.
    """

    _embedding_service = None
    _config_manager = None
    _initialized = False

    @classmethod
    def is_available(cls) -> bool:
        """Check if backend integration is available."""
        if cls._initialized:
            return cls._embedding_service is not None

        cls._initialized = True
        try:
            # Try to import backend embedding service
            from app.services.embedding_service import get_embedding_service
            cls._embedding_service = get_embedding_service()
            logger.info("Backend embedding service integration available")
            return True
        except ImportError:
            logger.debug(
                "Backend embedding service not available, "
                "using local sentence-transformers"
            )
            return False
        except Exception as e:
            logger.warning(f"Failed to initialize backend integration: {e}")
            return False

    @classmethod
    def get_embedding_service(cls):
        """Get the backend embedding service if available."""
        if cls.is_available():
            return cls._embedding_service
        return None

    @classmethod
    def get_config_manager(cls):
        """Get the backend config manager if available."""
        if cls._config_manager is not None:
            return cls._config_manager

        try:
            from app.core.service_registry import get_ai_config_manager
            cls._config_manager = get_ai_config_manager()
            return cls._config_manager
        except ImportError:
            return None
        except Exception as e:
            logger.debug(f"Config manager not available: {e}")
            return None

    @classmethod
    def get_embedding_provider(cls) -> str | None:
        """Get the configured embedding provider."""
        config_manager = cls.get_config_manager()
        if not config_manager:
            return None

        try:
            if not config_manager.is_loaded():
                return None

            config = config_manager.get_config()
            for name, provider in config.providers.items():
                if (
                    provider.enabled
                    and provider.capabilities.supports_embeddings
                ):
                    return name
            return None
        except Exception:
            return None


class EmbeddingModel:
    """
    Text embedding model for generating vector representations.

    Used for:
    - Embedding target LLM responses for strategy retrieval
    - Computing similarity between responses

    The paper uses text embeddings to index the strategy library,
    allowing retrieval of relevant strategies based on response
    similarity.

    Enhanced with:
    - Optional backend config integration for API-based embeddings
    - Config-driven model selection when backend is available
    - Automatic fallback to sentence-transformers for standalone use
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cuda",
        batch_size: int = 32,
        use_backend_if_available: bool = True,
        provider: str | None = None,
    ):
        """
        Initialize the embedding model.

        Args:
            model_name: Name of the sentence-transformer model
            device: Device to run on ('cuda', 'cpu', 'auto')
            batch_size: Batch size for encoding
            use_backend_if_available: Use backend embedding service if avail
            provider: Override provider for backend embeddings
        """
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.use_backend = use_backend_if_available
        self.provider_override = provider
        self._model = None
        self._dimension = None
        self._use_api = False

        # Check if we should use backend API embeddings
        if use_backend_if_available and ConfigIntegration.is_available():
            self._use_api = True
            self._init_api_mode()
        else:
            self._use_api = False

    def _init_api_mode(self):
        """Initialize for API-based embeddings."""
        try:
            embedding_service = ConfigIntegration.get_embedding_service()
            if embedding_service:
                provider = (
                    self.provider_override
                    or ConfigIntegration.get_embedding_provider()
                    or "openai"
                )
                self._dimension = embedding_service.get_embedding_dimensions(
                    provider
                )
                logger.info(
                    f"Using API embeddings via {provider}, "
                    f"dimension={self._dimension}"
                )
            else:
                self._use_api = False
        except Exception as e:
            logger.warning(f"Failed to init API mode, falling back: {e}")
            self._use_api = False

    def _load_model(self):
        """Lazy load the local model."""
        if self._model is None and not self._use_api:
            try:
                from sentence_transformers import SentenceTransformer

                logger.info(f"Loading embedding model: {self.model_name}")

                # Handle device selection
                if self.device == "auto":
                    import torch

                    device = "cuda" if torch.cuda.is_available() else "cpu"
                else:
                    device = self.device

                self._model = SentenceTransformer(
                    self.model_name, device=device
                )
                self._dimension = (
                    self._model.get_sentence_embedding_dimension()
                )

                logger.info(
                    f"Embedding model loaded. Dimension: {self._dimension}"
                )

            except ImportError:
                logger.error(
                    "sentence-transformers not installed. "
                    "Install with: pip install sentence-transformers"
                )
                raise
            except Exception as e:
                logger.error(f"Failed to load embedding model: {e}")
                raise

    @property
    def dimension(self) -> int:
        """Get the embedding dimension."""
        if self._dimension is not None:
            return self._dimension

        if self._use_api:
            self._init_api_mode()
        else:
            self._load_model()

        return self._dimension or 384  # Default fallback

    def encode(
        self,
        texts: str | list[str],
        normalize: bool = True,
        show_progress: bool = False
    ) -> np.ndarray:
        """
        Encode text(s) into embedding vectors.

        Args:
            texts: Single text or list of texts to encode
            normalize: Whether to L2-normalize embeddings
            show_progress: Whether to show progress bar (local only)

        Returns:
            Numpy array of shape (n_texts, dimension) or (dimension,)
            for single text
        """
        # Handle single text
        single_input = isinstance(texts, str)
        if single_input:
            texts = [texts]

        # Use API or local
        if self._use_api:
            embeddings = self._encode_via_api(texts, normalize)
        else:
            embeddings = self._encode_local(texts, normalize, show_progress)

        # Return single embedding if single input
        if single_input:
            return embeddings[0]

        return embeddings

    def _encode_via_api(
        self,
        texts: list[str],
        normalize: bool = True
    ) -> np.ndarray:
        """Encode texts using backend API embedding service."""
        try:
            import asyncio

            embedding_service = ConfigIntegration.get_embedding_service()
            if not embedding_service:
                logger.warning("API service unavailable, falling back")
                return self._encode_local(texts, normalize, False)

            provider = (
                self.provider_override
                or ConfigIntegration.get_embedding_provider()
            )

            # Run async in sync context
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # We're already in an async context
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    future = pool.submit(
                        asyncio.run,
                        embedding_service.generate_batch_embeddings(
                            texts,
                            provider=provider
                        )
                    )
                    embeddings_list = future.result()
            else:
                embeddings_list = loop.run_until_complete(
                    embedding_service.generate_batch_embeddings(
                        texts,
                        provider=provider
                    )
                )

            embeddings = np.array(embeddings_list, dtype=np.float32)

            if normalize:
                norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
                norms = np.where(norms == 0, 1, norms)
                embeddings = embeddings / norms

            return embeddings

        except Exception as e:
            logger.error(f"API embedding failed, falling back: {e}")
            return self._encode_local(texts, normalize, False)

    def _encode_local(
        self,
        texts: list[str],
        normalize: bool = True,
        show_progress: bool = False
    ) -> np.ndarray:
        """Encode texts using local sentence-transformers."""
        self._load_model()

        # Encode
        embeddings = self._model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=normalize,
            convert_to_numpy=True,
        )

        return embeddings

    def similarity(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray
    ) -> float:
        """
        Compute cosine similarity between two embeddings.

        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector

        Returns:
            Cosine similarity score (0-1 for normalized vectors)
        """
        # Ensure 1D arrays
        e1 = embedding1.flatten()
        e2 = embedding2.flatten()

        # Compute cosine similarity
        dot_product = np.dot(e1, e2)
        norm1 = np.linalg.norm(e1)
        norm2 = np.linalg.norm(e2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(dot_product / (norm1 * norm2))

    def batch_similarity(
        self, query_embedding: np.ndarray, corpus_embeddings: np.ndarray
    ) -> np.ndarray:
        """
        Compute similarity between a query and multiple corpus embeddings.

        Args:
            query_embedding: Query embedding vector (dimension,)
            corpus_embeddings: Corpus embeddings (n_corpus, dimension)

        Returns:
            Array of similarity scores (n_corpus,)
        """
        # Ensure query is 1D
        query = query_embedding.flatten()

        # Compute dot products (cosine similarity for normalized vectors)
        similarities = np.dot(corpus_embeddings, query)

        return similarities

    def find_most_similar(
        self,
        query_embedding: np.ndarray,
        corpus_embeddings: np.ndarray,
        top_k: int = 5
    ) -> list[tuple]:
        """
        Find the most similar embeddings in a corpus.

        Args:
            query_embedding: Query embedding vector
            corpus_embeddings: Corpus embeddings (n_corpus, dimension)
            top_k: Number of top results to return

        Returns:
            List of (index, similarity_score) tuples, sorted by similarity
        """
        similarities = self.batch_similarity(query_embedding, corpus_embeddings)

        # Get top-k indices
        if len(similarities) <= top_k:
            top_indices = np.argsort(similarities)[::-1]
        else:
            top_indices = np.argpartition(similarities, -top_k)[-top_k:]
            top_indices = top_indices[
                np.argsort(similarities[top_indices])[::-1]
            ]

        return [(int(idx), float(similarities[idx])) for idx in top_indices]

    def get_provider_info(self) -> dict:
        """Get information about the current embedding provider."""
        if self._use_api:
            provider = (
                self.provider_override
                or ConfigIntegration.get_embedding_provider()
            )
            return {
                "type": "api",
                "provider": provider,
                "dimension": self._dimension,
                "backend_available": True,
            }
        else:
            return {
                "type": "local",
                "model": self.model_name,
                "device": self.device,
                "dimension": self._dimension,
                "backend_available": ConfigIntegration.is_available(),
            }


class APIEmbeddingModel(EmbeddingModel):
    """
    Embedding model that always uses backend API embeddings.

    Falls back to local embeddings if backend is not available.
    """

    def __init__(
        self,
        provider: str | None = None,
        fallback_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "auto",
        batch_size: int = 32,
    ):
        """
        Initialize API embedding model.

        Args:
            provider: Provider to use (None = use config default)
            fallback_model: Local model to use if API unavailable
            device: Device for fallback model
            batch_size: Batch size for encoding
        """
        super().__init__(
            model_name=fallback_model,
            device=device,
            batch_size=batch_size,
            use_backend_if_available=True,
            provider=provider,
        )


class MockEmbeddingModel(EmbeddingModel):
    """
    Mock embedding model for testing without loading actual models.
    """

    def __init__(self, dimension: int = 384, **kwargs):
        """Initialize mock model with specified dimension."""
        super().__init__(**kwargs)
        self._dimension = dimension
        self._model = "mock"
        self._use_api = False

    def _load_model(self):
        """No-op for mock model."""
        pass

    def encode(
        self,
        texts: str | list[str],
        normalize: bool = True,
        _show_progress: bool = False
    ) -> np.ndarray:
        """Generate random embeddings for testing."""
        single_input = isinstance(texts, str)
        if single_input:
            texts = [texts]

        # Generate deterministic embeddings based on text hash
        embeddings = []
        for text in texts:
            # Use hash for reproducibility
            np.random.seed(hash(text) % (2**32))
            emb = np.random.randn(self._dimension).astype(np.float32)
            if normalize:
                emb = emb / np.linalg.norm(emb)
            embeddings.append(emb)

        embeddings = np.array(embeddings)

        if single_input:
            return embeddings[0]

        return embeddings


class ConfigAwareEmbeddingModel(EmbeddingModel):
    """
    Embedding model that adapts based on backend configuration.

    This model:
    - Uses API embeddings when backend config indicates embedding support
    - Falls back to local embeddings when config unavailable
    - Respects provider-specific model configurations
    """

    def __init__(
        self,
        preferred_provider: str | None = None,
        local_fallback: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "auto",
        batch_size: int = 32,
    ):
        """
        Initialize config-aware embedding model.

        Args:
            preferred_provider: Preferred API provider (None = auto-select)
            local_fallback: Local model for when API unavailable
            device: Device for local model
            batch_size: Batch size
        """
        # Determine provider from config if not specified
        provider = preferred_provider
        if not provider and ConfigIntegration.is_available():
            provider = ConfigIntegration.get_embedding_provider()

        super().__init__(
            model_name=local_fallback,
            device=device,
            batch_size=batch_size,
            use_backend_if_available=True,
            provider=provider,
        )

        logger.info(
            f"ConfigAwareEmbeddingModel initialized: "
            f"use_api={self._use_api}, "
            f"provider={provider or 'auto'}, "
            f"dimension={self._dimension}"
        )


def create_embedding_model(
    use_api: bool = True,
    provider: str | None = None,
    local_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    device: str = "auto",
) -> EmbeddingModel:
    """
    Factory function to create the appropriate embedding model.

    Args:
        use_api: Whether to prefer API embeddings
        provider: Specific provider to use
        local_model: Local model name for fallback
        device: Device for local model

    Returns:
        Configured EmbeddingModel instance
    """
    if use_api and ConfigIntegration.is_available():
        return ConfigAwareEmbeddingModel(
            preferred_provider=provider,
            local_fallback=local_model,
            device=device,
        )
    else:
        return EmbeddingModel(
            model_name=local_model,
            device=device,
            use_backend_if_available=False,
        )
