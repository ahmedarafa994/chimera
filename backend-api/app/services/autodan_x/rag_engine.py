"""AutoDAN-X RAG Engine - Retrieval-Augmented Generation for adversarial prompt engineering.

This module provides:
- FAISS-based vector store for knowledge base
- Semantic search using embeddings
- Context retrieval and augmentation
- Knowledge base management
"""

import asyncio
import hashlib
import logging
import os
import time
from pathlib import Path

import numpy as np
import yaml

from app.core.config import settings

from .models import KnowledgeEntry, RAGContext, RAGStats

logger = logging.getLogger(__name__)


class RAGEngine:
    """RAG Engine for AutoDAN-X knowledge base retrieval.

    Uses FAISS for efficient similarity search and GeminiEmbeddingModel
    for generating embeddings.
    """

    def __init__(
        self,
        knowledge_base_path: str | None = None,
        index_path: str | None = None,
        embedding_dimension: int = 768,
    ) -> None:
        """Initialize the RAG Engine.

        Args:
            knowledge_base_path: Path to knowledge base YAML files
            index_path: Path to store/load FAISS index
            embedding_dimension: Dimension of embedding vectors

        """
        self.knowledge_base_path = Path(
            knowledge_base_path
            or os.path.join(os.path.dirname(__file__), "data", "knowledge_base"),
        )
        self.index_path = Path(
            index_path or os.path.join(os.path.dirname(__file__), "data", "embeddings"),
        )
        self.embedding_dimension = embedding_dimension

        # Storage
        self.entries: dict[str, KnowledgeEntry] = {}
        self.embeddings: dict[str, np.ndarray] = {}
        self.index = None
        self.id_to_index: dict[str, int] = {}
        self.index_to_id: dict[int, str] = {}

        # Embedding model (lazy loaded)
        self._embedding_model = None

        # Stats
        self._last_updated = None
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the RAG engine by loading knowledge base and building index."""
        if self._initialized:
            return

        logger.info("Initializing RAG Engine...")

        # Load knowledge base
        await self._load_knowledge_base()

        # Try to load existing index or build new one
        if not await self._load_index():
            await self._build_index()

        self._initialized = True
        logger.info(f"RAG Engine initialized with {len(self.entries)} entries")

    def _get_embedding_model(self):
        """Get or create the embedding model."""
        if self._embedding_model is None:
            try:
                from app.services.autodan.llm import GeminiEmbeddingModel

                self._embedding_model = GeminiEmbeddingModel(
                    model_name="models/text-embedding-004",
                    api_key=settings.GOOGLE_API_KEY,
                )
                logger.info("Initialized GeminiEmbeddingModel for RAG")
            except Exception as e:
                logger.exception(f"Failed to initialize embedding model: {e}")
                msg = f"RAG Engine requires embedding model: {e}"
                raise RuntimeError(msg)
        return self._embedding_model

    async def _load_knowledge_base(self) -> None:
        """Load all knowledge base YAML files."""
        if not self.knowledge_base_path.exists():
            logger.warning(f"Knowledge base path does not exist: {self.knowledge_base_path}")
            return

        for yaml_file in self.knowledge_base_path.glob("*.yaml"):
            try:
                await self._load_yaml_file(yaml_file)
            except Exception as e:
                logger.exception(f"Error loading {yaml_file}: {e}")

    async def _load_yaml_file(self, file_path: Path) -> None:
        """Load a single YAML file into the knowledge base."""
        with open(file_path, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if not data:
            return

        # Handle different YAML structures
        entries_data = []

        if "techniques" in data:
            entries_data.extend(data["techniques"])
        if "personas" in data:
            entries_data.extend(data["personas"])
        if "architectures" in data:
            entries_data.extend(data["architectures"])
        if "exploits" in data:
            entries_data.extend(data["exploits"])
        if "patterns" in data:
            entries_data.extend(data["patterns"])

        # Also check for flat list
        if isinstance(data, list):
            entries_data.extend(data)

        for entry_data in entries_data:
            entry = self._parse_entry(entry_data, file_path.stem)
            if entry:
                self.entries[entry.id] = entry

        logger.info(f"Loaded {len(entries_data)} entries from {file_path.name}")

    def _parse_entry(self, data: dict, default_category: str) -> KnowledgeEntry | None:
        """Parse a dictionary into a KnowledgeEntry."""
        try:
            entry_id = (
                data.get("id")
                or hashlib.md5(
                    str(data.get("title", "") + data.get("content", "")).encode(),
                ).hexdigest()[:12]
            )

            # Build content from various fields
            content_parts = []
            if data.get("title"):
                content_parts.append(f"# {data['title']}")
            if data.get("content"):
                content_parts.append(data["content"])
            if data.get("template"):
                content_parts.append(f"\nTemplate:\n{data['template']}")
            if data.get("jargon_injection"):
                content_parts.append(f"\nJargon: {', '.join(data['jargon_injection'])}")

            content = "\n".join(content_parts)

            return KnowledgeEntry(
                id=entry_id,
                category=data.get("category", default_category),
                title=data.get("title") or data.get("name", "Untitled"),
                content=content,
                tags=data.get("tags", []),
                effectiveness_rating=float(data.get("effectiveness_rating", 5.0)),
                success_count=int(data.get("success_count", 0)),
                failure_count=int(data.get("failure_count", 0)),
            )
        except Exception as e:
            logger.warning(f"Failed to parse entry: {e}")
            return None

    async def _build_index(self) -> None:
        """Build FAISS index from knowledge base entries."""
        if not self.entries:
            logger.warning("No entries to index")
            return

        logger.info(f"Building FAISS index for {len(self.entries)} entries...")

        try:
            import faiss
        except ImportError:
            logger.exception("FAISS not installed. Install with: pip install faiss-cpu")
            return

        # Get embeddings for all entries
        embedding_model = self._get_embedding_model()

        entry_ids = list(self.entries.keys())
        texts = [self.entries[eid].content for eid in entry_ids]

        # Generate embeddings in batches
        batch_size = 10
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            try:
                # Use the embedding model's embed method
                batch_embeddings = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda b=batch_texts: [embedding_model.embed(text) for text in b],
                )
                all_embeddings.extend(batch_embeddings)
            except Exception as e:
                logger.exception(f"Error generating embeddings for batch {i}: {e}")
                # Use zero vectors as fallback
                all_embeddings.extend([np.zeros(self.embedding_dimension) for _ in batch_texts])

        # Convert to numpy array
        embeddings_array = np.array(all_embeddings, dtype=np.float32)

        # Normalize for cosine similarity
        faiss.normalize_L2(embeddings_array)

        # Build index
        self.index = faiss.IndexFlatIP(self.embedding_dimension)  # Inner product for cosine sim
        self.index.add(embeddings_array)

        # Build ID mappings
        for i, entry_id in enumerate(entry_ids):
            self.id_to_index[entry_id] = i
            self.index_to_id[i] = entry_id
            self.embeddings[entry_id] = embeddings_array[i]

        # Save index
        await self._save_index()

        self._last_updated = time.time()
        logger.info(f"FAISS index built with {self.index.ntotal} vectors")

    async def _save_index(self) -> None:
        """Save FAISS index to disk."""
        if self.index is None:
            return

        try:
            import faiss

            self.index_path.mkdir(parents=True, exist_ok=True)

            # Save FAISS index
            index_file = self.index_path / "faiss.index"
            faiss.write_index(self.index, str(index_file))

            # Save ID mappings
            mappings = {
                "id_to_index": self.id_to_index,
                "index_to_id": {str(k): v for k, v in self.index_to_id.items()},
            }
            mappings_file = self.index_path / "mappings.yaml"
            with open(mappings_file, "w") as f:
                yaml.dump(mappings, f)

            logger.info(f"Saved FAISS index to {self.index_path}")
        except Exception as e:
            logger.exception(f"Failed to save index: {e}")

    async def _load_index(self) -> bool:
        """Load FAISS index from disk."""
        try:
            import faiss

            index_file = self.index_path / "faiss.index"
            mappings_file = self.index_path / "mappings.yaml"

            if not index_file.exists() or not mappings_file.exists():
                return False

            # Load FAISS index
            self.index = faiss.read_index(str(index_file))

            # Load ID mappings
            with open(mappings_file) as f:
                mappings = yaml.safe_load(f)

            self.id_to_index = mappings["id_to_index"]
            self.index_to_id = {int(k): v for k, v in mappings["index_to_id"].items()}

            logger.info(f"Loaded FAISS index with {self.index.ntotal} vectors")
            return True
        except Exception as e:
            logger.warning(f"Failed to load index: {e}")
            return False

    async def search(
        self,
        query: str,
        top_k: int = 5,
        min_relevance: float = 0.0,
        category_filter: str | None = None,
    ) -> RAGContext:
        """Search the knowledge base for relevant entries.

        Args:
            query: Search query
            top_k: Number of results to return
            min_relevance: Minimum relevance score (0-1)
            category_filter: Optional category to filter by

        Returns:
            RAGContext with retrieved entries and scores

        """
        if not self._initialized:
            await self.initialize()

        start_time = time.time()

        if self.index is None or self.index.ntotal == 0:
            logger.warning("Index is empty, returning empty context")
            return RAGContext(query=query)

        try:
            # Get query embedding
            embedding_model = self._get_embedding_model()
            query_embedding = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: embedding_model.embed(query),
            )
            query_vector = np.array([query_embedding], dtype=np.float32)

            # Normalize for cosine similarity
            import faiss

            faiss.normalize_L2(query_vector)

            # Search
            scores, indices = self.index.search(query_vector, min(top_k * 2, self.index.ntotal))

            # Filter and collect results
            entries = []
            relevance_scores = []

            for score, idx in zip(scores[0], indices[0], strict=False):
                if idx < 0:  # FAISS returns -1 for empty slots
                    continue

                entry_id = self.index_to_id.get(int(idx))
                if not entry_id or entry_id not in self.entries:
                    continue

                entry = self.entries[entry_id]

                # Apply filters
                if category_filter and entry.category != category_filter:
                    continue

                # Convert score to 0-1 range (inner product of normalized vectors)
                relevance = float(max(0, min(1, (score + 1) / 2)))

                if relevance < min_relevance:
                    continue

                entries.append(entry)
                relevance_scores.append(relevance)

                if len(entries) >= top_k:
                    break

            # Build combined context
            combined_context = self._build_combined_context(entries, relevance_scores)

            retrieval_time = (time.time() - start_time) * 1000

            return RAGContext(
                entries=entries,
                relevance_scores=relevance_scores,
                combined_context=combined_context,
                total_tokens=len(combined_context.split()),  # Rough estimate
                query=query,
                retrieval_time_ms=retrieval_time,
            )

        except Exception as e:
            logger.exception(f"Search error: {e}")
            return RAGContext(query=query)

    def _build_combined_context(
        self,
        entries: list[KnowledgeEntry],
        scores: list[float],
    ) -> str:
        """Build combined context string from retrieved entries."""
        if not entries:
            return ""

        context_parts = ["[RETRIEVED KNOWLEDGE BASE CONTEXT]", ""]

        for i, (entry, score) in enumerate(zip(entries, scores, strict=False), 1):
            context_parts.append(f"--- Context {i} (Relevance: {score:.2f}) ---")
            context_parts.append(f"Category: {entry.category}")
            context_parts.append(f"Title: {entry.title}")
            context_parts.append(f"Effectiveness: {entry.effectiveness_rating}/10")
            context_parts.append("")
            context_parts.append(entry.content)
            context_parts.append("")

        context_parts.append("[END RETRIEVED CONTEXT]")

        return "\n".join(context_parts)

    async def add_entry(self, entry: KnowledgeEntry) -> bool:
        """Add a new entry to the knowledge base.

        Args:
            entry: KnowledgeEntry to add

        Returns:
            True if successful

        """
        if not self._initialized:
            await self.initialize()

        try:
            # Generate embedding
            embedding_model = self._get_embedding_model()
            embedding = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: embedding_model.embed(entry.content),
            )
            embedding_vector = np.array([embedding], dtype=np.float32)

            # Normalize
            import faiss

            faiss.normalize_L2(embedding_vector)

            # Add to index
            if self.index is None:
                self.index = faiss.IndexFlatIP(self.embedding_dimension)

            idx = self.index.ntotal
            self.index.add(embedding_vector)

            # Update mappings
            self.entries[entry.id] = entry
            self.embeddings[entry.id] = embedding_vector[0]
            self.id_to_index[entry.id] = idx
            self.index_to_id[idx] = entry.id

            # Save updated index
            await self._save_index()

            logger.info(f"Added entry {entry.id} to knowledge base")
            return True

        except Exception as e:
            logger.exception(f"Failed to add entry: {e}")
            return False

    def get_stats(self) -> RAGStats:
        """Get statistics about the knowledge base."""
        entries_by_category: dict[str, int] = {}
        for entry in self.entries.values():
            entries_by_category[entry.category] = entries_by_category.get(entry.category, 0) + 1

        # Estimate index size
        index_size_mb = 0.0
        if self.index is not None:
            # Rough estimate: 4 bytes per float * dimension * num_vectors
            index_size_mb = (4 * self.embedding_dimension * self.index.ntotal) / (1024 * 1024)

        return RAGStats(
            total_entries=len(self.entries),
            entries_by_category=entries_by_category,
            index_size_mb=index_size_mb,
            embedding_dimension=self.embedding_dimension,
            last_updated=self._last_updated,
        )

    async def rebuild_index(self) -> None:
        """Force rebuild of the FAISS index."""
        self.index = None
        self.id_to_index.clear()
        self.index_to_id.clear()
        self.embeddings.clear()
        await self._build_index()


# Singleton instance
_rag_engine: RAGEngine | None = None


def get_rag_engine() -> RAGEngine:
    """Get the singleton RAG engine instance."""
    global _rag_engine
    if _rag_engine is None:
        _rag_engine = RAGEngine()
    return _rag_engine


async def initialize_rag_engine() -> RAGEngine:
    """Initialize and return the RAG engine."""
    engine = get_rag_engine()
    await engine.initialize()
    return engine
