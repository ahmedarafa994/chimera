import asyncio
import logging
import sys
import time
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import numpy as np

# Add backend-api to path
sys.path.append(str(Path(__file__).parent.parent / "backend-api"))

from app.engines.autodan_turbo.strategy_extractor import StrategyExtractor
from app.engines.autodan_turbo.strategy_library import (JailbreakStrategy,
                                                        StrategyLibrary,
                                                        StrategyMetadata,
                                                        StrategySource)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class TestPhase2Optimizations(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path("test_strategies")
        self.test_dir.mkdir(exist_ok=True)
        self.embedding_model = MagicMock()
        # Return random vectors to ensure they are not collinear
        # We need distinct directions for cosine similarity < 1.0
        self.embedding_model.encode.side_effect = lambda x: np.random.rand(768).tolist()

    def tearDown(self):
        # Cleanup
        for f in self.test_dir.glob("*.yaml"):
            f.unlink()
        self.test_dir.rmdir()

    def test_faiss_integration(self):
        logger.info("Testing FAISS Integration...")
        library = StrategyLibrary(storage_dir=self.test_dir, embedding_model=self.embedding_model)

        # Check if FAISS is available
        try:
            import faiss  # noqa: F401

            has_faiss = True
        except ImportError:
            has_faiss = False

        logger.info(f"FAISS Available: {has_faiss}")

        # Add strategies
        for i in range(10):
            strategy = JailbreakStrategy(
                id=f"strat_{i}",
                name=f"Strategy {i}",
                description=f"Description {i}",
                template=f"Template {{payload}} {i}",
                tags=["test"],
                metadata=StrategyMetadata(source=StrategySource.SEED),
            )
            library.add_strategy(strategy)

        # Verify Index
        if has_faiss:
            self.assertIsNotNone(library.index, "FAISS index should be initialized")
            self.assertEqual(library.index.ntotal, 10, "FAISS index should have 10 vectors")
            logger.info("FAISS index verified.")
        else:
            logger.warning("Skipping FAISS verification (faiss-cpu not installed)")

        # Verify Search
        results = library.search("query", top_k=3)
        self.assertEqual(len(results), 3, "Search should return 3 results")
        logger.info("Search verified.")

    def test_async_batch_extraction(self):
        logger.info("Testing Async Batch Extraction...")

        async def run_test():
            # Mock LLM Client
            llm_client = MagicMock()

            # Simulate latency
            async def mock_generate(_prompt):
                await asyncio.sleep(0.5)  # 500ms delay
                return "NAME: Test\nDESCRIPTION: Desc\nTEMPLATE: T {payload}\nTAGS: tag"

            llm_client.chat = mock_generate

            library = MagicMock()
            library.add_strategy.return_value = (True, "id")

            extractor = StrategyExtractor(llm_client=llm_client, library=library)
            # Mock novelty check to always pass
            extractor._check_novelty = AsyncMock(return_value=True)

            attacks = [("p", "r", 10.0)] * 5  # 5 attacks

            start_time = time.time()
            strategies = await extractor.batch_extract(attacks)
            end_time = time.time()
            duration = end_time - start_time

            logger.info(f"Batch extraction took {duration:.2f}s for 5 tasks (0.5s each)")

            # If sequential, it would take ~2.5s. If parallel, ~0.5s + overhead.
            self.assertLess(duration, 1.5, "Extraction should be parallelized")
            self.assertEqual(len(strategies), 5, "Should extract 5 strategies")

        asyncio.run(run_test())
        logger.info("Async batch extraction verified.")


if __name__ == "__main__":
    unittest.main()
