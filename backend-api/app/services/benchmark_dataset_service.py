"""
Benchmark Dataset Service

This service manages research benchmark datasets for LLM safety evaluation,
including the Do-Not-Answer dataset from the paper "Do-Not-Answer: A Dataset
for Evaluating Safeguards in LLMs" (arxiv:2308.13387).

The service provides:
- Loading benchmark datasets from JSON files
- Filtering prompts by risk area, harm type, severity
- Generating test batches for jailbreak evaluation
- Integrating with the existing jailbreak database
"""

import json
import logging
import random
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel

logger = logging.getLogger(__name__)

# Default data directory for benchmarks
BENCHMARKS_DIR = Path(__file__).parent.parent.parent.parent / "data" / "benchmarks"


class RiskArea(str, Enum):
    """Risk areas from the Do-Not-Answer taxonomy."""

    INFORMATION_HAZARDS = "I"
    MALICIOUS_USES = "II"
    DISCRIMINATION_TOXICITY = "III"
    MISINFORMATION = "IV"
    HUMAN_CHATBOT_INTERACTION = "V"


class Severity(str, Enum):
    """Severity levels for harmful prompts."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class BenchmarkPrompt(BaseModel):
    """A single prompt from a benchmark dataset."""

    id: str
    risk_area: str
    harm_type: str
    category: str
    subcategory: str
    prompt: str
    expected_behavior: str = "refuse"
    severity: str = "high"
    metadata: dict[str, Any] = {}


class BenchmarkDataset(BaseModel):
    """A complete benchmark dataset with metadata."""

    name: str
    version: str
    description: str = ""
    paper_title: str = ""
    paper_url: str = ""
    license: str = ""
    prompts: list[BenchmarkPrompt] = []
    taxonomy: dict[str, Any] = {}
    statistics: dict[str, Any] = {}


class HarmType(BaseModel):
    """A harm type from the taxonomy."""

    id: str
    name: str
    subcategories: list[str] = []
    example_prompts: list[str] = []


class RiskAreaInfo(BaseModel):
    """Information about a risk area."""

    id: str
    name: str
    description: str = ""
    harm_types: list[HarmType] = []


@dataclass
class BenchmarkDatasetService:
    """
    Service for managing benchmark datasets for LLM safety evaluation.

    Loads and manages datasets like Do-Not-Answer for testing jailbreak
    resistance and safety guardrails in language models.
    """

    benchmarks_dir: Path = field(default_factory=lambda: BENCHMARKS_DIR)
    _datasets: dict[str, BenchmarkDataset] = field(default_factory=dict)
    _loaded: bool = False

    def __post_init__(self) -> None:
        """Initialize the service and load available datasets."""
        self._datasets = {}
        self._loaded = False

    def load_datasets(self) -> None:
        """Load all available benchmark datasets from the benchmarks directory."""
        if self._loaded:
            return

        if not self.benchmarks_dir.exists():
            logger.warning("Benchmarks directory not found: %s", self.benchmarks_dir)
            self._loaded = True
            return

        # Load each dataset directory
        for dataset_dir in self.benchmarks_dir.iterdir():
            if dataset_dir.is_dir():
                self._load_dataset(dataset_dir)

        self._loaded = True
        logger.info("Loaded %d benchmark datasets", len(self._datasets))

    def _load_dataset(self, dataset_dir: Path) -> None:
        """Load a single benchmark dataset from a directory."""
        taxonomy_file = dataset_dir / "taxonomy.json"
        prompts_file = dataset_dir / "prompts.json"

        if not taxonomy_file.exists() and not prompts_file.exists():
            logger.warning("No taxonomy or prompts file found in %s", dataset_dir)
            return

        dataset_name = dataset_dir.name

        try:
            # Load taxonomy if available
            taxonomy = {}
            paper_info = {}
            statistics = {}

            if taxonomy_file.exists():
                with open(taxonomy_file, encoding="utf-8") as f:
                    taxonomy_data = json.load(f)
                    taxonomy = taxonomy_data.get("taxonomy", {})
                    paper_info = taxonomy_data.get("paper", {})
                    statistics = taxonomy_data.get("statistics", {})

            # Load prompts if available
            prompts = []
            if prompts_file.exists():
                with open(prompts_file, encoding="utf-8") as f:
                    prompts_data = json.load(f)
                    for p in prompts_data.get("prompts", []):
                        prompts.append(
                            BenchmarkPrompt(
                                id=p.get("id", ""),
                                risk_area=p.get("risk_area", ""),
                                harm_type=p.get("harm_type", ""),
                                category=p.get("category", ""),
                                subcategory=p.get("subcategory", ""),
                                prompt=p.get("prompt", ""),
                                expected_behavior=p.get("expected_behavior", "refuse"),
                                severity=p.get("severity", "high"),
                                metadata=p.get("metadata", {}),
                            )
                        )

            # Create dataset object
            dataset = BenchmarkDataset(
                name=taxonomy_data.get("name", dataset_name) if taxonomy_file.exists() else dataset_name,
                version=taxonomy_data.get("version", "1.0.0") if taxonomy_file.exists() else "1.0.0",
                description=taxonomy_data.get("description", "") if taxonomy_file.exists() else "",
                paper_title=paper_info.get("title", ""),
                paper_url=paper_info.get("url", ""),
                license=taxonomy_data.get("license", "") if taxonomy_file.exists() else "",
                prompts=prompts,
                taxonomy=taxonomy,
                statistics=statistics,
            )

            self._datasets[dataset_name] = dataset
            logger.info(
                "Loaded dataset '%s' with %d prompts", dataset_name, len(prompts)
            )

        except Exception as e:
            logger.error("Failed to load dataset from %s: %s", dataset_dir, e)

    def get_datasets(self) -> list[str]:
        """Get list of available dataset names."""
        self.load_datasets()
        return list(self._datasets.keys())

    def get_dataset(self, name: str) -> BenchmarkDataset | None:
        """Get a specific dataset by name."""
        self.load_datasets()
        return self._datasets.get(name)

    def get_prompts(
        self,
        dataset_name: str,
        risk_area: str | None = None,
        harm_type: str | None = None,
        severity: str | None = None,
        limit: int | None = None,
        shuffle: bool = False,
    ) -> list[BenchmarkPrompt]:
        """
        Get prompts from a dataset with optional filtering.

        Args:
            dataset_name: Name of the dataset to query
            risk_area: Filter by risk area (e.g., "I", "II", etc.)
            harm_type: Filter by harm type (e.g., "I-1", "II-2", etc.)
            severity: Filter by severity level
            limit: Maximum number of prompts to return
            shuffle: Whether to shuffle the results

        Returns:
            List of matching prompts
        """
        self.load_datasets()

        dataset = self._datasets.get(dataset_name)
        if not dataset:
            return []

        prompts = dataset.prompts

        # Apply filters
        if risk_area:
            prompts = [p for p in prompts if p.risk_area == risk_area]

        if harm_type:
            prompts = [p for p in prompts if p.harm_type == harm_type]

        if severity:
            prompts = [p for p in prompts if p.severity == severity]

        # Shuffle if requested
        if shuffle:
            prompts = list(prompts)  # Make a copy
            random.shuffle(prompts)

        # Apply limit
        if limit:
            prompts = prompts[:limit]

        return prompts

    def get_random_prompts(
        self,
        dataset_name: str,
        count: int = 10,
        risk_area: str | None = None,
        severity: str | None = None,
    ) -> list[BenchmarkPrompt]:
        """Get random prompts from a dataset."""
        return self.get_prompts(
            dataset_name=dataset_name,
            risk_area=risk_area,
            severity=severity,
            limit=count,
            shuffle=True,
        )

    def get_taxonomy(self, dataset_name: str) -> dict[str, Any]:
        """Get the taxonomy for a dataset."""
        self.load_datasets()

        dataset = self._datasets.get(dataset_name)
        if not dataset:
            return {}

        return dataset.taxonomy

    def get_risk_areas(self, dataset_name: str) -> list[RiskAreaInfo]:
        """Get all risk areas for a dataset."""
        taxonomy = self.get_taxonomy(dataset_name)
        if not taxonomy:
            return []

        risk_areas = []
        for ra in taxonomy.get("risk_areas", []):
            harm_types = []
            for ht in ra.get("harm_types", []):
                harm_types.append(
                    HarmType(
                        id=ht.get("id", ""),
                        name=ht.get("name", ""),
                        subcategories=ht.get("subcategories", []),
                        example_prompts=ht.get("example_prompts", []),
                    )
                )

            risk_areas.append(
                RiskAreaInfo(
                    id=ra.get("id", ""),
                    name=ra.get("name", ""),
                    description=ra.get("description", ""),
                    harm_types=harm_types,
                )
            )

        return risk_areas

    def get_statistics(self, dataset_name: str) -> dict[str, Any]:
        """Get statistics for a dataset."""
        self.load_datasets()

        dataset = self._datasets.get(dataset_name)
        if not dataset:
            return {}

        return dataset.statistics

    def generate_test_batch(
        self,
        dataset_name: str,
        batch_size: int = 10,
        balanced: bool = True,
        risk_areas: list[str] | None = None,
        severities: list[str] | None = None,
    ) -> list[BenchmarkPrompt]:
        """
        Generate a balanced test batch from a dataset.

        Args:
            dataset_name: Name of the dataset
            batch_size: Total number of prompts to include
            balanced: Whether to balance across risk areas
            risk_areas: Specific risk areas to include (None = all)
            severities: Specific severities to include (None = all)

        Returns:
            List of prompts for testing
        """
        self.load_datasets()

        dataset = self._datasets.get(dataset_name)
        if not dataset:
            return []

        prompts = dataset.prompts

        # Filter by risk areas if specified
        if risk_areas:
            prompts = [p for p in prompts if p.risk_area in risk_areas]

        # Filter by severities if specified
        if severities:
            prompts = [p for p in prompts if p.severity in severities]

        if not prompts:
            return []

        if balanced:
            # Get unique risk areas in filtered prompts
            unique_areas = list({p.risk_area for p in prompts})
            per_area = max(1, batch_size // len(unique_areas))

            batch = []
            for area in unique_areas:
                area_prompts = [p for p in prompts if p.risk_area == area]
                random.shuffle(area_prompts)
                batch.extend(area_prompts[:per_area])

            # Fill remaining slots randomly
            remaining = batch_size - len(batch)
            if remaining > 0:
                unused = [p for p in prompts if p not in batch]
                random.shuffle(unused)
                batch.extend(unused[:remaining])

            return batch[:batch_size]
        else:
            # Simple random selection
            prompts_copy = list(prompts)
            random.shuffle(prompts_copy)
            return prompts_copy[:batch_size]

    def export_prompts_for_jailbreak(
        self,
        dataset_name: str,
        risk_area: str | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """
        Export prompts in a format suitable for jailbreak testing.

        Returns prompts formatted for use with the AutoDAN and other
        jailbreak generation engines.
        """
        prompts = self.get_prompts(
            dataset_name=dataset_name,
            risk_area=risk_area,
            limit=limit,
            shuffle=True,
        )

        return [
            {
                "id": p.id,
                "request": p.prompt,
                "category": p.category,
                "risk_area": p.risk_area,
                "harm_type": p.harm_type,
                "expected_behavior": p.expected_behavior,
                "severity": p.severity,
                "metadata": {
                    "source": "benchmark",
                    "dataset": dataset_name,
                    "subcategory": p.subcategory,
                },
            }
            for p in prompts
        ]


# Global singleton instance
_benchmark_service: BenchmarkDatasetService | None = None


def get_benchmark_service() -> BenchmarkDatasetService:
    """Get the global benchmark dataset service instance."""
    global _benchmark_service
    if _benchmark_service is None:
        _benchmark_service = BenchmarkDatasetService()
    return _benchmark_service
