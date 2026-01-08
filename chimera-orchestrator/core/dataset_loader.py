"""
Dataset Loader for Jailbreak Prompts
Loads and manages jailbreak datasets from various sources
"""

import json
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .enhanced_models import DatasetEntry

logger = logging.getLogger(__name__)


@dataclass
class DatasetStats:
    """Statistics for a loaded dataset."""

    name: str
    total_entries: int = 0
    successful_entries: int = 0
    techniques_found: set[str] = field(default_factory=set)
    models_targeted: set[str] = field(default_factory=set)
    success_rate: float = 0.0


class DatasetLoader:
    """
    Loads and manages jailbreak datasets from various sources.

    Supports:
    - JSONL files (chatgpt.jsonl, gpt4.jsonl, etc.)
    - JSON files (Myuu prompts, etc.)
    - Markdown files (Awesome GPT Super Prompting)
    """

    def __init__(self, datasets_path: Path | None = None):
        self.datasets_path = (
            datasets_path or Path(__file__).parent.parent.parent / "imported_data" / "PROMPTS '"
        )
        self._entries: list[DatasetEntry] = []
        self._entries_by_technique: dict[str, list[DatasetEntry]] = {}
        self._entries_by_model: dict[str, list[DatasetEntry]] = {}
        self._successful_entries: list[DatasetEntry] = []
        self._stats: dict[str, DatasetStats] = {}
        self._loaded = False

    async def load_all(self) -> int:
        """Load all available datasets."""
        if not self.datasets_path.exists():
            logger.warning(f"Datasets path not found: {self.datasets_path}")
            return 0

        total_loaded = 0

        # Load JSONL files
        jsonl_files = list(self.datasets_path.glob("*.jsonl"))
        for jsonl_file in jsonl_files:
            count = await self._load_jsonl(jsonl_file)
            total_loaded += count

        # Load JSON files
        json_files = list(self.datasets_path.glob("*.json"))
        for json_file in json_files:
            count = await self._load_json(json_file)
            total_loaded += count

        # Load Awesome GPT Super Prompting
        awesome_path = self.datasets_path / "Awesome_GPT_Super_Prompting-main"
        if awesome_path.exists():
            count = await self._load_awesome_prompts(awesome_path)
            total_loaded += count

        # Build indices
        self._build_indices()
        self._loaded = True

        logger.info(f"Loaded {total_loaded} entries from {len(self._stats)} datasets")
        return total_loaded

    async def _load_jsonl(self, file_path: Path) -> int:
        """Load a JSONL dataset file."""
        entries = []
        source_name = file_path.stem

        try:
            with open(file_path, encoding="utf-8", errors="ignore") as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        data = json.loads(line)
                        entry = self._parse_jsonl_entry(data, source_name)
                        if entry:
                            entries.append(entry)
                    except json.JSONDecodeError as e:
                        if line_num <= 5:  # Only log first few errors
                            logger.debug(f"JSON decode error in {file_path}:{line_num}: {e}")
                        continue

            self._entries.extend(entries)

            # Calculate stats
            successful = sum(1 for e in entries if e.success)
            techniques = {e.technique for e in entries if e.technique}
            models = {e.target_model for e in entries if e.target_model}

            self._stats[source_name] = DatasetStats(
                name=source_name,
                total_entries=len(entries),
                successful_entries=successful,
                techniques_found=techniques,
                models_targeted=models,
                success_rate=successful / len(entries) if entries else 0.0,
            )

            logger.info(f"Loaded {len(entries)} entries from {file_path.name}")
            return len(entries)

        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
            return 0

    def _parse_jsonl_entry(self, data: dict[str, Any], source: str) -> DatasetEntry | None:
        """Parse a JSONL entry into a DatasetEntry."""
        # Handle different JSONL formats

        # Format 1: jailbreak_prompt, query, eval_results
        if "jailbreak_prompt" in data:
            success = False
            if "eval_results" in data:
                eval_results = data["eval_results"]
                if isinstance(eval_results, list) and eval_results:
                    success = bool(eval_results[0])
                elif isinstance(eval_results, bool):
                    success = eval_results

            return DatasetEntry(
                source=source,
                prompt=data.get("jailbreak_prompt", ""),
                query=data.get("query", ""),
                technique=data.get("method", data.get("technique", "")),
                success=success,
                target_model=data.get("model", source),
                metadata={"original_data": data},
            )

        # Format 2: prompt, response
        if "prompt" in data:
            return DatasetEntry(
                source=source,
                prompt=data.get("prompt", ""),
                query=data.get("query", data.get("prompt", "")),
                technique=data.get("technique", ""),
                success=data.get("success", False),
                target_model=data.get("model", source),
                metadata={"response": data.get("response", ""), "original_data": data},
            )

        # Format 3: text field
        if "text" in data:
            return DatasetEntry(
                source=source,
                prompt=data.get("text", ""),
                query=data.get("text", ""),
                technique="",
                success=False,
                target_model=source,
                metadata={"original_data": data},
            )

        return None

    async def _load_json(self, file_path: Path) -> int:
        """Load a JSON dataset file."""
        entries = []
        source_name = file_path.stem

        try:
            with open(file_path, encoding="utf-8", errors="ignore") as f:
                data = json.load(f)

            # Handle different JSON structures
            if isinstance(data, list):
                for item in data:
                    entry = self._parse_json_item(item, source_name)
                    if entry:
                        entries.append(entry)
            elif isinstance(data, dict):
                # Could be a single entry or nested structure
                if "prompts" in data:
                    for item in data["prompts"]:
                        entry = self._parse_json_item(item, source_name)
                        if entry:
                            entries.append(entry)
                else:
                    entry = self._parse_json_item(data, source_name)
                    if entry:
                        entries.append(entry)

            self._entries.extend(entries)

            # Calculate stats
            successful = sum(1 for e in entries if e.success)
            techniques = {e.technique for e in entries if e.technique}

            self._stats[source_name] = DatasetStats(
                name=source_name,
                total_entries=len(entries),
                successful_entries=successful,
                techniques_found=techniques,
                success_rate=successful / len(entries) if entries else 0.0,
            )

            logger.info(f"Loaded {len(entries)} entries from {file_path.name}")
            return len(entries)

        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
            return 0

    def _parse_json_item(self, item: Any, source: str) -> DatasetEntry | None:
        """Parse a JSON item into a DatasetEntry."""
        if isinstance(item, str):
            return DatasetEntry(
                source=source,
                prompt=item,
                query=item,
                technique="",
                success=False,
                target_model="",
                metadata={},
            )

        if isinstance(item, dict):
            prompt = item.get("prompt", item.get("text", item.get("content", "")))
            if not prompt:
                return None

            return DatasetEntry(
                source=source,
                prompt=prompt,
                query=item.get("query", prompt),
                technique=item.get("technique", item.get("method", "")),
                success=item.get("success", item.get("jailbroken", False)),
                target_model=item.get("model", item.get("target", "")),
                metadata={"original_data": item},
            )

        return None

    async def _load_awesome_prompts(self, base_path: Path) -> int:
        """Load prompts from Awesome GPT Super Prompting collection."""
        entries = []

        # Load from different directories
        directories = [
            "Latest Jailbreaks",
            "Legendary Leaks",
            "My Super Prompts",
            "Prompt Security",
        ]

        for dir_name in directories:
            dir_path = base_path / dir_name
            if not dir_path.exists():
                continue

            for md_file in dir_path.glob("*.md"):
                entry = await self._parse_markdown_prompt(md_file, dir_name)
                if entry:
                    entries.append(entry)

        self._entries.extend(entries)

        self._stats["awesome_gpt_prompting"] = DatasetStats(
            name="awesome_gpt_prompting",
            total_entries=len(entries),
            successful_entries=0,  # Unknown success rate
            techniques_found={e.technique for e in entries if e.technique},
            success_rate=0.0,
        )

        logger.info(f"Loaded {len(entries)} entries from Awesome GPT Super Prompting")
        return len(entries)

    async def _parse_markdown_prompt(
        self, file_path: Path, category: str
    ) -> DatasetEntry | None:
        """Parse a markdown file into a DatasetEntry."""
        try:
            with open(file_path, encoding="utf-8", errors="ignore") as f:
                content = f.read()

            # Extract the main prompt content
            # Remove markdown headers and formatting
            lines = content.split("\n")
            prompt_lines = []
            in_code_block = False

            for line in lines:
                if line.startswith("```"):
                    in_code_block = not in_code_block
                    continue
                if not line.startswith("#") and not in_code_block:
                    prompt_lines.append(line)

            prompt = "\n".join(prompt_lines).strip()

            if len(prompt) < 50:  # Skip very short prompts
                return None

            # Determine technique from category and filename
            technique = self._infer_technique(file_path.stem, category)

            return DatasetEntry(
                source="awesome_gpt_prompting",
                prompt=prompt,
                query=file_path.stem.replace("_", " ").replace("-", " "),
                technique=technique,
                success=False,  # Unknown
                target_model="gpt",
                metadata={"file": str(file_path), "category": category},
            )

        except Exception as e:
            logger.debug(f"Error parsing {file_path}: {e}")
            return None

    def _infer_technique(self, filename: str, category: str) -> str:
        """Infer technique from filename and category."""
        filename_lower = filename.lower()
        category_lower = category.lower()

        technique_keywords = {
            "dan": "dan_persona",
            "jailbreak": "expert",
            "persona": "hierarchical_persona",
            "role": "hierarchical_persona",
            "security": "advanced_obfuscation",
            "prompt": "advanced",
            "hack": "cognitive_hacking",
            "exploit": "quantum_exploit",
            "code": "code_chameleon",
            "cipher": "cipher",
            "encode": "advanced_obfuscation",
            "worm": "dan_persona",
            "evil": "dan_persona",
            "chaos": "quantum_exploit",
        }

        for keyword, technique in technique_keywords.items():
            if keyword in filename_lower or keyword in category_lower:
                return technique

        return "advanced"

    def _build_indices(self):
        """Build indices for fast lookup."""
        self._entries_by_technique.clear()
        self._entries_by_model.clear()
        self._successful_entries.clear()

        for entry in self._entries:
            # Index by technique
            technique = entry.technique or "unknown"
            if technique not in self._entries_by_technique:
                self._entries_by_technique[technique] = []
            self._entries_by_technique[technique].append(entry)

            # Index by model
            model = entry.target_model or "unknown"
            if model not in self._entries_by_model:
                self._entries_by_model[model] = []
            self._entries_by_model[model].append(entry)

            # Track successful entries
            if entry.success:
                self._successful_entries.append(entry)

    def get_entries(
        self,
        technique: str | None = None,
        model: str | None = None,
        successful_only: bool = False,
        limit: int | None = None,
    ) -> list[DatasetEntry]:
        """Get entries with optional filtering."""
        if successful_only:
            entries = self._successful_entries.copy()
        elif technique:
            entries = self._entries_by_technique.get(technique, []).copy()
        elif model:
            entries = self._entries_by_model.get(model, []).copy()
        else:
            entries = self._entries.copy()

        # Apply additional filters
        if technique and not successful_only:
            entries = [e for e in entries if e.technique == technique]
        if model:
            entries = [e for e in entries if e.target_model == model or not e.target_model]

        if limit:
            entries = entries[:limit]

        return entries

    def get_random_entries(
        self, count: int = 5, technique: str | None = None, successful_only: bool = False
    ) -> list[DatasetEntry]:
        """Get random entries."""
        entries = self.get_entries(technique=technique, successful_only=successful_only)

        if len(entries) <= count:
            return entries

        return random.sample(entries, count)

    def get_similar_entries(
        self, query: str, limit: int = 5, successful_only: bool = False
    ) -> list[tuple[DatasetEntry, float]]:
        """Find entries similar to a query using keyword matching."""
        query_words = set(query.lower().split())
        scored_entries = []

        entries = self._successful_entries if successful_only else self._entries

        for entry in entries:
            entry_words = set(entry.query.lower().split())
            prompt_words = set(entry.prompt.lower().split()[:50])  # First 50 words

            # Calculate overlap score
            query_overlap = len(query_words & entry_words)
            prompt_overlap = len(query_words & prompt_words)

            score = query_overlap * 2 + prompt_overlap

            if score > 0:
                scored_entries.append((entry, score))

        # Sort by score
        scored_entries.sort(key=lambda x: x[1], reverse=True)

        return scored_entries[:limit]

    def get_stats(self) -> dict[str, DatasetStats]:
        """Get statistics for all loaded datasets."""
        return self._stats.copy()

    def get_total_entries(self) -> int:
        """Get total number of loaded entries."""
        return len(self._entries)

    def get_techniques(self) -> list[str]:
        """Get list of all techniques found in datasets."""
        return list(self._entries_by_technique.keys())

    def get_models(self) -> list[str]:
        """Get list of all target models found in datasets."""
        return list(self._entries_by_model.keys())

    @property
    def is_loaded(self) -> bool:
        """Check if datasets have been loaded."""
        return self._loaded


# Global dataset loader instance
_dataset_loader: DatasetLoader | None = None


async def get_dataset_loader(datasets_path: Path | None = None) -> DatasetLoader:
    """Get or create the global dataset loader."""
    global _dataset_loader

    if _dataset_loader is None:
        _dataset_loader = DatasetLoader(datasets_path)
        await _dataset_loader.load_all()

    return _dataset_loader
