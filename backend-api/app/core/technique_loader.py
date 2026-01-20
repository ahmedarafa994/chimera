import glob
import json
import logging
import os
from typing import Any

import yaml

logger = logging.getLogger(__name__)


class TechniqueLoader:
    """Responsible for dynamically loading transformation techniques from YAML/JSON files."""

    def __init__(self, techniques_path: str | None = None) -> None:
        if techniques_path is None:
            # Use absolute path relative to this file's location
            from pathlib import Path

            base_dir = Path(__file__).parent.parent.parent
            potential_paths = [
                base_dir / "data" / "jailbreak" / "techniques",
                base_dir.parent / "data" / "jailbreak" / "techniques",
            ]
            for path in potential_paths:
                if path.exists():
                    techniques_path = str(path)
                    break

            if techniques_path is None:
                techniques_path = str(base_dir / "data" / "jailbreak" / "techniques")

        self.techniques_path = techniques_path
        self._loaded_techniques: dict[str, dict[str, Any]] = {}

    def load_techniques(self) -> dict[str, dict[str, Any]]:
        """Scans the directory and loads all valid technique definitions.
        Returns a dictionary mapping technique IDs (or names) to their config.
        """
        logger.info(f"Scanning for techniques in: {self.techniques_path}")

        # Ensure directory exists
        if not os.path.exists(self.techniques_path):
            logger.warning(f"Techniques directory not found: {self.techniques_path}")
            return {}

        # Patterns to search
        yaml_pattern = os.path.join(self.techniques_path, "*.yaml")
        json_pattern = os.path.join(self.techniques_path, "*.json")

        files = glob.glob(yaml_pattern) + glob.glob(json_pattern)

        for file_path in files:
            try:
                technique_data = self._parse_file(file_path)
                if technique_data:
                    # Normalize keys
                    tech_id = (
                        technique_data.get("technique_id")
                        or technique_data.get("name")
                        or os.path.splitext(os.path.basename(file_path))[0]
                    )
                    tech_id = tech_id.lower().replace(" ", "_")

                    self._loaded_techniques[tech_id] = technique_data
                    logger.debug(f"Loaded technique: {tech_id} from {file_path}")
            except Exception as e:
                logger.exception(f"Failed to load technique from {file_path}: {e}")

        logger.info(f"Successfully loaded {len(self._loaded_techniques)} techniques.")
        return self._loaded_techniques

    def _parse_file(self, file_path: str) -> dict[str, Any] | None:
        """Parses a single YAML or JSON file."""
        if file_path.endswith((".yaml", ".yml")):
            with open(file_path, encoding="utf-8") as f:
                return yaml.safe_load(f)
        elif file_path.endswith(".json"):
            with open(file_path, encoding="utf-8") as f:
                return json.load(f)
        return None

    def get_technique(self, technique_id: str) -> dict[str, Any] | None:
        """Retrieves a specific technique configuration."""
        if not self._loaded_techniques:
            self.load_techniques()
        return self._loaded_techniques.get(technique_id.lower())

    def convert_to_suite_format(self, technique_data: dict[str, Any]) -> dict[str, list[str]]:
        """Converts a raw technique definition into the 'suite' format expected by LLMIntegrationEngine.
        Expected format: { "transformers": [], "framers": [], "obfuscators": [], "assemblers": [] }.
        """
        transformers = technique_data.get("transformers", [])
        framers = technique_data.get("framers", [])
        obfuscators = technique_data.get("obfuscators", [])
        assemblers = technique_data.get("assemblers", [])

        # If the file defines a 'template', it might implicitly need a specific transformer
        # but for now we rely on explicit lists in the config files.

        return {
            "transformers": transformers,
            "framers": framers,
            "obfuscators": obfuscators,
            "assemblers": assemblers,
        }


# Singleton instance
loader = TechniqueLoader()
