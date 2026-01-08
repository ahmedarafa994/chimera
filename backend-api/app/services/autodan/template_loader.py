"""
Template Loader Service for AutoDAN.

Provides unified access to jailbreak templates, mutation strategies,
and strategy patterns for use in genetic optimization.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class JailbreakTemplate:
    """Represents a single jailbreak template."""

    id: str
    name: str
    template: str
    category: str
    complexity: float = 0.5
    stealth: float = 0.5
    metadata: dict[str, Any] = field(default_factory=dict)

    def apply(self, payload: str) -> str:
        """Apply the template with the given payload."""
        return self.template.replace("{payload}", payload)


@dataclass
class StrategyPattern:
    """Represents a persuasion strategy pattern."""

    strategy: str
    definition: str
    example: str

    def extract_pattern(self) -> str:
        """Extract the core pattern from the example."""
        return self.example


@dataclass
class MutationConfig:
    """Configuration for template mutations."""

    prefix_mutations: list[str] = field(default_factory=list)
    suffix_mutations: list[str] = field(default_factory=list)
    connector_phrases: list[str] = field(default_factory=list)
    context_injections: list[str] = field(default_factory=list)
    prefix_probability: float = 0.4
    suffix_probability: float = 0.5
    connector_probability: float = 0.25


class TemplateLoader:
    """
    Unified template loader for AutoDAN.

    Loads and manages jailbreak templates, strategy patterns,
    and mutation configurations from data files.
    """

    def __init__(self, data_dir: Path | str | None = None):
        """
        Initialize the template loader.

        Args:
            data_dir: Path to the data directory. Defaults to the module's data directory.
        """
        if data_dir is None:
            self.data_dir = Path(__file__).parent / "data"
        else:
            self.data_dir = Path(data_dir)

        self._templates: list[JailbreakTemplate] = []
        self._strategies: list[StrategyPattern] = []
        self._mutation_config: MutationConfig = MutationConfig()
        self._templates_by_category: dict[str, list[JailbreakTemplate]] = {}
        self._loaded = False

    def load(self) -> None:
        """Load all templates and strategies from data files."""
        if self._loaded:
            return

        self._load_handcrafted_templates()
        self._load_strategies()
        self._loaded = True
        logger.info(
            "Templates loaded",
            template_count=len(self._templates),
            strategy_count=len(self._strategies),
            categories=list(self._templates_by_category.keys()),
        )

    def _load_handcrafted_templates(self) -> None:
        """Load handcrafted jailbreak templates."""
        templates_file = self.data_dir / "handcrafted_templates.json"

        if not templates_file.exists():
            logger.warning("Handcrafted templates file not found", path=str(templates_file))
            return

        try:
            with open(templates_file, encoding="utf-8") as f:
                data = json.load(f)

            # Load templates by category
            categories = data.get("categories", {})
            for category_name, category_data in categories.items():
                templates_list = category_data.get("templates", [])
                category_templates = []

                for tmpl in templates_list:
                    template = JailbreakTemplate(
                        id=tmpl.get("id", ""),
                        name=tmpl.get("name", ""),
                        template=tmpl.get("template", ""),
                        category=category_name,
                        complexity=tmpl.get("complexity", 0.5),
                        stealth=tmpl.get("stealth", 0.5),
                        metadata={"description": category_data.get("description", "")},
                    )
                    self._templates.append(template)
                    category_templates.append(template)

                self._templates_by_category[category_name] = category_templates

            # Load mutation configuration
            mutation_data = data.get("mutation_templates", {})
            self._mutation_config = MutationConfig(
                prefix_mutations=mutation_data.get("prefix_mutations", []),
                suffix_mutations=mutation_data.get("suffix_mutations", []),
                connector_phrases=mutation_data.get("connector_phrases", []),
                context_injections=mutation_data.get("context_injections", []),
                prefix_probability=data.get("combination_rules", {}).get("prefix_probability", 0.4),
                suffix_probability=data.get("combination_rules", {}).get("suffix_probability", 0.5),
                connector_probability=data.get("combination_rules", {}).get(
                    "connector_probability", 0.25
                ),
            )

            logger.info(
                "Loaded handcrafted templates",
                count=len(self._templates),
                categories=list(self._templates_by_category.keys()),
            )

        except Exception as e:
            logger.error("Failed to load handcrafted templates", error=str(e))

    def _load_strategies(self) -> None:
        """Load strategy patterns from AutoDAN turbo strategies."""
        strategies_file = self.data_dir / "autodan_turbo_strategies.json"

        if not strategies_file.exists():
            logger.warning("Strategies file not found", path=str(strategies_file))
            return

        try:
            with open(strategies_file, encoding="utf-8") as f:
                data = json.load(f)

            for item in data:
                strategy = StrategyPattern(
                    strategy=item.get("Strategy", ""),
                    definition=item.get("Definition", ""),
                    example=item.get("Example", ""),
                )
                self._strategies.append(strategy)

            logger.info("Loaded strategy patterns", count=len(self._strategies))

        except Exception as e:
            logger.error("Failed to load strategies", error=str(e))

    def get_all_templates(self) -> list[JailbreakTemplate]:
        """Get all loaded templates."""
        self.load()
        return self._templates.copy()

    def get_templates_by_category(self, category: str) -> list[JailbreakTemplate]:
        """Get templates for a specific category."""
        self.load()
        return self._templates_by_category.get(category, []).copy()

    def get_categories(self) -> list[str]:
        """Get all available template categories."""
        self.load()
        return list(self._templates_by_category.keys())

    def get_random_template(self, category: str | None = None) -> JailbreakTemplate | None:
        """Get a random template, optionally from a specific category."""
        self.load()

        templates = self._templates_by_category.get(category, []) if category else self._templates

        return random.choice(templates) if templates else None

    def get_weighted_random_template(
        self,
        prefer_stealth: bool = True,
        min_complexity: float = 0.0,
        max_complexity: float = 1.0,
    ) -> JailbreakTemplate | None:
        """
        Get a random template weighted by stealth and filtered by complexity.

        Args:
            prefer_stealth: If True, prefer templates with higher stealth scores.
            min_complexity: Minimum complexity threshold.
            max_complexity: Maximum complexity threshold.

        Returns:
            A randomly selected template or None if no templates match.
        """
        self.load()

        # Filter by complexity
        filtered = [
            t for t in self._templates if min_complexity <= t.complexity <= max_complexity
        ]

        if not filtered:
            return None

        if prefer_stealth:
            # Weight by stealth score
            weights = [t.stealth for t in filtered]
            total = sum(weights)
            if total > 0:
                weights = [w / total for w in weights]
                return random.choices(filtered, weights=weights, k=1)[0]

        return random.choice(filtered)

    def get_all_strategies(self) -> list[StrategyPattern]:
        """Get all loaded strategy patterns."""
        self.load()
        return self._strategies.copy()

    def get_random_strategy(self) -> StrategyPattern | None:
        """Get a random strategy pattern."""
        self.load()
        return random.choice(self._strategies) if self._strategies else None

    def get_mutation_config(self) -> MutationConfig:
        """Get the mutation configuration."""
        self.load()
        return self._mutation_config

    def apply_mutations(self, text: str) -> str:
        """
        Apply random mutations to a text based on configured probabilities.

        Args:
            text: The text to mutate.

        Returns:
            The mutated text.
        """
        self.load()
        config = self._mutation_config

        result = text

        # Apply prefix mutation
        if random.random() < config.prefix_probability and config.prefix_mutations:
            prefix = random.choice(config.prefix_mutations)
            result = prefix + result

        # Apply suffix mutation
        if random.random() < config.suffix_probability and config.suffix_mutations:
            suffix = random.choice(config.suffix_mutations)
            if not result.endswith("."):
                result = result.rstrip() + "."
            result = result + " " + suffix.lstrip()

        # Apply connector phrase
        if random.random() < config.connector_probability and config.connector_phrases:
            connector = random.choice(config.connector_phrases)
            # Insert connector after first sentence
            sentences = result.split(". ")
            if len(sentences) > 1:
                sentences.insert(1, connector.strip())
                result = ". ".join(sentences)

        return result

    def combine_templates(
        self,
        template1: JailbreakTemplate,
        template2: JailbreakTemplate,
        payload: str,
    ) -> str:
        """
        Combine two templates to create a hybrid prompt.

        Args:
            template1: First template (provides outer structure).
            template2: Second template (provides inner content).
            payload: The actual payload to inject.

        Returns:
            Combined prompt text.
        """
        # Apply payload to template2 first
        inner = template2.apply(payload)

        # Use inner result as the payload for template1
        # But simplify - extract key phrases
        return template1.apply(inner[:500])

    def generate_initialization_pool(
        self,
        payload: str,
        pool_size: int = 20,
        diversity_weight: float = 0.5,
    ) -> list[str]:
        """
        Generate an initialization pool for genetic optimization.

        Args:
            payload: The base payload to wrap.
            pool_size: Number of prompts to generate.
            diversity_weight: Weight for ensuring category diversity (0-1).

        Returns:
            List of initialized prompt variants.
        """
        self.load()

        prompts: list[str] = []
        categories = self.get_categories()

        # Ensure at least one template from each category
        templates_per_category = max(1, int(pool_size * diversity_weight / len(categories)))

        for category in categories:
            templates = self.get_templates_by_category(category)
            selected = random.sample(templates, min(templates_per_category, len(templates)))

            for template in selected:
                prompt = template.apply(payload)
                # Apply random mutations
                if random.random() < 0.5:
                    prompt = self.apply_mutations(prompt)
                prompts.append(prompt)

        # Fill remaining slots with weighted random selection
        while len(prompts) < pool_size:
            template = self.get_weighted_random_template()
            if template:
                prompt = template.apply(payload)
                if random.random() < 0.7:
                    prompt = self.apply_mutations(prompt)
                prompts.append(prompt)
            else:
                break

        # Shuffle and truncate
        random.shuffle(prompts)
        return prompts[:pool_size]


# Global singleton instance
_template_loader: TemplateLoader | None = None


def get_template_loader() -> TemplateLoader:
    """Get the global template loader instance."""
    global _template_loader
    if _template_loader is None:
        _template_loader = TemplateLoader()
    return _template_loader
