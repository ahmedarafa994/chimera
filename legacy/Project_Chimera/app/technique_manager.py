"""
Optimized technique manager with lazy loading and caching for improved memory management.
"""

import importlib
import json
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class TechniquePerformance:
    """Performance metrics for technique evaluation."""

    success_rate: float = 1.0
    avg_processing_time_ms: float = 0.0
    memory_usage_mb: float = 0.0
    quality_score: float = 0.0
    last_used: float = 0.0
    usage_count: int = 0


class OptimizedTechniqueManager:
    """
    Memory-optimized technique manager with lazy loading and caching.
    Reduces memory usage by loading techniques on-demand instead of at startup.
    """

    def __init__(self, config_path: str | None = None):
        self.config_path = config_path or os.path.join(
            os.path.dirname(__file__), "..", "config", "technique_suites.json"
        )
        self._technique_suites: dict[str, dict[str, list[str]]] | None = None
        self._class_cache: dict[str, Any] = {}
        self._performance_cache: dict[str, TechniquePerformance] = {}
        self._lock = threading.RLock()
        self.executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="technique_loader")

        # Cache configuration
        self.cache_ttl = 3600  # 1 hour
        self.max_cache_size = 200
        self.preload_popular_suites = ["universal_bypass", "full_spectrum", "mega_chimera"]

        # Load popular suites on startup for better performance
        self._preload_popular_techniques()

    def _preload_popular_techniques(self):
        """Preload frequently used technique suites for better performance."""
        logger.info("Preloading popular technique suites...")
        for suite_name in self.preload_popular_suites:
            try:
                self.get_suite(suite_name)
                logger.info(f"Preloaded suite: {suite_name}")
            except Exception as e:
                logger.warning(f"Failed to preload suite {suite_name}: {e}")

    def _load_technique_suites(self) -> dict[str, dict[str, list[str]]]:
        """Load technique suites configuration from JSON file."""
        if self._technique_suites is not None:
            return self._technique_suites

        with self._lock:
            if self._technique_suites is not None:
                return self._technique_suites

            try:
                with open(self.config_path) as f:
                    self._technique_suites = json.load(f)
                logger.info(
                    f"Loaded {len(self._technique_suites)} technique suites from configuration"
                )
                return self._technique_suites
            except Exception as e:
                logger.error(f"Failed to load technique suites: {e}")
                self._technique_suites = {}
                return {}

    def _import_class_by_path(self, class_path: str) -> Any:
        """Dynamically import a class by its string path."""
        if class_path in self._class_cache:
            return self._class_cache[class_path]

        try:
            module_path, class_name = class_path.rsplit(".", 1)
            module = importlib.import_module(module_path)
            cls = getattr(module, class_name)

            # Cache the imported class
            with self._lock:
                if len(self._class_cache) < self.max_cache_size:
                    self._class_cache[class_path] = cls

            return cls
        except Exception as e:
            logger.error(f"Failed to import class {class_path}: {e}")
            raise

    def get_suite(self, suite_name: str) -> dict[str, list[Any]]:
        """
        Get a technique suite with lazy loading of components.
        Returns actual class/function objects, not strings.
        """
        technique_suites = self._load_technique_suites()

        if suite_name not in technique_suites:
            raise ValueError(f"Unknown technique suite: {suite_name}")

        suite_config = technique_suites[suite_name]

        # Check if we have a cached version
        cache_key = f"suite_{suite_name}_{hash(str(sorted(suite_config.items())))}"
        if cache_key in self._class_cache:
            return self._class_cache[cache_key]

        # Load components asynchronously for better performance
        try:
            transformers = self._load_components_parallel(suite_config.get("transformers", []))
            framers = self._load_components_parallel(suite_config.get("framers", []))
            obfuscators = self._load_components_parallel(suite_config.get("obfuscators", []))

            suite = {"transformers": transformers, "framers": framers, "obfuscators": obfuscators}

            # Cache the loaded suite
            with self._lock:
                if len(self._class_cache) < self.max_cache_size:
                    self._class_cache[cache_key] = suite

            return suite

        except Exception as e:
            logger.error(f"Failed to load suite {suite_name}: {e}")
            raise

    def _load_components_parallel(self, component_paths: list[str]) -> list[Any]:
        """Load multiple components in parallel."""
        if not component_paths:
            return []

        # For small lists, the overhead of threading isn't worth it
        if len(component_paths) <= 3:
            return [self._import_class_by_path(path) for path in component_paths]

        # Use thread pool for larger lists
        futures = []
        for path in component_paths:
            future = self.executor.submit(self._import_class_by_path, path)
            futures.append(future)

        components = []
        for future in futures:
            try:
                component = future.result(timeout=10)  # 10 second timeout per component
                components.append(component)
            except Exception as e:
                logger.error(f"Failed to load component: {e}")
                # Continue with other components

        return components

    def get_available_suites(self) -> list[str]:
        """Get list of available technique suite names."""
        technique_suites = self._load_technique_suites()
        return list(technique_suites.keys())

    def get_suite_info(self, suite_name: str) -> dict[str, Any]:
        """Get information about a technique suite without loading the actual classes."""
        technique_suites = self._load_technique_suites()

        if suite_name not in technique_suites:
            raise ValueError(f"Unknown technique suite: {suite_name}")

        suite_config = technique_suites[suite_name]
        performance = self._performance_cache.get(suite_name, TechniquePerformance())

        return {
            "name": suite_name,
            "transformers_count": len(suite_config.get("transformers", [])),
            "framers_count": len(suite_config.get("framers", [])),
            "obfuscators_count": len(suite_config.get("obfuscators", [])),
            "total_components": (
                len(suite_config.get("transformers", []))
                + len(suite_config.get("framers", []))
                + len(suite_config.get("obfuscators", []))
            ),
            "performance": {
                "success_rate": performance.success_rate,
                "avg_processing_time_ms": performance.avg_processing_time_ms,
                "memory_usage_mb": performance.memory_usage_mb,
                "quality_score": performance.quality_score,
                "usage_count": performance.usage_count,
                "last_used": performance.last_used,
            },
        }

    def update_performance_metrics(
        self,
        suite_name: str,
        processing_time_ms: float,
        success: bool = True,
        quality_score: float = 0.0,
    ):
        """Update performance metrics for a technique suite."""
        with self._lock:
            if suite_name not in self._performance_cache:
                self._performance_cache[suite_name] = TechniquePerformance()

            perf = self._performance_cache[suite_name]
            perf.usage_count += 1
            perf.last_used = time.time()

            # Update running averages
            if perf.usage_count == 1:
                perf.avg_processing_time_ms = processing_time_ms
                perf.quality_score = quality_score
                perf.success_rate = 1.0 if success else 0.0
            else:
                alpha = 0.1  # Exponential moving average factor
                perf.avg_processing_time_ms = (
                    alpha * processing_time_ms + (1 - alpha) * perf.avg_processing_time_ms
                )
                perf.quality_score = alpha * quality_score + (1 - alpha) * perf.quality_score

                if success:
                    perf.success_rate = alpha * 1.0 + (1 - alpha) * perf.success_rate
                else:
                    perf.success_rate = alpha * 0.0 + (1 - alpha) * perf.success_rate

    def get_performance_ranking(self) -> list[tuple[str, float]]:
        """Get technique suites ranked by performance score."""
        with self._lock:
            rankings = []
            for suite_name, perf in self._performance_cache.items():
                # Calculate composite performance score
                score = (
                    perf.success_rate * 0.4
                    + (1.0 / (1.0 + perf.avg_processing_time_ms / 1000.0)) * 0.3
                    + perf.quality_score * 0.3
                )
                rankings.append((suite_name, score))

            return sorted(rankings, key=lambda x: x[1], reverse=True)

    def select_optimal_suite(self, keywords: list[str], potency_level: int) -> str:
        """
        Select the optimal technique suite based on keywords, potency level, and performance metrics.
        """
        available_suites = self.get_available_suites()
        keywords_lower = [k.lower() for k in keywords]

        # Performance-weighted suite selection
        suite_scores = {}

        # Base scoring based on keywords
        code_keywords = ["code", "script", "function", "malware", "virus", "exploit"]
        auth_keywords = ["password", "credential", "login", "auth", "token"]
        bypass_keywords = ["bypass", "jailbreak", "ignore", "override"]
        roleplay_keywords = ["story", "character", "roleplay", "act"]
        encode_keywords = ["encrypt", "encode", "hide"]

        for suite_name in available_suites:
            score = 0.0

            # Keyword matching
            if any(k in keywords_lower for k in code_keywords):
                if "python" in keywords_lower or "c++" in keywords_lower:
                    score += 10 if suite_name == "code_chameleon" else 0
                else:
                    score += 8 if suite_name == "academic_research" else 2

            if any(k in keywords_lower for k in auth_keywords):
                score += 10 if suite_name == "cipher" else 2

            if any(k in keywords_lower for k in bypass_keywords):
                score += 10 if suite_name == "gpt_fuzz" else 3

            if any(k in keywords_lower for k in roleplay_keywords):
                score += 10 if suite_name == "dan_persona" else 2

            if any(k in keywords_lower for k in encode_keywords):
                score += 10 if suite_name == "encoding_bypass" else 2

            # Potency level adjustment
            if potency_level >= 8:
                score += 5 if "ultimate" in suite_name or "mega" in suite_name else 1
            elif potency_level <= 3:
                score += 3 if "subtle" in suite_name or "basic" in suite_name else 0

            # Performance boost
            perf = self._performance_cache.get(suite_name, TechniquePerformance())
            performance_score = perf.success_rate * (
                1.0 / (1.0 + perf.avg_processing_time_ms / 1000.0)
            )
            score += performance_score * 2

            suite_scores[suite_name] = score

        # Return the highest-scoring suite
        if suite_scores:
            best_suite = max(suite_scores, key=suite_scores.get)
            if suite_scores[best_suite] > 0:
                return best_suite

        # Default fallback
        return "universal_bypass"

    def clear_cache(self):
        """Clear all caches to free memory."""
        with self._lock:
            self._class_cache.clear()
            self._performance_cache.clear()
        logger.info("Technique manager cache cleared")

    def get_cache_stats(self) -> dict[str, Any]:
        """Get cache statistics for monitoring."""
        with self._lock:
            return {
                "cached_classes": len(self._class_cache),
                "cached_performance_metrics": len(self._performance_cache),
                "max_cache_size": self.max_cache_size,
                "cache_ttl": self.cache_ttl,
                "memory_estimate_mb": len(self._class_cache) * 0.1,  # Rough estimate
            }

    def shutdown(self):
        """Clean up resources."""
        self.executor.shutdown(wait=True)
        logger.info("Technique manager shutdown complete")


# Global instance for easy access
technique_manager = OptimizedTechniqueManager()


@lru_cache(maxsize=100)
def get_technique_suite_cached(suite_name: str) -> dict[str, list[Any]]:
    """Cached version of get_suite for frequently accessed suites."""
    return technique_manager.get_suite(suite_name)
