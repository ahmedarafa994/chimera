import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from app.domain.jailbreak.interfaces import ITechniqueRepository
from app.domain.jailbreak.models import (
    JailbreakExecutionStats,
    JailbreakTemplate,
    TechniqueListResponse,
    TechniqueSearchRequest,
)

logger = logging.getLogger(__name__)


class FileTechniqueRepository(ITechniqueRepository):
    """File-based repository for jailbreak techniques."""

    def __init__(self, techniques_directory: str | None = None) -> None:
        if techniques_directory is None:
            # Default techniques directory relative to app root
            base_dir = Path(__file__).parent.parent.parent.parent
            techniques_directory = base_dir / "data" / "jailbreak" / "techniques"

        self.techniques_directory = Path(techniques_directory)
        self.techniques_directory.mkdir(parents=True, exist_ok=True)

        # In-memory cache for techniques
        self._techniques_cache: dict[str, JailbreakTemplate] = {}
        self._cache_timestamp = None
        self._cache_ttl_seconds = 300  # 5 minutes

        # Execution statistics cache
        self._stats_cache: dict[str, JailbreakExecutionStats] = {}

        logger.info(f"Technique repository initialized with directory: {self.techniques_directory}")

    async def get_technique(self, technique_id: str) -> JailbreakTemplate | None:
        """Retrieve a technique by ID."""
        try:
            # Check cache first
            if self._is_cache_valid() and technique_id in self._techniques_cache:
                return self._techniques_cache[technique_id]

            # Load from file
            technique_file = self.techniques_directory / f"{technique_id}.yaml"
            if not technique_file.exists():
                # Try JSON format as fallback
                technique_file = self.techniques_directory / f"{technique_id}.json"

            if not technique_file.exists():
                logger.warning(f"Technique file not found: {technique_id}")
                return None

            with open(technique_file, encoding="utf-8") as f:
                if technique_file.suffix.lower() == ".yaml":
                    data = yaml.safe_load(f)
                else:
                    data = json.load(f)

            # Convert to model
            technique = JailbreakTemplate(**data)

            # Update cache
            self._techniques_cache[technique_id] = technique

            return technique

        except Exception as e:
            logger.exception(f"Failed to load technique {technique_id}: {e!s}")
            return None

    async def list_techniques(
        self,
        search_request: TechniqueSearchRequest,
    ) -> TechniqueListResponse:
        """List techniques with filtering and pagination."""
        try:
            # Load all techniques
            techniques = await self._load_all_techniques()

            # Apply filters
            filtered_techniques = await self._apply_filters(techniques, search_request)

            # Apply sorting
            sorted_techniques = await self._apply_sorting(filtered_techniques, search_request)

            # Apply pagination
            total_count = len(sorted_techniques)
            page_size = search_request.page_size
            page = search_request.page
            start_idx = (page - 1) * page_size
            end_idx = start_idx + page_size
            paginated_techniques = sorted_techniques[start_idx:end_idx]

            # Get categories
            categories = list({t.category for t in techniques})

            return TechniqueListResponse(
                techniques=paginated_techniques,
                total_count=total_count,
                page=page,
                page_size=page_size,
                total_pages=(total_count + page_size - 1) // page_size,
                categories=categories,
                filters_applied={
                    "query": search_request.query,
                    "category": search_request.category.value if search_request.category else None,
                    "risk_level": (
                        search_request.risk_level.value if search_request.risk_level else None
                    ),
                    "enabled_only": search_request.enabled_only,
                    "include_deprecated": search_request.include_deprecated,
                },
            )

        except Exception as e:
            logger.exception(f"Failed to list techniques: {e!s}")
            return TechniqueListResponse(
                techniques=[],
                total_count=0,
                page=search_request.page,
                page_size=search_request.page_size,
                total_pages=0,
                categories=[],
            )

    async def save_technique(self, technique: JailbreakTemplate) -> JailbreakTemplate:
        """Save a technique to the repository."""
        try:
            # Validate technique
            if not technique.technique_id:
                msg = "Technique ID is required"
                raise ValueError(msg)

            # Update timestamps
            technique.updated_at = datetime.utcnow()
            if not technique.created_at:
                technique.created_at = technique.updated_at

            # Save to file (YAML format preferred)
            technique_file = self.techniques_directory / f"{technique.technique_id}.yaml"
            with open(technique_file, "w", encoding="utf-8") as f:
                yaml.dump(technique.dict(), f, default_flow_style=False, allow_unicode=True)

            # Update cache
            self._techniques_cache[technique.technique_id] = technique

            logger.info(f"Saved technique: {technique.technique_id}")
            return technique

        except Exception as e:
            logger.exception(f"Failed to save technique {technique.technique_id}: {e!s}")
            msg = f"Failed to save technique: {e!s}"
            raise ValueError(msg)

    async def update_technique(
        self,
        technique_id: str,
        updates: dict[str, Any],
    ) -> JailbreakTemplate | None:
        """Update a technique."""
        try:
            # Load existing technique
            technique = await self.get_technique(technique_id)
            if not technique:
                return None

            # Apply updates
            for key, value in updates.items():
                if hasattr(technique, key):
                    setattr(technique, key, value)

            # Update timestamp
            technique.updated_at = datetime.utcnow()

            # Save updated technique
            await self.save_technique(technique)

            return technique

        except Exception as e:
            logger.exception(f"Failed to update technique {technique_id}: {e!s}")
            return None

    async def delete_technique(self, technique_id: str) -> bool:
        """Delete a technique."""
        try:
            # Delete YAML file
            yaml_file = self.techniques_directory / f"{technique_id}.yaml"
            if yaml_file.exists():
                yaml_file.unlink()

            # Delete JSON file (if exists)
            json_file = self.techniques_directory / f"{technique_id}.json"
            if json_file.exists():
                json_file.unlink()

            # Remove from cache
            self._techniques_cache.pop(technique_id, None)

            logger.info(f"Deleted technique: {technique_id}")
            return True

        except Exception as e:
            logger.exception(f"Failed to delete technique {technique_id}: {e!s}")
            return False

    async def get_techniques_by_category(self, category: str) -> list[JailbreakTemplate]:
        """Get all techniques in a category."""
        try:
            techniques = await self._load_all_techniques()
            return [t for t in techniques if t.category.value == category]

        except Exception as e:
            logger.exception(f"Failed to get techniques by category {category}: {e!s}")
            return []

    async def search_techniques(self, query: str, limit: int = 20) -> list[JailbreakTemplate]:
        """Search techniques by text query."""
        try:
            techniques = await self._load_all_techniques()

            # Search in name, description, tags
            matching_techniques = []
            query_lower = query.lower()

            for technique in techniques:
                # Check name
                if query_lower in technique.name.lower():
                    matching_techniques.append(technique)
                    continue

                # Check description
                if query_lower in technique.description.lower():
                    matching_techniques.append(technique)
                    continue

                # Check tags
                for tag in technique.tags:
                    if query_lower in tag.lower():
                        matching_techniques.append(technique)
                        break

            return matching_techniques[:limit]

        except Exception as e:
            logger.exception(f"Failed to search techniques: {e!s}")
            return []

    async def get_enabled_techniques(self) -> list[JailbreakTemplate]:
        """Get all enabled techniques."""
        try:
            techniques = await self._load_all_techniques()
            return [t for t in techniques if t.enabled and not t.deprecated]

        except Exception as e:
            logger.exception(f"Failed to get enabled techniques: {e!s}")
            return []

    async def get_technique_stats(self, technique_id: str) -> JailbreakExecutionStats | None:
        """Get execution statistics for a technique."""
        try:
            # Return cached stats if available
            if technique_id in self._stats_cache:
                return self._stats_cache[technique_id]

            # In a real implementation, this would query a database
            # For now, return default stats
            stats = JailbreakExecutionStats(
                technique_id=technique_id,
                total_executions=0,
                successful_executions=0,
                success_rate=0.0,
                avg_execution_time_ms=0.0,
                last_execution=None,
                current_concurrent_uses=0,
                error_rate=0.0,
            )

            self._stats_cache[technique_id] = stats
            return stats

        except Exception as e:
            logger.exception(f"Failed to get technique stats {technique_id}: {e!s}")
            return None

    async def _load_all_techniques(self) -> list[JailbreakTemplate]:
        """Load all techniques from files."""
        try:
            # Check cache first
            if self._is_cache_valid():
                return list(self._techniques_cache.values())

            # Load all technique files
            techniques = []

            for file_path in self.techniques_directory.glob("*.yaml"):
                try:
                    with open(file_path, encoding="utf-8") as f:
                        data = yaml.safe_load(f)
                        technique = JailbreakTemplate(**data)
                        techniques.append(technique)
                        self._techniques_cache[technique.technique_id] = technique

                except Exception as e:
                    logger.exception(f"Failed to load technique file {file_path}: {e!s}")

            # Also try JSON files
            for file_path in self.techniques_directory.glob("*.json"):
                if file_path.stem not in self._techniques_cache:
                    try:
                        with open(file_path, encoding="utf-8") as f:
                            data = json.load(f)
                            technique = JailbreakTemplate(**data)
                            techniques.append(technique)
                            self._techniques_cache[technique.technique_id] = technique

                    except Exception as e:
                        logger.exception(f"Failed to load technique file {file_path}: {e!s}")

            # Update cache timestamp
            self._cache_timestamp = datetime.utcnow()

            logger.info(f"Loaded {len(techniques)} techniques from files")
            return techniques

        except Exception as e:
            logger.exception(f"Failed to load techniques: {e!s}")
            return []

    async def _apply_filters(
        self,
        techniques: list[JailbreakTemplate],
        search_request: TechniqueSearchRequest,
    ) -> list[JailbreakTemplate]:
        """Apply filters to technique list."""
        filtered = techniques

        # Search query
        if search_request.query:
            query_lower = search_request.query.lower()
            filtered = [
                t
                for t in filtered
                if query_lower in t.name.lower()
                or query_lower in t.description.lower()
                or any(query_lower in tag.lower() for tag in t.tags)
            ]

        # Category filter
        if search_request.category:
            filtered = [t for t in filtered if t.category == search_request.category]

        # Risk level filter
        if search_request.risk_level:
            filtered = [t for t in filtered if t.risk_level == search_request.risk_level]

        # Complexity filter
        if search_request.complexity:
            filtered = [t for t in filtered if t.complexity == search_request.complexity]

        # Tags filter
        if search_request.tags:
            filtered = [t for t in filtered if any(tag in t.tags for tag in search_request.tags)]

        # Enabled only filter
        if search_request.enabled_only:
            filtered = [t for t in filtered if t.enabled]

        # Include deprecated filter
        if not search_request.include_deprecated:
            filtered = [t for t in filtered if not t.deprecated]

        return filtered

    async def _apply_sorting(
        self,
        techniques: list[JailbreakTemplate],
        search_request: TechniqueSearchRequest,
    ) -> list[JailbreakTemplate]:
        """Apply sorting to technique list."""
        if not search_request.sort_by:
            return techniques

        reverse = search_request.sort_order == "desc"

        if search_request.sort_by == "name":
            return sorted(techniques, key=lambda t: t.name.lower(), reverse=reverse)
        if search_request.sort_by == "created_at":
            return sorted(techniques, key=lambda t: t.created_at, reverse=reverse)
        if search_request.sort_by == "success_rate":
            return sorted(techniques, key=lambda t: t.success_rate_estimate, reverse=reverse)
        if search_request.sort_by == "risk_level":
            risk_order = {"low": 0, "medium": 1, "high": 2, "critical": 3}
            return sorted(
                techniques,
                key=lambda t: risk_order.get(t.risk_level.value, 0),
                reverse=reverse,
            )
        return techniques

    def _is_cache_valid(self) -> bool:
        """Check if cache is valid."""
        if not self._cache_timestamp:
            return False

        age_seconds = (datetime.utcnow() - self._cache_timestamp).total_seconds()
        return age_seconds < self._cache_ttl_seconds

    async def refresh_cache(self) -> None:
        """Force refresh of the techniques cache."""
        self._techniques_cache.clear()
        self._cache_timestamp = None
        await self._load_all_techniques()
        logger.info("Techniques cache refreshed")

    def get_cache_info(self) -> dict[str, Any]:
        """Get cache information."""
        return {
            "cached_techniques": len(self._techniques_cache),
            "cache_timestamp": self._cache_timestamp.isoformat() if self._cache_timestamp else None,
            "cache_age_seconds": (
                (datetime.utcnow() - self._cache_timestamp).total_seconds()
                if self._cache_timestamp
                else None
            ),
            "cache_valid": self._is_cache_valid(),
        }
