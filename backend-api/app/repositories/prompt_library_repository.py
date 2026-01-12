"""
Prompt Library Repository - In-Memory Storage with Optional JSON Persistence

This module provides a repository pattern implementation for prompt templates
with in-memory storage, supporting CRUD operations, search, filtering,
versioning, and ratings. Optionally persists data to JSON file for development.
"""

import json
import logging
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from app.domain.prompt_library_models import (
    PromptTemplate,
    RatingStatistics,
    SharingLevel,
    SortOrder,
    TemplateMetadata,
    TemplateRating,
    TemplateSearchFilters,
    TemplateSearchRequest,
    TemplateSearchResult,
    TemplateSortField,
    TemplateStatus,
    TemplateVersion,
    TechniqueType,
    VulnerabilityType,
    generate_rating_id,
    generate_template_id,
    generate_version_id,
)

logger = logging.getLogger(__name__)


class RepositoryError(Exception):
    """Base exception for repository operations."""


class EntityNotFoundError(RepositoryError):
    """Raised when entity is not found."""


class DuplicateEntityError(RepositoryError):
    """Raised when entity already exists."""


class ValidationError(RepositoryError):
    """Raised when validation fails."""


class PromptLibraryRepository:
    """
    In-memory repository for prompt templates with optional JSON persistence.

    Provides CRUD operations, search/filter, versioning, and rating support
    for prompt templates. Thread-safe for concurrent access.

    Attributes:
        persistence_path: Optional path to JSON file for persistence.
        auto_save: Whether to automatically save after each modification.
    """

    def __init__(
        self,
        persistence_path: str | Path | None = None,
        auto_save: bool = True,
    ):
        """
        Initialize the repository.

        Args:
            persistence_path: Optional path to JSON file for persistence.
            auto_save: Whether to automatically save after modifications.
        """
        self._templates: dict[str, PromptTemplate] = {}
        self._versions: dict[str, list[TemplateVersion]] = {}  # template_id -> versions
        self._ratings: dict[str, list[TemplateRating]] = {}  # template_id -> ratings
        self._user_ratings: dict[str, dict[str, str]] = {}  # user_id -> {template_id -> rating_id}

        self._lock = threading.RLock()
        self._persistence_path = Path(persistence_path) if persistence_path else None
        self._auto_save = auto_save

        # Load existing data if persistence file exists
        if self._persistence_path and self._persistence_path.exists():
            self._load_from_file()

        logger.info(
            f"PromptLibraryRepository initialized with "
            f"persistence={'enabled' if self._persistence_path else 'disabled'}"
        )

    # =========================================================================
    # Template CRUD Operations
    # =========================================================================

    def create_template(
        self,
        name: str,
        prompt_content: str,
        description: str = "",
        system_instruction: str | None = None,
        metadata: TemplateMetadata | None = None,
        status: TemplateStatus = TemplateStatus.ACTIVE,
        sharing_level: SharingLevel = SharingLevel.PRIVATE,
        created_by: str | None = None,
        team_id: str | None = None,
        source_campaign_id: str | None = None,
        source_execution_id: str | None = None,
    ) -> PromptTemplate:
        """
        Create a new prompt template.

        Args:
            name: Human-readable template name.
            prompt_content: The actual prompt content.
            description: Optional description of the template.
            system_instruction: Optional system instruction.
            metadata: Template metadata (techniques, vulnerabilities, tags).
            status: Template status.
            sharing_level: Visibility level.
            created_by: User ID of the creator.
            team_id: Team ID if team template.
            source_campaign_id: Campaign ID if saved from campaign.
            source_execution_id: Execution ID if saved from execution.

        Returns:
            Created PromptTemplate.

        Raises:
            ValidationError: If required fields are missing.
        """
        if not name or not name.strip():
            raise ValidationError("Template name is required")
        if not prompt_content or not prompt_content.strip():
            raise ValidationError("Prompt content is required")

        template_id = generate_template_id()
        now = datetime.utcnow()

        template = PromptTemplate(
            id=template_id,
            name=name.strip(),
            description=description,
            prompt_content=prompt_content,
            system_instruction=system_instruction,
            metadata=metadata or TemplateMetadata(),
            status=status,
            sharing_level=sharing_level,
            current_version=1,
            rating_stats=RatingStatistics(),
            created_by=created_by,
            team_id=team_id,
            created_at=now,
            updated_at=now,
            source_campaign_id=source_campaign_id,
            source_execution_id=source_execution_id,
        )

        with self._lock:
            self._templates[template_id] = template
            self._versions[template_id] = []
            self._ratings[template_id] = []

            # Create initial version
            initial_version = TemplateVersion(
                version_id=generate_version_id(),
                version_number=1,
                template_id=template_id,
                prompt_content=prompt_content,
                change_summary="Initial version",
                created_by=created_by,
                created_at=now,
                parent_version_id=None,
            )
            self._versions[template_id].append(initial_version)

            self._save_if_enabled()

        logger.info(f"Created template '{name}' with ID {template_id}")
        return template

    def get_template(self, template_id: str) -> PromptTemplate | None:
        """
        Get a template by ID.

        Args:
            template_id: Template identifier.

        Returns:
            PromptTemplate if found, None otherwise.
        """
        with self._lock:
            return self._templates.get(template_id)

    def get_template_or_raise(self, template_id: str) -> PromptTemplate:
        """
        Get a template by ID or raise exception.

        Args:
            template_id: Template identifier.

        Returns:
            PromptTemplate.

        Raises:
            EntityNotFoundError: If template not found.
        """
        template = self.get_template(template_id)
        if not template:
            raise EntityNotFoundError(f"Template '{template_id}' not found")
        return template

    def update_template(
        self,
        template_id: str,
        name: str | None = None,
        description: str | None = None,
        prompt_content: str | None = None,
        system_instruction: str | None = None,
        metadata: TemplateMetadata | None = None,
        status: TemplateStatus | None = None,
        sharing_level: SharingLevel | None = None,
        team_id: str | None = None,
        create_version: bool = True,
        change_summary: str = "",
        updated_by: str | None = None,
    ) -> PromptTemplate:
        """
        Update an existing template.

        Args:
            template_id: Template identifier.
            name: New name (optional).
            description: New description (optional).
            prompt_content: New prompt content (optional).
            system_instruction: New system instruction (optional).
            metadata: New metadata (optional).
            status: New status (optional).
            sharing_level: New sharing level (optional).
            team_id: New team ID (optional).
            create_version: Whether to create a new version if content changes.
            change_summary: Summary of changes for version.
            updated_by: User ID of the updater.

        Returns:
            Updated PromptTemplate.

        Raises:
            EntityNotFoundError: If template not found.
        """
        with self._lock:
            template = self.get_template_or_raise(template_id)
            now = datetime.utcnow()
            content_changed = False

            # Update fields if provided
            if name is not None:
                template.name = name.strip()
            if description is not None:
                template.description = description
            if prompt_content is not None and prompt_content != template.prompt_content:
                template.prompt_content = prompt_content
                content_changed = True
            if system_instruction is not None:
                template.system_instruction = system_instruction
            if metadata is not None:
                template.metadata = metadata
            if status is not None:
                template.status = status
            if sharing_level is not None:
                template.sharing_level = sharing_level
            if team_id is not None:
                template.team_id = team_id

            template.updated_at = now

            # Create new version if content changed
            if content_changed and create_version:
                previous_version = self._get_latest_version(template_id)
                new_version_number = template.current_version + 1

                new_version = TemplateVersion(
                    version_id=generate_version_id(),
                    version_number=new_version_number,
                    template_id=template_id,
                    prompt_content=template.prompt_content,
                    change_summary=change_summary or f"Version {new_version_number}",
                    created_by=updated_by,
                    created_at=now,
                    parent_version_id=previous_version.version_id if previous_version else None,
                )
                self._versions[template_id].append(new_version)
                template.current_version = new_version_number

            self._save_if_enabled()

        logger.info(f"Updated template '{template_id}'")
        return template

    def delete_template(self, template_id: str) -> bool:
        """
        Delete a template and all associated data.

        Args:
            template_id: Template identifier.

        Returns:
            True if deleted, False if not found.
        """
        with self._lock:
            if template_id not in self._templates:
                return False

            # Remove template
            del self._templates[template_id]

            # Remove versions
            if template_id in self._versions:
                del self._versions[template_id]

            # Remove ratings
            if template_id in self._ratings:
                # Also clean up user_ratings index
                for rating in self._ratings[template_id]:
                    if rating.user_id in self._user_ratings:
                        self._user_ratings[rating.user_id].pop(template_id, None)
                del self._ratings[template_id]

            self._save_if_enabled()

        logger.info(f"Deleted template '{template_id}'")
        return True

    def list_templates(
        self,
        limit: int = 100,
        offset: int = 0,
        sharing_levels: list[SharingLevel] | None = None,
        status: list[TemplateStatus] | None = None,
        user_id: str | None = None,
    ) -> list[PromptTemplate]:
        """
        List templates with basic filtering.

        Args:
            limit: Maximum number of templates to return.
            offset: Number of templates to skip.
            sharing_levels: Filter by sharing levels.
            status: Filter by status.
            user_id: Filter by creator.

        Returns:
            List of matching templates.
        """
        with self._lock:
            templates = list(self._templates.values())

        # Apply filters
        if sharing_levels:
            templates = [t for t in templates if t.sharing_level in sharing_levels]
        if status:
            templates = [t for t in templates if t.status in status]
        if user_id:
            templates = [t for t in templates if t.created_by == user_id]

        # Sort by updated_at descending
        templates.sort(key=lambda t: t.updated_at, reverse=True)

        # Apply pagination
        return templates[offset : offset + limit]

    def count_templates(
        self,
        sharing_levels: list[SharingLevel] | None = None,
        status: list[TemplateStatus] | None = None,
        user_id: str | None = None,
    ) -> int:
        """
        Count templates with basic filtering.

        Args:
            sharing_levels: Filter by sharing levels.
            status: Filter by status.
            user_id: Filter by creator.

        Returns:
            Count of matching templates.
        """
        return len(
            self.list_templates(
                limit=1000000,
                offset=0,
                sharing_levels=sharing_levels,
                status=status,
                user_id=user_id,
            )
        )

    def exists(self, template_id: str) -> bool:
        """Check if a template exists."""
        with self._lock:
            return template_id in self._templates

    # =========================================================================
    # Search and Filter Operations
    # =========================================================================

    def search_templates(
        self,
        request: TemplateSearchRequest,
        user_id: str | None = None,
        include_private: bool = False,
    ) -> TemplateSearchResult:
        """
        Search templates with advanced filtering, sorting, and pagination.

        Args:
            request: Search request with filters, sorting, and pagination.
            user_id: Current user ID (for visibility filtering).
            include_private: Whether to include private templates.

        Returns:
            TemplateSearchResult with matching templates and pagination info.
        """
        with self._lock:
            templates = list(self._templates.values())

        # Apply visibility filter
        if not include_private:
            templates = self._filter_by_visibility(templates, user_id)

        # Apply filters
        templates = self._apply_filters(templates, request.filters)

        # Get total count before pagination
        total_count = len(templates)

        # Apply sorting
        templates = self._apply_sorting(templates, request.sort_by, request.sort_order)

        # Apply pagination
        page = request.page
        page_size = request.page_size
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        paginated_templates = templates[start_idx:end_idx]

        # Calculate total pages
        total_pages = (total_count + page_size - 1) // page_size if total_count > 0 else 0

        return TemplateSearchResult(
            templates=paginated_templates,
            total_count=total_count,
            page=page,
            page_size=page_size,
            total_pages=total_pages,
        )

    def _filter_by_visibility(
        self,
        templates: list[PromptTemplate],
        user_id: str | None,
    ) -> list[PromptTemplate]:
        """Filter templates based on visibility rules."""
        result = []
        for template in templates:
            if template.sharing_level == SharingLevel.PUBLIC:
                result.append(template)
            elif template.sharing_level == SharingLevel.PRIVATE:
                if user_id and template.created_by == user_id:
                    result.append(template)
            elif template.sharing_level == SharingLevel.TEAM:
                # For team visibility, include if user is creator
                # (full team support would require team membership check)
                if user_id and template.created_by == user_id:
                    result.append(template)
        return result

    def _apply_filters(
        self,
        templates: list[PromptTemplate],
        filters: TemplateSearchFilters,
    ) -> list[PromptTemplate]:
        """Apply search filters to templates."""
        result = templates

        # Text search (name, description, content)
        if filters.query:
            query_lower = filters.query.lower()
            result = [
                t for t in result
                if query_lower in t.name.lower()
                or query_lower in t.description.lower()
                or query_lower in t.prompt_content.lower()
            ]

        # Technique types filter (OR logic)
        if filters.technique_types:
            technique_set = set(filters.technique_types)
            result = [
                t for t in result
                if any(tech in technique_set for tech in t.metadata.technique_types)
            ]

        # Vulnerability types filter (OR logic)
        if filters.vulnerability_types:
            vuln_set = set(filters.vulnerability_types)
            result = [
                t for t in result
                if any(vuln in vuln_set for vuln in t.metadata.vulnerability_types)
            ]

        # Target models filter (OR logic)
        if filters.target_models:
            model_set = set(m.lower() for m in filters.target_models)
            result = [
                t for t in result
                if any(m.lower() in model_set for m in t.metadata.target_models)
            ]

        # Target providers filter (OR logic)
        if filters.target_providers:
            provider_set = set(p.lower() for p in filters.target_providers)
            result = [
                t for t in result
                if any(p.lower() in provider_set for p in t.metadata.target_providers)
            ]

        # Tags filter (AND logic)
        if filters.tags:
            tag_set = set(tag.lower() for tag in filters.tags)
            result = [
                t for t in result
                if tag_set.issubset(set(tag.lower() for tag in t.metadata.tags))
            ]

        # Status filter
        if filters.status:
            result = [t for t in result if t.status in filters.status]

        # Sharing levels filter
        if filters.sharing_levels:
            result = [t for t in result if t.sharing_level in filters.sharing_levels]

        # Minimum rating filter
        if filters.min_rating is not None:
            result = [
                t for t in result
                if t.rating_stats.average_rating >= filters.min_rating
            ]

        # Minimum success rate filter
        if filters.min_success_rate is not None:
            result = [
                t for t in result
                if t.metadata.success_rate is not None
                and t.metadata.success_rate >= filters.min_success_rate
            ]

        # Created by filter
        if filters.created_by:
            result = [t for t in result if t.created_by == filters.created_by]

        # Team ID filter
        if filters.team_id:
            result = [t for t in result if t.team_id == filters.team_id]

        # Date filters
        if filters.created_after:
            result = [t for t in result if t.created_at >= filters.created_after]
        if filters.created_before:
            result = [t for t in result if t.created_at <= filters.created_before]

        return result

    def _apply_sorting(
        self,
        templates: list[PromptTemplate],
        sort_by: TemplateSortField,
        sort_order: SortOrder,
    ) -> list[PromptTemplate]:
        """Apply sorting to templates."""
        reverse = sort_order == SortOrder.DESC

        sort_keys: dict[TemplateSortField, Callable[[PromptTemplate], Any]] = {
            TemplateSortField.CREATED_AT: lambda t: t.created_at,
            TemplateSortField.UPDATED_AT: lambda t: t.updated_at,
            TemplateSortField.NAME: lambda t: t.name.lower(),
            TemplateSortField.RATING: lambda t: t.rating_stats.average_rating,
            TemplateSortField.SUCCESS_RATE: lambda t: t.metadata.success_rate or 0,
            TemplateSortField.TEST_COUNT: lambda t: t.metadata.test_count,
            TemplateSortField.RATING_COUNT: lambda t: t.rating_stats.total_ratings,
        }

        key_func = sort_keys.get(sort_by, sort_keys[TemplateSortField.UPDATED_AT])
        return sorted(templates, key=key_func, reverse=reverse)

    def get_top_rated_templates(
        self,
        limit: int = 10,
        min_ratings: int = 1,
        sharing_level: SharingLevel | None = SharingLevel.PUBLIC,
    ) -> list[PromptTemplate]:
        """
        Get top-rated templates.

        Args:
            limit: Maximum number of templates to return.
            min_ratings: Minimum number of ratings required.
            sharing_level: Filter by sharing level.

        Returns:
            List of top-rated templates.
        """
        with self._lock:
            templates = list(self._templates.values())

        # Filter by sharing level
        if sharing_level:
            templates = [t for t in templates if t.sharing_level == sharing_level]

        # Filter by minimum ratings
        templates = [t for t in templates if t.rating_stats.total_ratings >= min_ratings]

        # Sort by average rating descending
        templates.sort(key=lambda t: t.rating_stats.average_rating, reverse=True)

        return templates[:limit]

    # =========================================================================
    # Version Operations
    # =========================================================================

    def get_versions(
        self,
        template_id: str,
        limit: int = 100,
        offset: int = 0,
    ) -> list[TemplateVersion]:
        """
        Get version history for a template.

        Args:
            template_id: Template identifier.
            limit: Maximum number of versions to return.
            offset: Number of versions to skip.

        Returns:
            List of versions, newest first.

        Raises:
            EntityNotFoundError: If template not found.
        """
        with self._lock:
            if template_id not in self._templates:
                raise EntityNotFoundError(f"Template '{template_id}' not found")

            versions = self._versions.get(template_id, [])
            # Sort by version number descending
            sorted_versions = sorted(versions, key=lambda v: v.version_number, reverse=True)
            return sorted_versions[offset : offset + limit]

    def get_version(
        self,
        template_id: str,
        version_number: int | None = None,
        version_id: str | None = None,
    ) -> TemplateVersion | None:
        """
        Get a specific version of a template.

        Args:
            template_id: Template identifier.
            version_number: Version number to retrieve.
            version_id: Version ID to retrieve.

        Returns:
            TemplateVersion if found, None otherwise.
        """
        with self._lock:
            versions = self._versions.get(template_id, [])

            if version_id:
                for version in versions:
                    if version.version_id == version_id:
                        return version
            elif version_number:
                for version in versions:
                    if version.version_number == version_number:
                        return version

            return None

    def _get_latest_version(self, template_id: str) -> TemplateVersion | None:
        """Get the latest version for a template."""
        versions = self._versions.get(template_id, [])
        if not versions:
            return None
        return max(versions, key=lambda v: v.version_number)

    def create_version(
        self,
        template_id: str,
        prompt_content: str,
        change_summary: str = "",
        created_by: str | None = None,
    ) -> TemplateVersion:
        """
        Create a new version for a template.

        Args:
            template_id: Template identifier.
            prompt_content: New prompt content.
            change_summary: Summary of changes.
            created_by: User ID of the creator.

        Returns:
            Created TemplateVersion.

        Raises:
            EntityNotFoundError: If template not found.
        """
        with self._lock:
            template = self.get_template_or_raise(template_id)
            previous_version = self._get_latest_version(template_id)

            now = datetime.utcnow()
            new_version_number = template.current_version + 1

            new_version = TemplateVersion(
                version_id=generate_version_id(),
                version_number=new_version_number,
                template_id=template_id,
                prompt_content=prompt_content,
                change_summary=change_summary or f"Version {new_version_number}",
                created_by=created_by,
                created_at=now,
                parent_version_id=previous_version.version_id if previous_version else None,
            )

            self._versions[template_id].append(new_version)
            template.current_version = new_version_number
            template.prompt_content = prompt_content
            template.updated_at = now

            self._save_if_enabled()

        logger.info(f"Created version {new_version_number} for template '{template_id}'")
        return new_version

    def restore_version(
        self,
        template_id: str,
        version_number: int,
        restored_by: str | None = None,
    ) -> PromptTemplate:
        """
        Restore a template to a previous version.

        Args:
            template_id: Template identifier.
            version_number: Version number to restore.
            restored_by: User ID of the restorer.

        Returns:
            Updated PromptTemplate.

        Raises:
            EntityNotFoundError: If template or version not found.
        """
        with self._lock:
            template = self.get_template_or_raise(template_id)
            version = self.get_version(template_id, version_number=version_number)

            if not version:
                raise EntityNotFoundError(
                    f"Version {version_number} not found for template '{template_id}'"
                )

            # Create new version with restored content
            self.create_version(
                template_id=template_id,
                prompt_content=version.prompt_content,
                change_summary=f"Restored from version {version_number}",
                created_by=restored_by,
            )

        return self.get_template_or_raise(template_id)

    def count_versions(self, template_id: str) -> int:
        """Count versions for a template."""
        with self._lock:
            return len(self._versions.get(template_id, []))

    # =========================================================================
    # Rating Operations
    # =========================================================================

    def add_rating(
        self,
        template_id: str,
        user_id: str,
        rating: int,
        effectiveness_score: int | None = None,
        comment: str | None = None,
        reported_success: bool | None = None,
        target_model_tested: str | None = None,
    ) -> TemplateRating:
        """
        Add or update a rating for a template.

        Args:
            template_id: Template identifier.
            user_id: User ID.
            rating: Star rating (1-5).
            effectiveness_score: Effectiveness score (1-5).
            comment: Optional comment.
            reported_success: Whether the template worked.
            target_model_tested: Model tested against.

        Returns:
            Created or updated TemplateRating.

        Raises:
            EntityNotFoundError: If template not found.
            ValidationError: If rating is invalid.
        """
        if rating < 1 or rating > 5:
            raise ValidationError("Rating must be between 1 and 5")
        if effectiveness_score is not None and (effectiveness_score < 1 or effectiveness_score > 5):
            raise ValidationError("Effectiveness score must be between 1 and 5")

        with self._lock:
            if template_id not in self._templates:
                raise EntityNotFoundError(f"Template '{template_id}' not found")

            now = datetime.utcnow()

            # Check if user already rated this template
            existing_rating_id = None
            if user_id in self._user_ratings:
                existing_rating_id = self._user_ratings[user_id].get(template_id)

            if existing_rating_id:
                # Update existing rating
                for r in self._ratings[template_id]:
                    if r.rating_id == existing_rating_id:
                        r.rating = rating
                        r.effectiveness_score = effectiveness_score
                        r.comment = comment
                        r.reported_success = reported_success
                        r.target_model_tested = target_model_tested
                        r.updated_at = now
                        new_rating = r
                        break
            else:
                # Create new rating
                new_rating = TemplateRating(
                    rating_id=generate_rating_id(),
                    template_id=template_id,
                    user_id=user_id,
                    rating=rating,
                    effectiveness_score=effectiveness_score,
                    comment=comment,
                    reported_success=reported_success,
                    target_model_tested=target_model_tested,
                    created_at=now,
                    updated_at=now,
                )
                self._ratings[template_id].append(new_rating)

                # Update user ratings index
                if user_id not in self._user_ratings:
                    self._user_ratings[user_id] = {}
                self._user_ratings[user_id][template_id] = new_rating.rating_id

            # Recalculate rating statistics
            self._recalculate_rating_stats(template_id)
            self._save_if_enabled()

        logger.info(f"User '{user_id}' rated template '{template_id}' with {rating} stars")
        return new_rating

    def get_rating(
        self,
        template_id: str,
        user_id: str,
    ) -> TemplateRating | None:
        """
        Get a user's rating for a template.

        Args:
            template_id: Template identifier.
            user_id: User ID.

        Returns:
            TemplateRating if found, None otherwise.
        """
        with self._lock:
            rating_id = self._user_ratings.get(user_id, {}).get(template_id)
            if rating_id:
                for r in self._ratings.get(template_id, []):
                    if r.rating_id == rating_id:
                        return r
            return None

    def get_ratings(
        self,
        template_id: str,
        limit: int = 100,
        offset: int = 0,
    ) -> list[TemplateRating]:
        """
        Get ratings for a template.

        Args:
            template_id: Template identifier.
            limit: Maximum number of ratings to return.
            offset: Number of ratings to skip.

        Returns:
            List of ratings, newest first.

        Raises:
            EntityNotFoundError: If template not found.
        """
        with self._lock:
            if template_id not in self._templates:
                raise EntityNotFoundError(f"Template '{template_id}' not found")

            ratings = self._ratings.get(template_id, [])
            # Sort by created_at descending
            sorted_ratings = sorted(ratings, key=lambda r: r.created_at, reverse=True)
            return sorted_ratings[offset : offset + limit]

    def delete_rating(
        self,
        template_id: str,
        user_id: str,
    ) -> bool:
        """
        Delete a user's rating for a template.

        Args:
            template_id: Template identifier.
            user_id: User ID.

        Returns:
            True if deleted, False if not found.
        """
        with self._lock:
            rating_id = self._user_ratings.get(user_id, {}).get(template_id)
            if not rating_id:
                return False

            # Remove from ratings list
            ratings = self._ratings.get(template_id, [])
            self._ratings[template_id] = [r for r in ratings if r.rating_id != rating_id]

            # Remove from user ratings index
            del self._user_ratings[user_id][template_id]

            # Recalculate stats
            self._recalculate_rating_stats(template_id)
            self._save_if_enabled()

        logger.info(f"Deleted rating from user '{user_id}' for template '{template_id}'")
        return True

    def get_rating_statistics(self, template_id: str) -> RatingStatistics:
        """
        Get rating statistics for a template.

        Args:
            template_id: Template identifier.

        Returns:
            RatingStatistics for the template.

        Raises:
            EntityNotFoundError: If template not found.
        """
        with self._lock:
            template = self.get_template_or_raise(template_id)
            return template.rating_stats

    def _recalculate_rating_stats(self, template_id: str) -> None:
        """Recalculate rating statistics for a template."""
        template = self._templates.get(template_id)
        if not template:
            return

        ratings = self._ratings.get(template_id, [])

        if not ratings:
            template.rating_stats = RatingStatistics()
            return

        # Calculate statistics
        total_ratings = len(ratings)
        sum_ratings = sum(r.rating for r in ratings)
        average_rating = sum_ratings / total_ratings if total_ratings > 0 else 0.0

        # Effectiveness scores
        effectiveness_ratings = [r for r in ratings if r.effectiveness_score is not None]
        if effectiveness_ratings:
            avg_effectiveness = sum(r.effectiveness_score for r in effectiveness_ratings) / len(effectiveness_ratings)
        else:
            avg_effectiveness = None

        # Success/failure counts
        success_count = sum(1 for r in ratings if r.reported_success is True)
        failure_count = sum(1 for r in ratings if r.reported_success is False)

        # Rating distribution
        distribution = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        for r in ratings:
            distribution[r.rating] += 1

        template.rating_stats = RatingStatistics(
            total_ratings=total_ratings,
            average_rating=round(average_rating, 2),
            average_effectiveness=round(avg_effectiveness, 2) if avg_effectiveness else None,
            success_count=success_count,
            failure_count=failure_count,
            rating_distribution=distribution,
        )

    def count_ratings(self, template_id: str) -> int:
        """Count ratings for a template."""
        with self._lock:
            return len(self._ratings.get(template_id, []))

    # =========================================================================
    # Persistence Operations
    # =========================================================================

    def _save_if_enabled(self) -> None:
        """Save to file if auto-save is enabled."""
        if self._auto_save and self._persistence_path:
            self._save_to_file()

    def _save_to_file(self) -> None:
        """Save repository data to JSON file."""
        if not self._persistence_path:
            return

        try:
            data = {
                "templates": {
                    tid: t.model_dump(mode="json") for tid, t in self._templates.items()
                },
                "versions": {
                    tid: [v.model_dump(mode="json") for v in versions]
                    for tid, versions in self._versions.items()
                },
                "ratings": {
                    tid: [r.model_dump(mode="json") for r in ratings]
                    for tid, ratings in self._ratings.items()
                },
                "user_ratings": self._user_ratings,
                "saved_at": datetime.utcnow().isoformat(),
            }

            # Ensure directory exists
            self._persistence_path.parent.mkdir(parents=True, exist_ok=True)

            # Write atomically with temp file
            temp_path = self._persistence_path.with_suffix(".tmp")
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, default=str)
            temp_path.replace(self._persistence_path)

            logger.debug(f"Saved repository to {self._persistence_path}")
        except Exception as e:
            logger.error(f"Failed to save repository: {e}")

    def _load_from_file(self) -> None:
        """Load repository data from JSON file."""
        if not self._persistence_path or not self._persistence_path.exists():
            return

        try:
            with open(self._persistence_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Load templates
            for tid, template_data in data.get("templates", {}).items():
                self._templates[tid] = PromptTemplate.model_validate(template_data)

            # Load versions
            for tid, versions_data in data.get("versions", {}).items():
                self._versions[tid] = [
                    TemplateVersion.model_validate(v) for v in versions_data
                ]

            # Load ratings
            for tid, ratings_data in data.get("ratings", {}).items():
                self._ratings[tid] = [
                    TemplateRating.model_validate(r) for r in ratings_data
                ]

            # Load user ratings index
            self._user_ratings = data.get("user_ratings", {})

            logger.info(
                f"Loaded {len(self._templates)} templates from {self._persistence_path}"
            )
        except Exception as e:
            logger.error(f"Failed to load repository: {e}")

    def save(self) -> None:
        """Manually save repository to file."""
        self._save_to_file()

    def clear(self) -> None:
        """Clear all data from repository."""
        with self._lock:
            self._templates.clear()
            self._versions.clear()
            self._ratings.clear()
            self._user_ratings.clear()
            self._save_if_enabled()
        logger.info("Cleared all repository data")

    # =========================================================================
    # Statistics and Analytics
    # =========================================================================

    def get_statistics(self) -> dict[str, Any]:
        """
        Get repository statistics.

        Returns:
            Dictionary with repository statistics.
        """
        with self._lock:
            templates = list(self._templates.values())

            # Count by sharing level
            by_sharing = {level.value: 0 for level in SharingLevel}
            for t in templates:
                by_sharing[t.sharing_level.value] += 1

            # Count by status
            by_status = {status.value: 0 for status in TemplateStatus}
            for t in templates:
                by_status[t.status.value] += 1

            # Count techniques used
            technique_counts: dict[str, int] = {}
            for t in templates:
                for tech in t.metadata.technique_types:
                    technique_counts[tech.value] = technique_counts.get(tech.value, 0) + 1

            # Rating statistics
            total_ratings = sum(len(r) for r in self._ratings.values())
            total_versions = sum(len(v) for v in self._versions.values())

            return {
                "total_templates": len(templates),
                "by_sharing_level": by_sharing,
                "by_status": by_status,
                "top_techniques": dict(
                    sorted(technique_counts.items(), key=lambda x: x[1], reverse=True)[:10]
                ),
                "total_ratings": total_ratings,
                "total_versions": total_versions,
                "persistence_enabled": self._persistence_path is not None,
            }


# Singleton instance
_repository_instance: PromptLibraryRepository | None = None
_repository_lock = threading.Lock()


def get_prompt_library_repository(
    persistence_path: str | Path | None = None,
    reset: bool = False,
) -> PromptLibraryRepository:
    """
    Get the singleton repository instance.

    Args:
        persistence_path: Optional path to JSON file for persistence.
        reset: Whether to reset the singleton instance.

    Returns:
        PromptLibraryRepository instance.
    """
    global _repository_instance

    with _repository_lock:
        if reset or _repository_instance is None:
            _repository_instance = PromptLibraryRepository(
                persistence_path=persistence_path
            )

    return _repository_instance
