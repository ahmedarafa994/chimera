"""
Prompt Library Service - Business Logic Layer for Template Management.

This module implements the service layer for prompt template operations,
providing a clean interface between API endpoints and the repository layer.
Includes template CRUD, search/filter, campaign saves, versioning, and statistics.
"""

import logging
from datetime import datetime
from typing import Any

from app.domain.prompt_library_models import (
    PromptTemplate,
    RatingStatistics,
    SaveFromCampaignRequest,
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
)
from app.repositories.prompt_library_repository import (
    DuplicateEntityError,
    EntityNotFoundError,
    PromptLibraryRepository,
    RepositoryError,
    ValidationError,
    get_prompt_library_repository,
)

logger = logging.getLogger(__name__)


class PromptLibraryServiceError(Exception):
    """Base exception for prompt library service operations."""

    def __init__(self, message: str, details: dict[str, Any] | None = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}


class TemplateNotFoundError(PromptLibraryServiceError):
    """Raised when a template is not found."""


class TemplateValidationError(PromptLibraryServiceError):
    """Raised when template validation fails."""


class TemplatePermissionError(PromptLibraryServiceError):
    """Raised when user lacks permission for an operation."""


class PromptLibraryService:
    """
    Service layer for prompt template management.

    Provides business logic for template CRUD operations, search/filtering,
    campaign integration, version control, and analytics. Wraps the repository
    layer with additional validation, authorization, and business rules.

    Attributes:
        repository: The underlying repository for data persistence.
        default_page_size: Default number of results per page for list operations.
    """

    def __init__(
        self,
        repository: PromptLibraryRepository | None = None,
        default_page_size: int = 20,
    ):
        """
        Initialize the prompt library service.

        Args:
            repository: Repository instance (uses singleton if None).
            default_page_size: Default page size for list operations.
        """
        self._repository = repository
        self._default_page_size = default_page_size
        self._operation_count = 0
        self._cache_hits = 0
        self._cache_misses = 0

        logger.info("PromptLibraryService initialized")

    @property
    def repository(self) -> PromptLibraryRepository:
        """Get the repository instance (lazy initialization)."""
        if self._repository is None:
            self._repository = get_prompt_library_repository()
        return self._repository

    # =========================================================================
    # Template CRUD Operations
    # =========================================================================

    def create_template(
        self,
        name: str,
        prompt_content: str,
        description: str = "",
        system_instruction: str | None = None,
        technique_types: list[TechniqueType] | None = None,
        vulnerability_types: list[VulnerabilityType] | None = None,
        target_models: list[str] | None = None,
        target_providers: list[str] | None = None,
        cve_references: list[str] | None = None,
        paper_references: list[str] | None = None,
        tags: list[str] | None = None,
        discovery_source: str | None = None,
        status: TemplateStatus = TemplateStatus.ACTIVE,
        sharing_level: SharingLevel = SharingLevel.PRIVATE,
        created_by: str | None = None,
        team_id: str | None = None,
    ) -> PromptTemplate:
        """
        Create a new prompt template.

        Args:
            name: Human-readable template name.
            prompt_content: The actual prompt content.
            description: Optional description of the template.
            system_instruction: Optional system instruction.
            technique_types: List of techniques used.
            vulnerability_types: List of targeted vulnerabilities.
            target_models: List of models this works against.
            target_providers: List of target providers.
            cve_references: Related CVE references.
            paper_references: Research paper references.
            tags: Custom organization tags.
            discovery_source: Source of discovery.
            status: Template status.
            sharing_level: Visibility level.
            created_by: User ID of the creator.
            team_id: Team ID for team templates.

        Returns:
            Created PromptTemplate.

        Raises:
            TemplateValidationError: If validation fails.
        """
        self._operation_count += 1

        # Validate inputs
        if not name or not name.strip():
            raise TemplateValidationError(
                "Template name is required",
                details={"field": "name"},
            )
        if not prompt_content or not prompt_content.strip():
            raise TemplateValidationError(
                "Prompt content is required",
                details={"field": "prompt_content"},
            )

        # Validate team sharing requires team_id
        if sharing_level == SharingLevel.TEAM and not team_id:
            raise TemplateValidationError(
                "Team sharing level requires a team_id",
                details={"field": "team_id", "sharing_level": sharing_level.value},
            )

        # Build metadata
        metadata = TemplateMetadata(
            technique_types=technique_types or [],
            vulnerability_types=vulnerability_types or [],
            target_models=target_models or [],
            target_providers=target_providers or [],
            cve_references=cve_references or [],
            paper_references=paper_references or [],
            tags=tags or [],
            discovery_source=discovery_source,
            discovery_date=datetime.utcnow() if discovery_source else None,
        )

        try:
            template = self.repository.create_template(
                name=name,
                prompt_content=prompt_content,
                description=description,
                system_instruction=system_instruction,
                metadata=metadata,
                status=status,
                sharing_level=sharing_level,
                created_by=created_by,
                team_id=team_id,
            )

            logger.info(
                f"Created template '{template.id}' with name '{name}' "
                f"by user '{created_by or 'anonymous'}'"
            )
            return template

        except ValidationError as e:
            raise TemplateValidationError(str(e)) from e
        except RepositoryError as e:
            raise PromptLibraryServiceError(
                f"Failed to create template: {e}",
                details={"name": name},
            ) from e

    def get_template(
        self,
        template_id: str,
        user_id: str | None = None,
        check_visibility: bool = True,
    ) -> PromptTemplate:
        """
        Get a template by ID.

        Args:
            template_id: Template identifier.
            user_id: Current user ID for visibility check.
            check_visibility: Whether to enforce visibility rules.

        Returns:
            PromptTemplate.

        Raises:
            TemplateNotFoundError: If template not found.
            TemplatePermissionError: If user lacks access.
        """
        self._operation_count += 1

        try:
            template = self.repository.get_template_or_raise(template_id)

            # Check visibility permissions
            if check_visibility and not self._can_view_template(template, user_id):
                raise TemplatePermissionError(
                    f"Access denied to template '{template_id}'",
                    details={"template_id": template_id, "user_id": user_id},
                )

            return template

        except EntityNotFoundError as e:
            raise TemplateNotFoundError(
                f"Template '{template_id}' not found",
                details={"template_id": template_id},
            ) from e

    def update_template(
        self,
        template_id: str,
        user_id: str | None = None,
        name: str | None = None,
        description: str | None = None,
        prompt_content: str | None = None,
        system_instruction: str | None = None,
        technique_types: list[TechniqueType] | None = None,
        vulnerability_types: list[VulnerabilityType] | None = None,
        target_models: list[str] | None = None,
        target_providers: list[str] | None = None,
        cve_references: list[str] | None = None,
        paper_references: list[str] | None = None,
        tags: list[str] | None = None,
        success_rate: float | None = None,
        status: TemplateStatus | None = None,
        sharing_level: SharingLevel | None = None,
        team_id: str | None = None,
        create_version: bool = True,
        change_summary: str = "",
    ) -> PromptTemplate:
        """
        Update an existing template.

        Args:
            template_id: Template identifier.
            user_id: User performing the update.
            name: New name (optional).
            description: New description (optional).
            prompt_content: New content (optional).
            system_instruction: New system instruction (optional).
            technique_types: New technique types (optional).
            vulnerability_types: New vulnerability types (optional).
            target_models: New target models (optional).
            target_providers: New target providers (optional).
            cve_references: New CVE references (optional).
            paper_references: New paper references (optional).
            tags: New tags (optional).
            success_rate: New success rate (optional).
            status: New status (optional).
            sharing_level: New sharing level (optional).
            team_id: New team ID (optional).
            create_version: Create version if content changes.
            change_summary: Version change summary.

        Returns:
            Updated PromptTemplate.

        Raises:
            TemplateNotFoundError: If template not found.
            TemplatePermissionError: If user lacks permission.
            TemplateValidationError: If validation fails.
        """
        self._operation_count += 1

        # Get existing template and check permissions
        template = self.get_template(template_id, user_id, check_visibility=True)

        if not self._can_edit_template(template, user_id):
            raise TemplatePermissionError(
                f"User lacks permission to edit template '{template_id}'",
                details={"template_id": template_id, "user_id": user_id},
            )

        # Validate inputs
        if name is not None and not name.strip():
            raise TemplateValidationError(
                "Template name cannot be empty",
                details={"field": "name"},
            )
        if prompt_content is not None and not prompt_content.strip():
            raise TemplateValidationError(
                "Prompt content cannot be empty",
                details={"field": "prompt_content"},
            )

        # Build updated metadata if any metadata fields changed
        metadata = None
        if any([
            technique_types is not None,
            vulnerability_types is not None,
            target_models is not None,
            target_providers is not None,
            cve_references is not None,
            paper_references is not None,
            tags is not None,
            success_rate is not None,
        ]):
            current_metadata = template.metadata
            metadata = TemplateMetadata(
                technique_types=(
                    technique_types
                    if technique_types is not None
                    else current_metadata.technique_types
                ),
                vulnerability_types=(
                    vulnerability_types
                    if vulnerability_types is not None
                    else current_metadata.vulnerability_types
                ),
                target_models=(
                    target_models
                    if target_models is not None
                    else current_metadata.target_models
                ),
                target_providers=(
                    target_providers
                    if target_providers is not None
                    else current_metadata.target_providers
                ),
                cve_references=(
                    cve_references
                    if cve_references is not None
                    else current_metadata.cve_references
                ),
                paper_references=(
                    paper_references
                    if paper_references is not None
                    else current_metadata.paper_references
                ),
                tags=tags if tags is not None else current_metadata.tags,
                success_rate=(
                    success_rate
                    if success_rate is not None
                    else current_metadata.success_rate
                ),
                test_count=current_metadata.test_count,
                discovery_date=current_metadata.discovery_date,
                discovery_source=current_metadata.discovery_source,
                extra=current_metadata.extra,
            )

        try:
            updated = self.repository.update_template(
                template_id=template_id,
                name=name,
                description=description,
                prompt_content=prompt_content,
                system_instruction=system_instruction,
                metadata=metadata,
                status=status,
                sharing_level=sharing_level,
                team_id=team_id,
                create_version=create_version,
                change_summary=change_summary,
                updated_by=user_id,
            )

            logger.info(
                f"Updated template '{template_id}' by user '{user_id or 'anonymous'}'"
            )
            return updated

        except ValidationError as e:
            raise TemplateValidationError(str(e)) from e
        except RepositoryError as e:
            raise PromptLibraryServiceError(
                f"Failed to update template: {e}",
                details={"template_id": template_id},
            ) from e

    def delete_template(
        self,
        template_id: str,
        user_id: str | None = None,
    ) -> bool:
        """
        Delete a template.

        Args:
            template_id: Template identifier.
            user_id: User performing the deletion.

        Returns:
            True if deleted.

        Raises:
            TemplateNotFoundError: If template not found.
            TemplatePermissionError: If user lacks permission.
        """
        self._operation_count += 1

        # Get existing template and check permissions
        template = self.get_template(template_id, user_id, check_visibility=True)

        if not self._can_delete_template(template, user_id):
            raise TemplatePermissionError(
                f"User lacks permission to delete template '{template_id}'",
                details={"template_id": template_id, "user_id": user_id},
            )

        success = self.repository.delete_template(template_id)

        if success:
            logger.info(
                f"Deleted template '{template_id}' by user '{user_id or 'anonymous'}'"
            )

        return success

    def list_templates(
        self,
        user_id: str | None = None,
        limit: int | None = None,
        offset: int = 0,
        sharing_levels: list[SharingLevel] | None = None,
        status: list[TemplateStatus] | None = None,
        created_by: str | None = None,
        include_private: bool = False,
    ) -> list[PromptTemplate]:
        """
        List templates with basic filtering.

        Args:
            user_id: Current user ID for visibility filtering.
            limit: Maximum number of templates to return.
            offset: Number of templates to skip.
            sharing_levels: Filter by sharing levels.
            status: Filter by status.
            created_by: Filter by creator.
            include_private: Whether to include private templates.

        Returns:
            List of matching templates.
        """
        self._operation_count += 1
        limit = limit or self._default_page_size

        templates = self.repository.list_templates(
            limit=limit,
            offset=offset,
            sharing_levels=sharing_levels,
            status=status,
            user_id=created_by,
        )

        # Apply visibility filtering if not including private
        if not include_private:
            templates = [
                t for t in templates
                if self._can_view_template(t, user_id)
            ]

        return templates

    # =========================================================================
    # Search and Filter Operations
    # =========================================================================

    def search_templates(
        self,
        query: str | None = None,
        technique_types: list[TechniqueType] | None = None,
        vulnerability_types: list[VulnerabilityType] | None = None,
        target_models: list[str] | None = None,
        target_providers: list[str] | None = None,
        tags: list[str] | None = None,
        status: list[TemplateStatus] | None = None,
        sharing_levels: list[SharingLevel] | None = None,
        min_rating: float | None = None,
        min_success_rate: float | None = None,
        created_by: str | None = None,
        team_id: str | None = None,
        created_after: datetime | None = None,
        created_before: datetime | None = None,
        sort_by: str = "updated_at",
        sort_order: str = "desc",
        page: int = 1,
        page_size: int | None = None,
        user_id: str | None = None,
        include_private: bool = False,
    ) -> TemplateSearchResult:
        """
        Search templates with advanced filtering.

        Args:
            query: Full-text search query.
            technique_types: Filter by techniques (OR logic).
            vulnerability_types: Filter by vulnerabilities (OR logic).
            target_models: Filter by target models (OR logic).
            target_providers: Filter by providers (OR logic).
            tags: Filter by tags (AND logic).
            status: Filter by status.
            sharing_levels: Filter by sharing levels.
            min_rating: Minimum average rating.
            min_success_rate: Minimum success rate.
            created_by: Filter by creator.
            team_id: Filter by team.
            created_after: Created after date.
            created_before: Created before date.
            sort_by: Field to sort by.
            sort_order: Sort order (asc/desc).
            page: Page number (1-indexed).
            page_size: Results per page.
            user_id: Current user for visibility filtering.
            include_private: Include private templates.

        Returns:
            TemplateSearchResult with matching templates.
        """
        self._operation_count += 1

        # Build filters
        filters = TemplateSearchFilters(
            query=query,
            technique_types=technique_types,
            vulnerability_types=vulnerability_types,
            target_models=target_models,
            target_providers=target_providers,
            tags=tags,
            status=status,
            sharing_levels=sharing_levels,
            min_rating=min_rating,
            min_success_rate=min_success_rate,
            created_by=created_by,
            team_id=team_id,
            created_after=created_after,
            created_before=created_before,
        )

        # Map string sort_by to enum
        sort_field_map = {
            "created_at": TemplateSortField.CREATED_AT,
            "updated_at": TemplateSortField.UPDATED_AT,
            "name": TemplateSortField.NAME,
            "rating": TemplateSortField.RATING,
            "success_rate": TemplateSortField.SUCCESS_RATE,
            "test_count": TemplateSortField.TEST_COUNT,
            "rating_count": TemplateSortField.RATING_COUNT,
        }
        sort_by_enum = sort_field_map.get(sort_by, TemplateSortField.UPDATED_AT)
        sort_order_enum = SortOrder.ASC if sort_order == "asc" else SortOrder.DESC

        # Build search request
        request = TemplateSearchRequest(
            filters=filters,
            sort_by=sort_by_enum,
            sort_order=sort_order_enum,
            page=page,
            page_size=page_size or self._default_page_size,
        )

        # Execute search
        result = self.repository.search_templates(
            request=request,
            user_id=user_id,
            include_private=include_private,
        )

        logger.debug(
            f"Search returned {result.total_count} results "
            f"(page {result.page}/{result.total_pages})"
        )

        return result

    def get_top_rated_templates(
        self,
        limit: int = 10,
        min_ratings: int = 1,
        sharing_level: SharingLevel | None = SharingLevel.PUBLIC,
    ) -> list[PromptTemplate]:
        """
        Get top-rated templates.

        Args:
            limit: Maximum number of templates.
            min_ratings: Minimum number of ratings required.
            sharing_level: Filter by sharing level.

        Returns:
            List of top-rated templates.
        """
        self._operation_count += 1

        return self.repository.get_top_rated_templates(
            limit=limit,
            min_ratings=min_ratings,
            sharing_level=sharing_level,
        )

    # =========================================================================
    # Save From Campaign
    # =========================================================================

    def save_from_campaign(
        self,
        name: str,
        prompt_content: str,
        description: str = "",
        system_instruction: str | None = None,
        technique_types: list[TechniqueType] | None = None,
        vulnerability_types: list[VulnerabilityType] | None = None,
        target_model: str | None = None,
        target_provider: str | None = None,
        tags: list[str] | None = None,
        sharing_level: SharingLevel = SharingLevel.PRIVATE,
        team_id: str | None = None,
        campaign_id: str | None = None,
        execution_id: str | None = None,
        was_successful: bool = True,
        initial_success_rate: float | None = None,
        created_by: str | None = None,
    ) -> PromptTemplate:
        """
        Save a successful prompt from a campaign execution to the library.

        This method provides a convenient way to capture effective prompts
        discovered during campaign testing with auto-populated metadata.

        Args:
            name: Template name.
            prompt_content: The prompt content to save.
            description: Template description.
            system_instruction: Optional system instruction.
            technique_types: Techniques used (auto-populated from campaign).
            vulnerability_types: Vulnerabilities targeted.
            target_model: Model tested against.
            target_provider: Provider tested against.
            tags: Custom tags.
            sharing_level: Visibility level.
            team_id: Team ID for team sharing.
            campaign_id: Source campaign ID.
            execution_id: Source execution ID.
            was_successful: Whether the prompt was successful.
            initial_success_rate: Initial success rate from campaign.
            created_by: User ID of the creator.

        Returns:
            Created PromptTemplate.

        Raises:
            TemplateValidationError: If validation fails.
        """
        self._operation_count += 1

        # Build target lists from single values
        target_models = [target_model] if target_model else []
        target_providers = [target_provider] if target_provider else []

        # Add campaign-specific tags
        campaign_tags = list(tags or [])
        if campaign_id:
            campaign_tags.append("from-campaign")
        if was_successful:
            campaign_tags.append("verified-success")

        # Build metadata
        metadata = TemplateMetadata(
            technique_types=technique_types or [],
            vulnerability_types=vulnerability_types or [],
            target_models=target_models,
            target_providers=target_providers,
            tags=campaign_tags,
            success_rate=initial_success_rate,
            test_count=1 if was_successful else 0,
            discovery_date=datetime.utcnow(),
            discovery_source=f"Campaign: {campaign_id}" if campaign_id else None,
        )

        try:
            template = self.repository.create_template(
                name=name,
                prompt_content=prompt_content,
                description=description,
                system_instruction=system_instruction,
                metadata=metadata,
                status=TemplateStatus.ACTIVE,
                sharing_level=sharing_level,
                created_by=created_by,
                team_id=team_id,
                source_campaign_id=campaign_id,
                source_execution_id=execution_id,
            )

            logger.info(
                f"Saved prompt from campaign '{campaign_id}' as template '{template.id}' "
                f"(success={was_successful})"
            )
            return template

        except ValidationError as e:
            raise TemplateValidationError(str(e)) from e
        except RepositoryError as e:
            raise PromptLibraryServiceError(
                f"Failed to save from campaign: {e}",
                details={"campaign_id": campaign_id, "name": name},
            ) from e

    # =========================================================================
    # Version Operations
    # =========================================================================

    def get_template_versions(
        self,
        template_id: str,
        user_id: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[TemplateVersion]:
        """
        Get version history for a template.

        Args:
            template_id: Template identifier.
            user_id: User for visibility check.
            limit: Maximum versions to return.
            offset: Versions to skip.

        Returns:
            List of versions, newest first.

        Raises:
            TemplateNotFoundError: If template not found.
            TemplatePermissionError: If user lacks access.
        """
        self._operation_count += 1

        # Verify access
        self.get_template(template_id, user_id, check_visibility=True)

        try:
            return self.repository.get_versions(
                template_id=template_id,
                limit=limit,
                offset=offset,
            )
        except EntityNotFoundError as e:
            raise TemplateNotFoundError(
                f"Template '{template_id}' not found",
                details={"template_id": template_id},
            ) from e

    def get_template_version(
        self,
        template_id: str,
        version_number: int | None = None,
        version_id: str | None = None,
        user_id: str | None = None,
    ) -> TemplateVersion | None:
        """
        Get a specific version of a template.

        Args:
            template_id: Template identifier.
            version_number: Version number to retrieve.
            version_id: Version ID to retrieve.
            user_id: User for visibility check.

        Returns:
            TemplateVersion if found, None otherwise.

        Raises:
            TemplateNotFoundError: If template not found.
            TemplatePermissionError: If user lacks access.
        """
        self._operation_count += 1

        # Verify access
        self.get_template(template_id, user_id, check_visibility=True)

        return self.repository.get_version(
            template_id=template_id,
            version_number=version_number,
            version_id=version_id,
        )

    def create_template_version(
        self,
        template_id: str,
        prompt_content: str,
        change_summary: str = "",
        user_id: str | None = None,
    ) -> TemplateVersion:
        """
        Create a new version for a template.

        Args:
            template_id: Template identifier.
            prompt_content: New prompt content.
            change_summary: Summary of changes.
            user_id: User creating the version.

        Returns:
            Created TemplateVersion.

        Raises:
            TemplateNotFoundError: If template not found.
            TemplatePermissionError: If user lacks permission.
            TemplateValidationError: If validation fails.
        """
        self._operation_count += 1

        # Get template and check edit permission
        template = self.get_template(template_id, user_id, check_visibility=True)

        if not self._can_edit_template(template, user_id):
            raise TemplatePermissionError(
                f"User lacks permission to create version for template '{template_id}'",
                details={"template_id": template_id, "user_id": user_id},
            )

        if not prompt_content or not prompt_content.strip():
            raise TemplateValidationError(
                "Prompt content is required",
                details={"field": "prompt_content"},
            )

        try:
            version = self.repository.create_version(
                template_id=template_id,
                prompt_content=prompt_content,
                change_summary=change_summary,
                created_by=user_id,
            )

            logger.info(
                f"Created version {version.version_number} for template '{template_id}'"
            )
            return version

        except EntityNotFoundError as e:
            raise TemplateNotFoundError(
                f"Template '{template_id}' not found",
                details={"template_id": template_id},
            ) from e
        except RepositoryError as e:
            raise PromptLibraryServiceError(
                f"Failed to create version: {e}",
                details={"template_id": template_id},
            ) from e

    def restore_template_version(
        self,
        template_id: str,
        version_number: int,
        user_id: str | None = None,
    ) -> PromptTemplate:
        """
        Restore a template to a previous version.

        Creates a new version with the content from the specified version.

        Args:
            template_id: Template identifier.
            version_number: Version number to restore.
            user_id: User performing the restore.

        Returns:
            Updated PromptTemplate.

        Raises:
            TemplateNotFoundError: If template or version not found.
            TemplatePermissionError: If user lacks permission.
        """
        self._operation_count += 1

        # Get template and check edit permission
        template = self.get_template(template_id, user_id, check_visibility=True)

        if not self._can_edit_template(template, user_id):
            raise TemplatePermissionError(
                f"User lacks permission to restore template '{template_id}'",
                details={"template_id": template_id, "user_id": user_id},
            )

        try:
            restored = self.repository.restore_version(
                template_id=template_id,
                version_number=version_number,
                restored_by=user_id,
            )

            logger.info(
                f"Restored template '{template_id}' to version {version_number}"
            )
            return restored

        except EntityNotFoundError as e:
            raise TemplateNotFoundError(
                str(e),
                details={"template_id": template_id, "version_number": version_number},
            ) from e

    def compare_versions(
        self,
        template_id: str,
        version_a: int,
        version_b: int,
        user_id: str | None = None,
    ) -> dict[str, Any]:
        """
        Compare two versions of a template.

        Args:
            template_id: Template identifier.
            version_a: First version number.
            version_b: Second version number.
            user_id: User for visibility check.

        Returns:
            Dict with version comparison data.

        Raises:
            TemplateNotFoundError: If template or versions not found.
            TemplatePermissionError: If user lacks access.
        """
        self._operation_count += 1

        # Verify access
        self.get_template(template_id, user_id, check_visibility=True)

        ver_a = self.repository.get_version(template_id, version_number=version_a)
        ver_b = self.repository.get_version(template_id, version_number=version_b)

        if not ver_a:
            raise TemplateNotFoundError(
                f"Version {version_a} not found for template '{template_id}'",
                details={"template_id": template_id, "version_number": version_a},
            )
        if not ver_b:
            raise TemplateNotFoundError(
                f"Version {version_b} not found for template '{template_id}'",
                details={"template_id": template_id, "version_number": version_b},
            )

        # Simple comparison - could be enhanced with diff library
        content_changed = ver_a.prompt_content != ver_b.prompt_content

        return {
            "template_id": template_id,
            "version_a": {
                "version_id": ver_a.version_id,
                "version_number": ver_a.version_number,
                "prompt_content": ver_a.prompt_content,
                "change_summary": ver_a.change_summary,
                "created_at": ver_a.created_at.isoformat(),
                "created_by": ver_a.created_by,
            },
            "version_b": {
                "version_id": ver_b.version_id,
                "version_number": ver_b.version_number,
                "prompt_content": ver_b.prompt_content,
                "change_summary": ver_b.change_summary,
                "created_at": ver_b.created_at.isoformat(),
                "created_by": ver_b.created_by,
            },
            "content_changed": content_changed,
            "content_length_a": len(ver_a.prompt_content),
            "content_length_b": len(ver_b.prompt_content),
            "length_diff": len(ver_b.prompt_content) - len(ver_a.prompt_content),
        }

    # =========================================================================
    # Statistics and Analytics
    # =========================================================================

    def get_library_statistics(
        self,
        user_id: str | None = None,
        include_user_stats: bool = True,
    ) -> dict[str, Any]:
        """
        Get comprehensive library statistics.

        Args:
            user_id: User ID for user-specific stats.
            include_user_stats: Whether to include user-specific statistics.

        Returns:
            Dict with library statistics.
        """
        self._operation_count += 1

        # Get repository stats
        repo_stats = self.repository.get_statistics()

        stats = {
            "total_templates": repo_stats["total_templates"],
            "by_sharing_level": repo_stats["by_sharing_level"],
            "by_status": repo_stats["by_status"],
            "top_techniques": repo_stats["top_techniques"],
            "total_ratings": repo_stats["total_ratings"],
            "total_versions": repo_stats["total_versions"],
        }

        # Add user-specific stats if requested
        if include_user_stats and user_id:
            user_templates = self.repository.list_templates(
                limit=100000,
                offset=0,
                user_id=user_id,
            )
            stats["user_stats"] = {
                "templates_created": len(user_templates),
                "public_templates": sum(
                    1 for t in user_templates
                    if t.sharing_level == SharingLevel.PUBLIC
                ),
                "private_templates": sum(
                    1 for t in user_templates
                    if t.sharing_level == SharingLevel.PRIVATE
                ),
            }

        # Add calculated metrics
        templates = self.repository.list_templates(limit=100000, offset=0)
        if templates:
            # Calculate average success rate
            success_rates = [
                t.metadata.success_rate
                for t in templates
                if t.metadata.success_rate is not None
            ]
            if success_rates:
                stats["average_success_rate"] = sum(success_rates) / len(success_rates)

            # Count templates with high ratings
            stats["top_rated_count"] = sum(
                1 for t in templates
                if t.rating_stats.average_rating >= 4.0
            )

            # Count by vulnerability type
            vuln_counts: dict[str, int] = {}
            for template in templates:
                for vuln in template.metadata.vulnerability_types:
                    vuln_counts[vuln.value] = vuln_counts.get(vuln.value, 0) + 1
            stats["vulnerabilities_targeted"] = dict(
                sorted(vuln_counts.items(), key=lambda x: x[1], reverse=True)[:10]
            )

        return stats

    def get_template_statistics(
        self,
        template_id: str,
        user_id: str | None = None,
    ) -> dict[str, Any]:
        """
        Get statistics for a specific template.

        Args:
            template_id: Template identifier.
            user_id: User for visibility check.

        Returns:
            Dict with template statistics.

        Raises:
            TemplateNotFoundError: If template not found.
        """
        self._operation_count += 1

        template = self.get_template(template_id, user_id, check_visibility=True)

        version_count = self.repository.count_versions(template_id)
        rating_count = self.repository.count_ratings(template_id)

        return {
            "template_id": template_id,
            "name": template.name,
            "current_version": template.current_version,
            "total_versions": version_count,
            "rating_stats": template.rating_stats.model_dump(),
            "total_ratings": rating_count,
            "success_rate": template.metadata.success_rate,
            "test_count": template.metadata.test_count,
            "technique_count": len(template.metadata.technique_types),
            "vulnerability_count": len(template.metadata.vulnerability_types),
            "tag_count": len(template.metadata.tags),
            "created_at": template.created_at.isoformat(),
            "updated_at": template.updated_at.isoformat(),
            "days_since_creation": (datetime.utcnow() - template.created_at).days,
            "days_since_update": (datetime.utcnow() - template.updated_at).days,
        }

    def get_service_stats(self) -> dict[str, Any]:
        """
        Get service-level statistics.

        Returns:
            Dict with service statistics.
        """
        repo_stats = self.repository.get_statistics()

        return {
            "service": "prompt_library",
            "operation_count": self._operation_count,
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "repository": {
                "total_templates": repo_stats["total_templates"],
                "total_ratings": repo_stats["total_ratings"],
                "total_versions": repo_stats["total_versions"],
                "persistence_enabled": repo_stats["persistence_enabled"],
            },
        }

    def reset_stats(self) -> None:
        """Reset service statistics."""
        self._operation_count = 0
        self._cache_hits = 0
        self._cache_misses = 0

    # =========================================================================
    # Permission Helpers
    # =========================================================================

    def _can_view_template(
        self,
        template: PromptTemplate,
        user_id: str | None,
    ) -> bool:
        """Check if user can view a template."""
        # Public templates are visible to everyone
        if template.sharing_level == SharingLevel.PUBLIC:
            return True

        # Private templates only visible to creator
        if template.sharing_level == SharingLevel.PRIVATE:
            return user_id is not None and template.created_by == user_id

        # Team templates visible to creator (full team support would check membership)
        if template.sharing_level == SharingLevel.TEAM:
            return user_id is not None and (
                template.created_by == user_id
                # Future: or user is team member
            )

        return False

    def _can_edit_template(
        self,
        template: PromptTemplate,
        user_id: str | None,
    ) -> bool:
        """Check if user can edit a template."""
        if user_id is None:
            return False

        # Creator can always edit
        if template.created_by == user_id:
            return True

        # Future: Team admins can edit team templates
        return False

    def _can_delete_template(
        self,
        template: PromptTemplate,
        user_id: str | None,
    ) -> bool:
        """Check if user can delete a template."""
        if user_id is None:
            return False

        # Only creator can delete
        return template.created_by == user_id


# =============================================================================
# Singleton Instance
# =============================================================================

_service_instance: PromptLibraryService | None = None


def get_prompt_library_service(
    reset: bool = False,
) -> PromptLibraryService:
    """
    Get the singleton service instance.

    Args:
        reset: Whether to reset the singleton instance.

    Returns:
        PromptLibraryService instance.
    """
    global _service_instance

    if reset or _service_instance is None:
        _service_instance = PromptLibraryService()

    return _service_instance
