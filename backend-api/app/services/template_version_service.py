"""
Template Version Service - Business Logic Layer for Version Control.

This module implements the service layer for template version operations,
providing a clean interface between API endpoints and the repository layer.
Includes version creation, tracking prompt refinements, comparing changes,
and version analytics.
"""

import difflib
import logging
from datetime import datetime
from typing import Any

from app.domain.prompt_library_models import (
    PromptTemplate,
    SharingLevel,
    TemplateVersion,
)
from app.repositories.prompt_library_repository import (
    EntityNotFoundError,
    PromptLibraryRepository,
    RepositoryError,
    ValidationError,
    get_prompt_library_repository,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Custom Exceptions
# =============================================================================


class TemplateVersionServiceError(Exception):
    """Base exception for template version service operations."""

    def __init__(self, message: str, details: dict[str, Any] | None = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}


class VersionNotFoundError(TemplateVersionServiceError):
    """Raised when a version is not found."""


class VersionValidationError(TemplateVersionServiceError):
    """Raised when version validation fails."""


class VersionPermissionError(TemplateVersionServiceError):
    """Raised when user lacks permission for a version operation."""


class TemplateNotFoundError(TemplateVersionServiceError):
    """Raised when the template is not found."""


# =============================================================================
# Diff Models
# =============================================================================


class VersionDiff:
    """Represents the difference between two versions."""

    def __init__(
        self,
        version_a: TemplateVersion,
        version_b: TemplateVersion,
        unified_diff: str,
        additions: int,
        deletions: int,
        changes: list[dict[str, Any]],
    ):
        self.version_a = version_a
        self.version_b = version_b
        self.unified_diff = unified_diff
        self.additions = additions
        self.deletions = deletions
        self.changes = changes

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "version_a": {
                "version_id": self.version_a.version_id,
                "version_number": self.version_a.version_number,
                "created_at": self.version_a.created_at.isoformat(),
                "created_by": self.version_a.created_by,
                "change_summary": self.version_a.change_summary,
            },
            "version_b": {
                "version_id": self.version_b.version_id,
                "version_number": self.version_b.version_number,
                "created_at": self.version_b.created_at.isoformat(),
                "created_by": self.version_b.created_by,
                "change_summary": self.version_b.change_summary,
            },
            "unified_diff": self.unified_diff,
            "additions": self.additions,
            "deletions": self.deletions,
            "changes": self.changes,
            "has_changes": self.additions > 0 or self.deletions > 0,
        }


# =============================================================================
# Template Version Service
# =============================================================================


class TemplateVersionService:
    """
    Service layer for template version management.

    Provides business logic for tracking prompt refinements, creating versions,
    comparing changes, restoring versions, and version analytics. Wraps the
    repository layer with additional validation, authorization, and business rules.

    Attributes:
        repository: The underlying repository for data persistence.
        default_page_size: Default number of results per page for list operations.
    """

    def __init__(
        self,
        repository: PromptLibraryRepository | None = None,
        default_page_size: int = 50,
    ):
        """
        Initialize the template version service.

        Args:
            repository: Repository instance (uses singleton if None).
            default_page_size: Default page size for list operations.
        """
        self._repository = repository
        self._default_page_size = default_page_size
        self._operation_count = 0
        self._versions_created = 0
        self._versions_restored = 0
        self._comparisons_made = 0

        logger.info("TemplateVersionService initialized")

    @property
    def repository(self) -> PromptLibraryRepository:
        """Get the repository instance (lazy initialization)."""
        if self._repository is None:
            self._repository = get_prompt_library_repository()
        return self._repository

    # =========================================================================
    # Version CRUD Operations
    # =========================================================================

    def get_versions(
        self,
        template_id: str,
        user_id: str | None = None,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[TemplateVersion]:
        """
        Get version history for a template.

        Args:
            template_id: Template identifier.
            user_id: User for visibility check.
            limit: Maximum number of versions to return.
            offset: Number of versions to skip.

        Returns:
            List of versions, newest first.

        Raises:
            TemplateNotFoundError: If template not found.
            VersionPermissionError: If user lacks access.
        """
        self._operation_count += 1
        limit = limit or self._default_page_size

        # Verify template exists and user can access it
        self._verify_template_access(template_id, user_id)

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

    def get_version(
        self,
        template_id: str,
        version_number: int | None = None,
        version_id: str | None = None,
        user_id: str | None = None,
    ) -> TemplateVersion:
        """
        Get a specific version of a template.

        Args:
            template_id: Template identifier.
            version_number: Version number to retrieve.
            version_id: Version ID to retrieve.
            user_id: User for visibility check.

        Returns:
            TemplateVersion.

        Raises:
            TemplateNotFoundError: If template not found.
            VersionNotFoundError: If version not found.
            VersionPermissionError: If user lacks access.
        """
        self._operation_count += 1

        # Verify template exists and user can access it
        self._verify_template_access(template_id, user_id)

        version = self.repository.get_version(
            template_id=template_id,
            version_number=version_number,
            version_id=version_id,
        )

        if not version:
            identifier = f"number {version_number}" if version_number else f"ID {version_id}"
            raise VersionNotFoundError(
                f"Version {identifier} not found for template '{template_id}'",
                details={
                    "template_id": template_id,
                    "version_number": version_number,
                    "version_id": version_id,
                },
            )

        return version

    def get_latest_version(
        self,
        template_id: str,
        user_id: str | None = None,
    ) -> TemplateVersion:
        """
        Get the latest version for a template.

        Args:
            template_id: Template identifier.
            user_id: User for visibility check.

        Returns:
            Latest TemplateVersion.

        Raises:
            TemplateNotFoundError: If template not found.
            VersionNotFoundError: If no versions exist.
        """
        self._operation_count += 1

        # Verify template exists and user can access it
        template = self._verify_template_access(template_id, user_id)

        return self.get_version(
            template_id=template_id,
            version_number=template.current_version,
            user_id=user_id,
        )

    def create_version(
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
            VersionPermissionError: If user lacks permission.
            VersionValidationError: If validation fails.
        """
        self._operation_count += 1

        # Verify template exists and user can edit it
        template = self._verify_template_access(template_id, user_id)

        if not self._can_edit_template(template, user_id):
            raise VersionPermissionError(
                f"User lacks permission to create version for template '{template_id}'",
                details={"template_id": template_id, "user_id": user_id},
            )

        # Validate content
        if not prompt_content or not prompt_content.strip():
            raise VersionValidationError(
                "Prompt content is required",
                details={"field": "prompt_content"},
            )

        # Check if content is different from current version
        current_version = self.repository.get_version(
            template_id=template_id,
            version_number=template.current_version,
        )
        if current_version and current_version.prompt_content == prompt_content:
            raise VersionValidationError(
                "New version content is identical to current version",
                details={"template_id": template_id, "current_version": template.current_version},
            )

        try:
            version = self.repository.create_version(
                template_id=template_id,
                prompt_content=prompt_content,
                change_summary=change_summary,
                created_by=user_id,
            )

            self._versions_created += 1
            logger.info(
                f"Created version {version.version_number} for template '{template_id}' "
                f"by user '{user_id or 'anonymous'}'"
            )
            return version

        except EntityNotFoundError as e:
            raise TemplateNotFoundError(
                f"Template '{template_id}' not found",
                details={"template_id": template_id},
            ) from e
        except RepositoryError as e:
            raise TemplateVersionServiceError(
                f"Failed to create version: {e}",
                details={"template_id": template_id},
            ) from e

    def restore_version(
        self,
        template_id: str,
        version_number: int,
        user_id: str | None = None,
        restore_summary: str | None = None,
    ) -> PromptTemplate:
        """
        Restore a template to a previous version.

        Creates a new version with the content from the specified version,
        preserving the version history.

        Args:
            template_id: Template identifier.
            version_number: Version number to restore.
            user_id: User performing the restore.
            restore_summary: Custom summary for the restore operation.

        Returns:
            Updated PromptTemplate.

        Raises:
            TemplateNotFoundError: If template not found.
            VersionNotFoundError: If version not found.
            VersionPermissionError: If user lacks permission.
        """
        self._operation_count += 1

        # Verify template exists and user can edit it
        template = self._verify_template_access(template_id, user_id)

        if not self._can_edit_template(template, user_id):
            raise VersionPermissionError(
                f"User lacks permission to restore template '{template_id}'",
                details={"template_id": template_id, "user_id": user_id},
            )

        # Verify version exists
        version = self.get_version(
            template_id=template_id,
            version_number=version_number,
            user_id=user_id,
        )

        # Check if already at this version
        if template.prompt_content == version.prompt_content:
            raise VersionValidationError(
                f"Template already has content from version {version_number}",
                details={"template_id": template_id, "version_number": version_number},
            )

        try:
            restored = self.repository.restore_version(
                template_id=template_id,
                version_number=version_number,
                restored_by=user_id,
            )

            self._versions_restored += 1
            logger.info(
                f"Restored template '{template_id}' to version {version_number} "
                f"by user '{user_id or 'anonymous'}'"
            )
            return restored

        except EntityNotFoundError as e:
            raise VersionNotFoundError(
                str(e),
                details={"template_id": template_id, "version_number": version_number},
            ) from e

    # =========================================================================
    # Version Comparison
    # =========================================================================

    def compare_versions(
        self,
        template_id: str,
        version_a: int,
        version_b: int,
        user_id: str | None = None,
        context_lines: int = 3,
    ) -> VersionDiff:
        """
        Compare two versions of a template.

        Args:
            template_id: Template identifier.
            version_a: First version number (older).
            version_b: Second version number (newer).
            user_id: User for visibility check.
            context_lines: Number of context lines in unified diff.

        Returns:
            VersionDiff with detailed comparison data.

        Raises:
            TemplateNotFoundError: If template not found.
            VersionNotFoundError: If either version not found.
            VersionPermissionError: If user lacks access.
        """
        self._operation_count += 1
        self._comparisons_made += 1

        # Verify template access
        self._verify_template_access(template_id, user_id)

        # Get both versions
        ver_a = self.get_version(
            template_id=template_id,
            version_number=version_a,
            user_id=user_id,
        )
        ver_b = self.get_version(
            template_id=template_id,
            version_number=version_b,
            user_id=user_id,
        )

        # Generate unified diff
        lines_a = ver_a.prompt_content.splitlines(keepends=True)
        lines_b = ver_b.prompt_content.splitlines(keepends=True)

        diff_lines = list(difflib.unified_diff(
            lines_a,
            lines_b,
            fromfile=f"Version {version_a}",
            tofile=f"Version {version_b}",
            fromfiledate=ver_a.created_at.isoformat(),
            tofiledate=ver_b.created_at.isoformat(),
            n=context_lines,
        ))

        unified_diff = "".join(diff_lines)

        # Calculate additions and deletions
        additions = 0
        deletions = 0
        for line in diff_lines:
            if line.startswith("+") and not line.startswith("+++"):
                additions += 1
            elif line.startswith("-") and not line.startswith("---"):
                deletions += 1

        # Generate detailed change list
        changes = self._generate_detailed_changes(lines_a, lines_b)

        logger.debug(
            f"Compared versions {version_a} and {version_b} for template '{template_id}': "
            f"+{additions}/-{deletions} lines"
        )

        return VersionDiff(
            version_a=ver_a,
            version_b=ver_b,
            unified_diff=unified_diff,
            additions=additions,
            deletions=deletions,
            changes=changes,
        )

    def compare_with_current(
        self,
        template_id: str,
        version_number: int,
        user_id: str | None = None,
        context_lines: int = 3,
    ) -> VersionDiff:
        """
        Compare a version with the current template content.

        Args:
            template_id: Template identifier.
            version_number: Version number to compare.
            user_id: User for visibility check.
            context_lines: Number of context lines in unified diff.

        Returns:
            VersionDiff comparing the specified version with current.

        Raises:
            TemplateNotFoundError: If template not found.
            VersionNotFoundError: If version not found.
        """
        self._operation_count += 1

        template = self._verify_template_access(template_id, user_id)

        return self.compare_versions(
            template_id=template_id,
            version_a=version_number,
            version_b=template.current_version,
            user_id=user_id,
            context_lines=context_lines,
        )

    def _generate_detailed_changes(
        self,
        lines_a: list[str],
        lines_b: list[str],
    ) -> list[dict[str, Any]]:
        """Generate detailed change list between two sets of lines."""
        changes = []
        matcher = difflib.SequenceMatcher(None, lines_a, lines_b)

        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == "equal":
                continue
            elif tag == "replace":
                changes.append({
                    "type": "replace",
                    "old_start": i1 + 1,
                    "old_end": i2,
                    "new_start": j1 + 1,
                    "new_end": j2,
                    "old_content": "".join(lines_a[i1:i2]),
                    "new_content": "".join(lines_b[j1:j2]),
                })
            elif tag == "delete":
                changes.append({
                    "type": "delete",
                    "old_start": i1 + 1,
                    "old_end": i2,
                    "old_content": "".join(lines_a[i1:i2]),
                })
            elif tag == "insert":
                changes.append({
                    "type": "insert",
                    "new_start": j1 + 1,
                    "new_end": j2,
                    "new_content": "".join(lines_b[j1:j2]),
                })

        return changes

    # =========================================================================
    # Version Lineage and History
    # =========================================================================

    def get_version_chain(
        self,
        template_id: str,
        user_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Get the complete version chain for a template.

        Returns versions with parent-child relationships for visualization.

        Args:
            template_id: Template identifier.
            user_id: User for visibility check.

        Returns:
            List of versions with relationship data.

        Raises:
            TemplateNotFoundError: If template not found.
        """
        self._operation_count += 1

        self._verify_template_access(template_id, user_id)

        versions = self.repository.get_versions(
            template_id=template_id,
            limit=1000,  # Get all versions
            offset=0,
        )

        # Build chain with relationships
        chain = []
        version_map = {v.version_id: v for v in versions}

        for version in sorted(versions, key=lambda v: v.version_number):
            parent = None
            if version.parent_version_id and version.parent_version_id in version_map:
                parent_ver = version_map[version.parent_version_id]
                parent = {
                    "version_id": parent_ver.version_id,
                    "version_number": parent_ver.version_number,
                }

            chain.append({
                "version_id": version.version_id,
                "version_number": version.version_number,
                "created_at": version.created_at.isoformat(),
                "created_by": version.created_by,
                "change_summary": version.change_summary,
                "parent": parent,
                "content_length": len(version.prompt_content),
                "is_restore": "Restored from version" in version.change_summary,
            })

        return chain

    def get_version_timeline(
        self,
        template_id: str,
        user_id: str | None = None,
        include_content: bool = False,
    ) -> list[dict[str, Any]]:
        """
        Get version history as a timeline with summaries.

        Args:
            template_id: Template identifier.
            user_id: User for visibility check.
            include_content: Whether to include prompt content.

        Returns:
            Timeline of version events.

        Raises:
            TemplateNotFoundError: If template not found.
        """
        self._operation_count += 1

        self._verify_template_access(template_id, user_id)

        versions = self.repository.get_versions(
            template_id=template_id,
            limit=1000,
            offset=0,
        )

        # Sort by creation date ascending for timeline
        versions = sorted(versions, key=lambda v: v.created_at)

        timeline = []
        prev_version = None

        for version in versions:
            entry = {
                "version_number": version.version_number,
                "version_id": version.version_id,
                "created_at": version.created_at.isoformat(),
                "created_by": version.created_by,
                "change_summary": version.change_summary,
                "content_length": len(version.prompt_content),
            }

            # Calculate delta from previous version
            if prev_version:
                entry["delta"] = {
                    "length_change": len(version.prompt_content) - len(prev_version.prompt_content),
                    "time_since_previous": (
                        version.created_at - prev_version.created_at
                    ).total_seconds(),
                }
            else:
                entry["delta"] = None

            if include_content:
                entry["prompt_content"] = version.prompt_content

            timeline.append(entry)
            prev_version = version

        return timeline

    # =========================================================================
    # Version Analytics
    # =========================================================================

    def get_version_statistics(
        self,
        template_id: str,
        user_id: str | None = None,
    ) -> dict[str, Any]:
        """
        Get statistics about a template's version history.

        Args:
            template_id: Template identifier.
            user_id: User for visibility check.

        Returns:
            Dictionary with version statistics.

        Raises:
            TemplateNotFoundError: If template not found.
        """
        self._operation_count += 1

        template = self._verify_template_access(template_id, user_id)

        versions = self.repository.get_versions(
            template_id=template_id,
            limit=1000,
            offset=0,
        )

        if not versions:
            return {
                "template_id": template_id,
                "total_versions": 0,
                "contributors": [],
                "average_change_interval": None,
                "total_content_growth": 0,
            }

        # Sort by version number
        versions = sorted(versions, key=lambda v: v.version_number)

        # Calculate statistics
        contributors = list(set(v.created_by for v in versions if v.created_by))
        restore_count = sum(1 for v in versions if "Restored from version" in v.change_summary)

        # Calculate time intervals
        intervals = []
        for i in range(1, len(versions)):
            delta = (versions[i].created_at - versions[i - 1].created_at).total_seconds()
            intervals.append(delta)

        avg_interval = sum(intervals) / len(intervals) if intervals else None

        # Calculate content changes
        first_version = versions[0]
        last_version = versions[-1]
        content_growth = len(last_version.prompt_content) - len(first_version.prompt_content)

        # Find largest changes
        largest_additions = 0
        largest_deletions = 0
        for i in range(1, len(versions)):
            diff = len(versions[i].prompt_content) - len(versions[i - 1].prompt_content)
            if diff > largest_additions:
                largest_additions = diff
            if diff < largest_deletions:
                largest_deletions = diff

        return {
            "template_id": template_id,
            "template_name": template.name,
            "current_version": template.current_version,
            "total_versions": len(versions),
            "contributors": contributors,
            "contributor_count": len(contributors),
            "restore_count": restore_count,
            "average_change_interval_seconds": avg_interval,
            "total_content_growth": content_growth,
            "first_version_date": first_version.created_at.isoformat(),
            "last_version_date": last_version.created_at.isoformat(),
            "content_lengths": {
                "first": len(first_version.prompt_content),
                "current": len(last_version.prompt_content),
                "largest_addition": largest_additions,
                "largest_deletion": abs(largest_deletions),
            },
        }

    def get_contributor_activity(
        self,
        template_id: str,
        user_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Get activity breakdown by contributor for a template.

        Args:
            template_id: Template identifier.
            user_id: User for visibility check.

        Returns:
            List of contributor activity summaries.

        Raises:
            TemplateNotFoundError: If template not found.
        """
        self._operation_count += 1

        self._verify_template_access(template_id, user_id)

        versions = self.repository.get_versions(
            template_id=template_id,
            limit=1000,
            offset=0,
        )

        # Group by contributor
        contributor_data: dict[str, dict[str, Any]] = {}

        for version in versions:
            contributor = version.created_by or "anonymous"
            if contributor not in contributor_data:
                contributor_data[contributor] = {
                    "user_id": contributor,
                    "version_count": 0,
                    "first_contribution": version.created_at,
                    "last_contribution": version.created_at,
                    "versions": [],
                }

            contributor_data[contributor]["version_count"] += 1
            contributor_data[contributor]["versions"].append(version.version_number)

            if version.created_at < contributor_data[contributor]["first_contribution"]:
                contributor_data[contributor]["first_contribution"] = version.created_at
            if version.created_at > contributor_data[contributor]["last_contribution"]:
                contributor_data[contributor]["last_contribution"] = version.created_at

        # Format and sort by version count
        result = []
        for contributor, data in contributor_data.items():
            result.append({
                "user_id": data["user_id"],
                "version_count": data["version_count"],
                "first_contribution": data["first_contribution"].isoformat(),
                "last_contribution": data["last_contribution"].isoformat(),
                "versions": data["versions"],
            })

        result.sort(key=lambda x: x["version_count"], reverse=True)
        return result

    # =========================================================================
    # Bulk Operations
    # =========================================================================

    def get_recent_versions_across_templates(
        self,
        user_id: str | None = None,
        limit: int = 20,
        sharing_level: SharingLevel | None = SharingLevel.PUBLIC,
    ) -> list[dict[str, Any]]:
        """
        Get recent versions across all accessible templates.

        Args:
            user_id: User for visibility filtering.
            limit: Maximum number of versions to return.
            sharing_level: Filter by template sharing level.

        Returns:
            List of recent versions with template info.
        """
        self._operation_count += 1

        # Get accessible templates
        templates = self.repository.list_templates(
            limit=100,
            offset=0,
            sharing_levels=[sharing_level] if sharing_level else None,
        )

        # Collect all versions with template info
        all_versions = []
        for template in templates:
            # Check visibility
            if not self._can_view_template(template, user_id):
                continue

            versions = self.repository.get_versions(
                template_id=template.id,
                limit=5,  # Get recent versions per template
                offset=0,
            )

            for version in versions:
                all_versions.append({
                    "template_id": template.id,
                    "template_name": template.name,
                    "sharing_level": template.sharing_level.value,
                    "version_id": version.version_id,
                    "version_number": version.version_number,
                    "created_at": version.created_at,
                    "created_by": version.created_by,
                    "change_summary": version.change_summary,
                })

        # Sort by creation date descending
        all_versions.sort(key=lambda v: v["created_at"], reverse=True)

        # Format dates and limit results
        result = []
        for version in all_versions[:limit]:
            version["created_at"] = version["created_at"].isoformat()
            result.append(version)

        return result

    # =========================================================================
    # Service Statistics
    # =========================================================================

    def get_service_stats(self) -> dict[str, Any]:
        """
        Get service-level statistics.

        Returns:
            Dict with service statistics.
        """
        repo_stats = self.repository.get_statistics()

        return {
            "service": "template_version",
            "operation_count": self._operation_count,
            "versions_created": self._versions_created,
            "versions_restored": self._versions_restored,
            "comparisons_made": self._comparisons_made,
            "repository": {
                "total_templates": repo_stats["total_templates"],
                "total_versions": repo_stats["total_versions"],
            },
        }

    def reset_stats(self) -> None:
        """Reset service statistics."""
        self._operation_count = 0
        self._versions_created = 0
        self._versions_restored = 0
        self._comparisons_made = 0

    # =========================================================================
    # Private Helpers
    # =========================================================================

    def _verify_template_access(
        self,
        template_id: str,
        user_id: str | None,
    ) -> PromptTemplate:
        """
        Verify template exists and user can access it.

        Args:
            template_id: Template identifier.
            user_id: User ID for visibility check.

        Returns:
            PromptTemplate.

        Raises:
            TemplateNotFoundError: If template not found.
            VersionPermissionError: If user lacks access.
        """
        template = self.repository.get_template(template_id)

        if not template:
            raise TemplateNotFoundError(
                f"Template '{template_id}' not found",
                details={"template_id": template_id},
            )

        if not self._can_view_template(template, user_id):
            raise VersionPermissionError(
                f"User lacks permission to access template '{template_id}'",
                details={"template_id": template_id, "user_id": user_id},
            )

        return template

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


# =============================================================================
# Singleton Instance
# =============================================================================

_service_instance: TemplateVersionService | None = None


def get_template_version_service(
    reset: bool = False,
) -> TemplateVersionService:
    """
    Get the singleton service instance.

    Args:
        reset: Whether to reset the singleton instance.

    Returns:
        TemplateVersionService instance.
    """
    global _service_instance

    if reset or _service_instance is None:
        _service_instance = TemplateVersionService()

    return _service_instance
