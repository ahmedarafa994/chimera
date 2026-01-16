"""
Template Rating Service - Business Logic Layer for Template Ratings.

This module implements the service layer for template rating operations,
providing a clean interface between API endpoints and the repository layer.
Includes rating CRUD, effectiveness voting, and aggregated statistics.
"""

import logging
from datetime import datetime
from typing import Any

from app.domain.prompt_library_models import (
    PromptTemplate,
    RatingStatistics,
    SharingLevel,
    TemplateRating,
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


class TemplateRatingServiceError(Exception):
    """Base exception for template rating service operations."""

    def __init__(self, message: str, details: dict[str, Any] | None = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}


class RatingNotFoundError(TemplateRatingServiceError):
    """Raised when a rating is not found."""


class RatingValidationError(TemplateRatingServiceError):
    """Raised when rating validation fails."""


class RatingPermissionError(TemplateRatingServiceError):
    """Raised when user lacks permission for a rating operation."""


class TemplateNotFoundError(TemplateRatingServiceError):
    """Raised when the template being rated is not found."""


# =============================================================================
# Template Rating Service
# =============================================================================


class TemplateRatingService:
    """
    Service layer for template rating management.

    Provides business logic for user ratings, effectiveness voting,
    aggregated statistics, and rating analytics. Wraps the repository
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
        Initialize the template rating service.

        Args:
            repository: Repository instance (uses singleton if None).
            default_page_size: Default page size for list operations.
        """
        self._repository = repository
        self._default_page_size = default_page_size
        self._operation_count = 0
        self._ratings_submitted = 0
        self._ratings_updated = 0

        logger.info("TemplateRatingService initialized")

    @property
    def repository(self) -> PromptLibraryRepository:
        """Get the repository instance (lazy initialization)."""
        if self._repository is None:
            self._repository = get_prompt_library_repository()
        return self._repository

    # =========================================================================
    # Rating Operations
    # =========================================================================

    def rate_template(
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
        Rate a template or update an existing rating.

        If the user has already rated this template, their rating will be updated.
        Otherwise, a new rating will be created.

        Args:
            template_id: Template identifier.
            user_id: User ID of the rater.
            rating: Star rating (1-5).
            effectiveness_score: Optional effectiveness rating (1-5).
            comment: Optional review comment.
            reported_success: Whether the template worked for the user.
            target_model_tested: Model the user tested against.

        Returns:
            Created or updated TemplateRating.

        Raises:
            TemplateNotFoundError: If template not found.
            RatingValidationError: If rating validation fails.
            RatingPermissionError: If user lacks access to rate the template.
        """
        self._operation_count += 1

        # Validate inputs
        if rating < 1 or rating > 5:
            raise RatingValidationError(
                "Rating must be between 1 and 5",
                details={"field": "rating", "value": rating},
            )
        if effectiveness_score is not None and (effectiveness_score < 1 or effectiveness_score > 5):
            raise RatingValidationError(
                "Effectiveness score must be between 1 and 5",
                details={"field": "effectiveness_score", "value": effectiveness_score},
            )
        if not user_id or not user_id.strip():
            raise RatingValidationError(
                "User ID is required for rating",
                details={"field": "user_id"},
            )

        # Verify template exists and user can access it
        template = self._get_and_verify_template(template_id, user_id)

        # Check if updating existing rating
        existing_rating = self.repository.get_rating(template_id, user_id)
        is_update = existing_rating is not None

        try:
            template_rating = self.repository.add_rating(
                template_id=template_id,
                user_id=user_id,
                rating=rating,
                effectiveness_score=effectiveness_score,
                comment=comment,
                reported_success=reported_success,
                target_model_tested=target_model_tested,
            )

            if is_update:
                self._ratings_updated += 1
                logger.info(
                    f"User '{user_id}' updated rating for template '{template_id}' "
                    f"to {rating} stars"
                )
            else:
                self._ratings_submitted += 1
                logger.info(
                    f"User '{user_id}' rated template '{template_id}' "
                    f"with {rating} stars"
                )

            return template_rating

        except ValidationError as e:
            raise RatingValidationError(str(e)) from e
        except EntityNotFoundError as e:
            raise TemplateNotFoundError(
                f"Template '{template_id}' not found",
                details={"template_id": template_id},
            ) from e
        except RepositoryError as e:
            raise TemplateRatingServiceError(
                f"Failed to rate template: {e}",
                details={"template_id": template_id, "user_id": user_id},
            ) from e

    def update_rating(
        self,
        template_id: str,
        user_id: str,
        rating: int | None = None,
        effectiveness_score: int | None = None,
        comment: str | None = None,
        reported_success: bool | None = None,
        target_model_tested: str | None = None,
    ) -> TemplateRating:
        """
        Update an existing rating.

        Unlike rate_template, this method will fail if no existing rating exists.

        Args:
            template_id: Template identifier.
            user_id: User ID of the rater.
            rating: New star rating (1-5), if changing.
            effectiveness_score: New effectiveness rating (1-5), if changing.
            comment: New review comment, if changing.
            reported_success: New success status, if changing.
            target_model_tested: New tested model, if changing.

        Returns:
            Updated TemplateRating.

        Raises:
            RatingNotFoundError: If no existing rating found.
            RatingValidationError: If validation fails.
        """
        self._operation_count += 1

        # Verify existing rating exists
        existing_rating = self.get_user_rating(template_id, user_id)
        if not existing_rating:
            raise RatingNotFoundError(
                f"No rating found for user '{user_id}' on template '{template_id}'",
                details={"template_id": template_id, "user_id": user_id},
            )

        # Use existing values for fields not being updated
        new_rating = rating if rating is not None else existing_rating.rating
        new_effectiveness = (
            effectiveness_score
            if effectiveness_score is not None
            else existing_rating.effectiveness_score
        )
        new_comment = comment if comment is not None else existing_rating.comment
        new_success = (
            reported_success
            if reported_success is not None
            else existing_rating.reported_success
        )
        new_model = (
            target_model_tested
            if target_model_tested is not None
            else existing_rating.target_model_tested
        )

        # Call rate_template with merged values
        return self.rate_template(
            template_id=template_id,
            user_id=user_id,
            rating=new_rating,
            effectiveness_score=new_effectiveness,
            comment=new_comment,
            reported_success=new_success,
            target_model_tested=new_model,
        )

    def get_user_rating(
        self,
        template_id: str,
        user_id: str,
    ) -> TemplateRating | None:
        """
        Get a user's rating for a specific template.

        Args:
            template_id: Template identifier.
            user_id: User ID.

        Returns:
            TemplateRating if found, None otherwise.
        """
        self._operation_count += 1
        return self.repository.get_rating(template_id, user_id)

    def get_template_ratings(
        self,
        template_id: str,
        user_id: str | None = None,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[TemplateRating]:
        """
        Get all ratings for a template.

        Args:
            template_id: Template identifier.
            user_id: Optional current user ID for visibility check.
            limit: Maximum number of ratings to return.
            offset: Number of ratings to skip.

        Returns:
            List of ratings, newest first.

        Raises:
            TemplateNotFoundError: If template not found.
            RatingPermissionError: If user lacks access to view ratings.
        """
        self._operation_count += 1
        limit = limit or self._default_page_size

        # Verify template exists and user can access it
        self._get_and_verify_template(template_id, user_id)

        try:
            return self.repository.get_ratings(
                template_id=template_id,
                limit=limit,
                offset=offset,
            )
        except EntityNotFoundError as e:
            raise TemplateNotFoundError(
                f"Template '{template_id}' not found",
                details={"template_id": template_id},
            ) from e

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

        Raises:
            RatingPermissionError: If user lacks permission to delete.
        """
        self._operation_count += 1

        # Verify rating exists and belongs to user
        existing_rating = self.get_user_rating(template_id, user_id)
        if not existing_rating:
            return False

        # Users can only delete their own ratings
        if existing_rating.user_id != user_id:
            raise RatingPermissionError(
                "Users can only delete their own ratings",
                details={"template_id": template_id, "user_id": user_id},
            )

        success = self.repository.delete_rating(template_id, user_id)

        if success:
            logger.info(
                f"Deleted rating from user '{user_id}' for template '{template_id}'"
            )

        return success

    # =========================================================================
    # Effectiveness Voting
    # =========================================================================

    def vote_effectiveness(
        self,
        template_id: str,
        user_id: str,
        effectiveness_score: int,
        reported_success: bool | None = None,
        target_model_tested: str | None = None,
    ) -> TemplateRating:
        """
        Vote on a template's effectiveness.

        This is a simplified rating that focuses on effectiveness scoring
        rather than a full star rating. If the user has an existing rating,
        only the effectiveness fields will be updated.

        Args:
            template_id: Template identifier.
            user_id: User ID of the voter.
            effectiveness_score: Effectiveness rating (1-5).
            reported_success: Whether the template worked.
            target_model_tested: Model tested against.

        Returns:
            Created or updated TemplateRating.

        Raises:
            TemplateNotFoundError: If template not found.
            RatingValidationError: If validation fails.
        """
        self._operation_count += 1

        # Validate effectiveness score
        if effectiveness_score < 1 or effectiveness_score > 5:
            raise RatingValidationError(
                "Effectiveness score must be between 1 and 5",
                details={"field": "effectiveness_score", "value": effectiveness_score},
            )

        # Check for existing rating
        existing_rating = self.get_user_rating(template_id, user_id)

        if existing_rating:
            # Update only effectiveness fields
            return self.update_rating(
                template_id=template_id,
                user_id=user_id,
                effectiveness_score=effectiveness_score,
                reported_success=reported_success,
                target_model_tested=target_model_tested,
            )
        else:
            # Create new rating with default star rating
            # Default to 3 stars since user is only voting on effectiveness
            return self.rate_template(
                template_id=template_id,
                user_id=user_id,
                rating=3,  # Neutral default
                effectiveness_score=effectiveness_score,
                reported_success=reported_success,
                target_model_tested=target_model_tested,
            )

    def report_success(
        self,
        template_id: str,
        user_id: str,
        success: bool,
        target_model_tested: str | None = None,
        comment: str | None = None,
    ) -> TemplateRating:
        """
        Report whether a template was successful.

        This is a simplified interface for users to report success/failure
        without providing a full rating.

        Args:
            template_id: Template identifier.
            user_id: User ID of the reporter.
            success: Whether the template worked.
            target_model_tested: Model tested against.
            comment: Optional comment about the result.

        Returns:
            Created or updated TemplateRating.

        Raises:
            TemplateNotFoundError: If template not found.
        """
        self._operation_count += 1

        # Check for existing rating
        existing_rating = self.get_user_rating(template_id, user_id)

        if existing_rating:
            # Update only success-related fields
            return self.update_rating(
                template_id=template_id,
                user_id=user_id,
                reported_success=success,
                target_model_tested=target_model_tested,
                comment=comment,
            )
        else:
            # Create new rating with inferred star rating based on success
            inferred_rating = 4 if success else 2
            return self.rate_template(
                template_id=template_id,
                user_id=user_id,
                rating=inferred_rating,
                reported_success=success,
                target_model_tested=target_model_tested,
                comment=comment,
            )

    # =========================================================================
    # Rating Statistics
    # =========================================================================

    def get_rating_statistics(
        self,
        template_id: str,
        user_id: str | None = None,
    ) -> RatingStatistics:
        """
        Get aggregated rating statistics for a template.

        Args:
            template_id: Template identifier.
            user_id: Optional user ID for visibility check.

        Returns:
            RatingStatistics with aggregated data.

        Raises:
            TemplateNotFoundError: If template not found.
        """
        self._operation_count += 1

        # Verify template exists and user can access it
        self._get_and_verify_template(template_id, user_id)

        try:
            return self.repository.get_rating_statistics(template_id)
        except EntityNotFoundError as e:
            raise TemplateNotFoundError(
                f"Template '{template_id}' not found",
                details={"template_id": template_id},
            ) from e

    def get_rating_distribution(
        self,
        template_id: str,
        user_id: str | None = None,
    ) -> dict[int, int]:
        """
        Get rating distribution for a template.

        Args:
            template_id: Template identifier.
            user_id: Optional user ID for visibility check.

        Returns:
            Dictionary mapping star ratings (1-5) to counts.

        Raises:
            TemplateNotFoundError: If template not found.
        """
        self._operation_count += 1

        stats = self.get_rating_statistics(template_id, user_id)
        return stats.rating_distribution

    def get_success_rate(
        self,
        template_id: str,
        user_id: str | None = None,
    ) -> dict[str, Any]:
        """
        Get success rate statistics for a template.

        Args:
            template_id: Template identifier.
            user_id: Optional user ID for visibility check.

        Returns:
            Dictionary with success rate metrics.

        Raises:
            TemplateNotFoundError: If template not found.
        """
        self._operation_count += 1

        stats = self.get_rating_statistics(template_id, user_id)
        total_reports = stats.success_count + stats.failure_count

        success_rate = None
        if total_reports > 0:
            success_rate = stats.success_count / total_reports

        return {
            "template_id": template_id,
            "success_count": stats.success_count,
            "failure_count": stats.failure_count,
            "total_reports": total_reports,
            "success_rate": success_rate,
            "has_sufficient_data": total_reports >= 5,  # Threshold for statistical significance
        }

    # =========================================================================
    # Analytics and Leaderboards
    # =========================================================================

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
        self._operation_count += 1

        return self.repository.get_top_rated_templates(
            limit=limit,
            min_ratings=min_ratings,
            sharing_level=sharing_level,
        )

    def get_most_effective_templates(
        self,
        limit: int = 10,
        min_ratings: int = 3,
        sharing_level: SharingLevel | None = SharingLevel.PUBLIC,
    ) -> list[tuple[PromptTemplate, float]]:
        """
        Get templates with highest effectiveness scores.

        Args:
            limit: Maximum number of templates to return.
            min_ratings: Minimum number of ratings required.
            sharing_level: Filter by sharing level.

        Returns:
            List of tuples (template, effectiveness_score) sorted by effectiveness.
        """
        self._operation_count += 1

        # Get all templates matching criteria
        templates = self.repository.get_top_rated_templates(
            limit=limit * 3,  # Get more to filter by effectiveness
            min_ratings=min_ratings,
            sharing_level=sharing_level,
        )

        # Filter and sort by effectiveness
        effective_templates: list[tuple[PromptTemplate, float]] = []
        for template in templates:
            if template.rating_stats.average_effectiveness is not None:
                effective_templates.append(
                    (template, template.rating_stats.average_effectiveness)
                )

        # Sort by effectiveness descending
        effective_templates.sort(key=lambda x: x[1], reverse=True)

        return effective_templates[:limit]

    def get_highest_success_rate_templates(
        self,
        limit: int = 10,
        min_reports: int = 5,
        sharing_level: SharingLevel | None = SharingLevel.PUBLIC,
    ) -> list[tuple[PromptTemplate, float]]:
        """
        Get templates with highest success rates.

        Args:
            limit: Maximum number of templates to return.
            min_reports: Minimum number of success/failure reports required.
            sharing_level: Filter by sharing level.

        Returns:
            List of tuples (template, success_rate) sorted by success rate.
        """
        self._operation_count += 1

        # Get all public templates with ratings
        templates = self.repository.get_top_rated_templates(
            limit=100,  # Get more to filter
            min_ratings=1,
            sharing_level=sharing_level,
        )

        # Calculate and filter by success rate
        success_templates: list[tuple[PromptTemplate, float]] = []
        for template in templates:
            total_reports = (
                template.rating_stats.success_count +
                template.rating_stats.failure_count
            )
            if total_reports >= min_reports:
                success_rate = template.rating_stats.success_count / total_reports
                success_templates.append((template, success_rate))

        # Sort by success rate descending
        success_templates.sort(key=lambda x: x[1], reverse=True)

        return success_templates[:limit]

    def get_trending_templates(
        self,
        limit: int = 10,
        days: int = 30,
        sharing_level: SharingLevel | None = SharingLevel.PUBLIC,
    ) -> list[tuple[PromptTemplate, int]]:
        """
        Get templates trending based on recent rating activity.

        Args:
            limit: Maximum number of templates to return.
            days: Number of days to consider for trending.
            sharing_level: Filter by sharing level.

        Returns:
            List of tuples (template, rating_count_in_period) sorted by activity.
        """
        self._operation_count += 1

        from datetime import timedelta

        cutoff_date = datetime.utcnow() - timedelta(days=days)

        # Get templates with recent ratings
        # Note: This is a simplified implementation that counts all ratings
        # A full implementation would track rating timestamps
        templates = self.repository.get_top_rated_templates(
            limit=limit,
            min_ratings=1,
            sharing_level=sharing_level,
        )

        # For now, use total ratings as a proxy for trending
        # In a full implementation, you'd filter by rating creation date
        trending: list[tuple[PromptTemplate, int]] = [
            (t, t.rating_stats.total_ratings) for t in templates
        ]

        # Sort by recent rating count descending
        trending.sort(key=lambda x: x[1], reverse=True)

        return trending[:limit]

    # =========================================================================
    # User Analytics
    # =========================================================================

    def get_user_ratings(
        self,
        user_id: str,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[TemplateRating]:
        """
        Get all ratings submitted by a user.

        Args:
            user_id: User ID.
            limit: Maximum number of ratings to return.
            offset: Number of ratings to skip.

        Returns:
            List of user's ratings.
        """
        self._operation_count += 1
        limit = limit or self._default_page_size

        # Get all templates and filter user's ratings
        # Note: In a production system, you'd want a more efficient query
        all_ratings: list[TemplateRating] = []

        # Access the user ratings index from repository
        user_template_ratings = self.repository._user_ratings.get(user_id, {})

        for template_id in user_template_ratings:
            rating = self.repository.get_rating(template_id, user_id)
            if rating:
                all_ratings.append(rating)

        # Sort by created_at descending
        all_ratings.sort(key=lambda r: r.created_at, reverse=True)

        # Apply pagination
        return all_ratings[offset:offset + limit]

    def get_user_rating_summary(
        self,
        user_id: str,
    ) -> dict[str, Any]:
        """
        Get a summary of a user's rating activity.

        Args:
            user_id: User ID.

        Returns:
            Dictionary with user rating summary.
        """
        self._operation_count += 1

        user_ratings = self.get_user_ratings(user_id, limit=1000)

        if not user_ratings:
            return {
                "user_id": user_id,
                "total_ratings": 0,
                "average_rating_given": None,
                "average_effectiveness_given": None,
                "success_reports": 0,
                "failure_reports": 0,
                "models_tested": [],
            }

        # Calculate statistics
        total_ratings = len(user_ratings)
        avg_rating = sum(r.rating for r in user_ratings) / total_ratings

        effectiveness_ratings = [r for r in user_ratings if r.effectiveness_score]
        avg_effectiveness = None
        if effectiveness_ratings:
            avg_effectiveness = (
                sum(r.effectiveness_score for r in effectiveness_ratings) /
                len(effectiveness_ratings)
            )

        success_reports = sum(1 for r in user_ratings if r.reported_success is True)
        failure_reports = sum(1 for r in user_ratings if r.reported_success is False)

        models_tested = list(set(
            r.target_model_tested for r in user_ratings
            if r.target_model_tested
        ))

        return {
            "user_id": user_id,
            "total_ratings": total_ratings,
            "average_rating_given": round(avg_rating, 2),
            "average_effectiveness_given": (
                round(avg_effectiveness, 2) if avg_effectiveness else None
            ),
            "success_reports": success_reports,
            "failure_reports": failure_reports,
            "models_tested": models_tested,
        }

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
            "service": "template_rating",
            "operation_count": self._operation_count,
            "ratings_submitted": self._ratings_submitted,
            "ratings_updated": self._ratings_updated,
            "repository": {
                "total_templates": repo_stats["total_templates"],
                "total_ratings": repo_stats["total_ratings"],
            },
        }

    def reset_stats(self) -> None:
        """Reset service statistics."""
        self._operation_count = 0
        self._ratings_submitted = 0
        self._ratings_updated = 0

    # =========================================================================
    # Private Helpers
    # =========================================================================

    def _get_and_verify_template(
        self,
        template_id: str,
        user_id: str | None,
    ) -> PromptTemplate:
        """
        Get template and verify user can access it.

        Args:
            template_id: Template identifier.
            user_id: User ID for visibility check.

        Returns:
            PromptTemplate.

        Raises:
            TemplateNotFoundError: If template not found.
            RatingPermissionError: If user lacks access.
        """
        template = self.repository.get_template(template_id)

        if not template:
            raise TemplateNotFoundError(
                f"Template '{template_id}' not found",
                details={"template_id": template_id},
            )

        # Check visibility for rating
        if not self._can_rate_template(template, user_id):
            raise RatingPermissionError(
                f"User lacks permission to access template '{template_id}'",
                details={"template_id": template_id, "user_id": user_id},
            )

        return template

    def _can_rate_template(
        self,
        template: PromptTemplate,
        user_id: str | None,
    ) -> bool:
        """
        Check if user can rate a template.

        Users can rate:
        - Public templates (anyone)
        - Private templates (creator only)
        - Team templates (creator + team members)

        Args:
            template: The template to check.
            user_id: User ID.

        Returns:
            True if user can rate the template.
        """
        # Public templates are ratable by anyone with a user_id
        if template.sharing_level == SharingLevel.PUBLIC:
            return user_id is not None

        # Private templates only ratable by creator
        if template.sharing_level == SharingLevel.PRIVATE:
            return user_id is not None and template.created_by == user_id

        # Team templates ratable by creator (full team support would check membership)
        if template.sharing_level == SharingLevel.TEAM:
            return user_id is not None and (
                template.created_by == user_id
                # Future: or user is team member
            )

        return False


# =============================================================================
# Singleton Instance
# =============================================================================

_service_instance: TemplateRatingService | None = None


def get_template_rating_service(
    reset: bool = False,
) -> TemplateRatingService:
    """
    Get the singleton service instance.

    Args:
        reset: Whether to reset the singleton instance.

    Returns:
        TemplateRatingService instance.
    """
    global _service_instance

    if reset or _service_instance is None:
        _service_instance = TemplateRatingService()

    return _service_instance
