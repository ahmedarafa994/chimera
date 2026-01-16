"""
Prompt Library API Endpoints

REST API endpoints for managing prompt templates, including CRUD operations,
search/filtering, ratings, versioning, and campaign integration.
"""

import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status

from app.core.auth import TokenPayload, get_current_user
from app.domain.prompt_library_models import (
    SharingLevel,
    TemplateStatus,
    TechniqueType,
    VulnerabilityType,
)
from app.schemas.prompt_library import (
    CreateTemplateRequest,
    CreateVersionRequest,
    RateTemplateRequest,
    RatingListResponse,
    RatingResponse,
    RatingStatisticsResponse,
    SaveFromCampaignRequest,
    SearchTemplatesRequest,
    TemplateDeleteResponse,
    TemplateListResponse,
    TemplateResponse,
    TemplateStatsResponse,
    TemplateVersionListResponse,
    TemplateVersionResponse,
    TopRatedTemplatesResponse,
    UpdateRatingRequest,
    UpdateTemplateRequest,
)
from app.services.prompt_library_service import (
    PromptLibraryService,
    PromptLibraryServiceError,
    TemplateNotFoundError,
    TemplatePermissionError,
    TemplateValidationError,
    get_prompt_library_service,
)
from app.services.template_rating_service import (
    RatingNotFoundError,
    RatingPermissionError,
    RatingValidationError,
    TemplateRatingService,
    get_template_rating_service,
)
from app.services.template_version_service import (
    TemplateVersionService,
    VersionNotFoundError,
    VersionPermissionError,
    VersionValidationError,
    get_template_version_service,
)

logger = logging.getLogger(__name__)

router = APIRouter()


# =============================================================================
# Dependency Injection
# =============================================================================


def get_library_service() -> PromptLibraryService:
    """Get the prompt library service instance."""
    return get_prompt_library_service()


def get_rating_service() -> TemplateRatingService:
    """Get the template rating service instance."""
    return get_template_rating_service()


def get_version_service() -> TemplateVersionService:
    """Get the template version service instance."""
    return get_template_version_service()


# =============================================================================
# Template CRUD Endpoints
# =============================================================================


@router.post(
    "/templates",
    response_model=TemplateResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new prompt template",
    description="Create a new prompt template with metadata, techniques, and sharing settings.",
)
async def create_template(
    request: CreateTemplateRequest,
    current_user: TokenPayload = Depends(get_current_user),
    service: PromptLibraryService = Depends(get_library_service),
) -> TemplateResponse:
    """Create a new prompt template."""
    try:
        template = service.create_template(
            name=request.name,
            prompt_content=request.prompt_content,
            description=request.description,
            system_instruction=request.system_instruction,
            technique_types=request.technique_types,
            vulnerability_types=request.vulnerability_types,
            target_models=request.target_models,
            target_providers=request.target_providers,
            cve_references=request.cve_references,
            paper_references=request.paper_references,
            tags=request.tags,
            discovery_source=request.discovery_source,
            status=request.status,
            sharing_level=request.sharing_level,
            created_by=current_user.sub,
            team_id=request.team_id,
        )

        logger.info(f"Created template '{template.id}' for user '{current_user.sub}'")

        return TemplateResponse(
            id=template.id,
            created_at=template.created_at,
            updated_at=template.updated_at,
            template=template,
        )

    except TemplateValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"message": e.message, "details": e.details},
        )
    except PromptLibraryServiceError as e:
        logger.error(f"Error creating template: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"message": "Failed to create template", "error": str(e)},
        )


@router.get(
    "/templates",
    response_model=TemplateListResponse,
    summary="List prompt templates",
    description="Get a paginated list of prompt templates with optional filtering.",
)
async def list_templates(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Results per page"),
    sharing_level: Optional[SharingLevel] = Query(None, description="Filter by sharing level"),
    template_status: Optional[TemplateStatus] = Query(
        None, alias="status", description="Filter by status"
    ),
    created_by: Optional[str] = Query(None, description="Filter by creator user ID"),
    current_user: TokenPayload = Depends(get_current_user),
    service: PromptLibraryService = Depends(get_library_service),
) -> TemplateListResponse:
    """List templates with basic filtering."""
    offset = (page - 1) * page_size

    # Build filter lists
    sharing_levels = [sharing_level] if sharing_level else None
    statuses = [template_status] if template_status else None

    templates = service.list_templates(
        user_id=current_user.sub,
        limit=page_size,
        offset=offset,
        sharing_levels=sharing_levels,
        status=statuses,
        created_by=created_by,
        include_private=True,  # Include user's own private templates
    )

    # Get total count for pagination
    all_templates = service.list_templates(
        user_id=current_user.sub,
        limit=100000,
        offset=0,
        sharing_levels=sharing_levels,
        status=statuses,
        created_by=created_by,
        include_private=True,
    )
    total_count = len(all_templates)
    total_pages = (total_count + page_size - 1) // page_size if total_count > 0 else 0

    return TemplateListResponse(
        templates=templates,
        total_count=total_count,
        page=page,
        page_size=page_size,
        total_pages=total_pages,
    )


@router.get(
    "/templates/{template_id}",
    response_model=TemplateResponse,
    summary="Get a prompt template by ID",
    description="Retrieve a specific prompt template by its unique identifier.",
)
async def get_template(
    template_id: str,
    current_user: TokenPayload = Depends(get_current_user),
    service: PromptLibraryService = Depends(get_library_service),
) -> TemplateResponse:
    """Get a specific template by ID."""
    try:
        template = service.get_template(
            template_id=template_id,
            user_id=current_user.sub,
            check_visibility=True,
        )

        return TemplateResponse(
            id=template.id,
            created_at=template.created_at,
            updated_at=template.updated_at,
            template=template,
        )

    except TemplateNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Template '{template_id}' not found",
        )
    except TemplatePermissionError:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Access denied to template '{template_id}'",
        )


@router.put(
    "/templates/{template_id}",
    response_model=TemplateResponse,
    summary="Update a prompt template",
    description="Update an existing prompt template. Only provided fields will be updated.",
)
async def update_template(
    template_id: str,
    request: UpdateTemplateRequest,
    current_user: TokenPayload = Depends(get_current_user),
    service: PromptLibraryService = Depends(get_library_service),
) -> TemplateResponse:
    """Update an existing template."""
    try:
        template = service.update_template(
            template_id=template_id,
            user_id=current_user.sub,
            name=request.name,
            description=request.description,
            prompt_content=request.prompt_content,
            system_instruction=request.system_instruction,
            technique_types=request.technique_types,
            vulnerability_types=request.vulnerability_types,
            target_models=request.target_models,
            target_providers=request.target_providers,
            cve_references=request.cve_references,
            paper_references=request.paper_references,
            tags=request.tags,
            success_rate=request.success_rate,
            status=request.status,
            sharing_level=request.sharing_level,
            team_id=request.team_id,
            create_version=request.create_version,
            change_summary=request.change_summary or "",
        )

        logger.info(f"Updated template '{template_id}' by user '{current_user.sub}'")

        return TemplateResponse(
            id=template.id,
            created_at=template.created_at,
            updated_at=template.updated_at,
            template=template,
        )

    except TemplateNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Template '{template_id}' not found",
        )
    except TemplatePermissionError:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Permission denied to update template '{template_id}'",
        )
    except TemplateValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"message": e.message, "details": e.details},
        )


@router.delete(
    "/templates/{template_id}",
    response_model=TemplateDeleteResponse,
    summary="Delete a prompt template",
    description="Permanently delete a prompt template and all associated data.",
)
async def delete_template(
    template_id: str,
    current_user: TokenPayload = Depends(get_current_user),
    service: PromptLibraryService = Depends(get_library_service),
) -> TemplateDeleteResponse:
    """Delete a template."""
    try:
        success = service.delete_template(
            template_id=template_id,
            user_id=current_user.sub,
        )

        if success:
            logger.info(f"Deleted template '{template_id}' by user '{current_user.sub}'")
            return TemplateDeleteResponse(
                success=True,
                template_id=template_id,
                message="Template deleted successfully",
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to delete template",
            )

    except TemplateNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Template '{template_id}' not found",
        )
    except TemplatePermissionError:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Permission denied to delete template '{template_id}'",
        )


# =============================================================================
# Search Endpoint
# =============================================================================


@router.post(
    "/templates/search",
    response_model=TemplateListResponse,
    summary="Search prompt templates",
    description="Advanced search with filtering, sorting, and pagination.",
)
async def search_templates(
    request: SearchTemplatesRequest,
    current_user: TokenPayload = Depends(get_current_user),
    service: PromptLibraryService = Depends(get_library_service),
) -> TemplateListResponse:
    """Search templates with advanced filtering."""
    result = service.search_templates(
        query=request.query,
        technique_types=request.technique_types,
        vulnerability_types=request.vulnerability_types,
        target_models=request.target_models,
        target_providers=request.target_providers,
        tags=request.tags,
        status=request.status,
        sharing_levels=request.sharing_levels,
        min_rating=request.min_rating,
        min_success_rate=request.min_success_rate,
        created_by=request.created_by,
        team_id=request.team_id,
        created_after=request.created_after,
        created_before=request.created_before,
        sort_by=request.sort_by,
        sort_order=request.sort_order,
        page=request.page,
        page_size=request.page_size,
        user_id=current_user.sub,
        include_private=True,
    )

    return TemplateListResponse(
        templates=result.templates,
        total_count=result.total_count,
        page=result.page,
        page_size=result.page_size,
        total_pages=result.total_pages,
    )


# =============================================================================
# Save From Campaign Endpoint
# =============================================================================


@router.post(
    "/templates/save-from-campaign",
    response_model=TemplateResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Save a prompt from a campaign",
    description="Save a successful prompt from campaign execution to the library with auto-populated metadata.",
)
async def save_from_campaign(
    request: SaveFromCampaignRequest,
    current_user: TokenPayload = Depends(get_current_user),
    service: PromptLibraryService = Depends(get_library_service),
) -> TemplateResponse:
    """Save a prompt from a campaign execution to the library."""
    try:
        template = service.save_from_campaign(
            name=request.name,
            prompt_content=request.prompt_content,
            description=request.description,
            system_instruction=request.system_instruction,
            technique_types=request.technique_types,
            vulnerability_types=request.vulnerability_types,
            target_model=request.target_model,
            target_provider=request.target_provider,
            tags=request.tags,
            sharing_level=request.sharing_level,
            team_id=request.team_id,
            campaign_id=request.campaign_id,
            execution_id=request.execution_id,
            was_successful=request.was_successful,
            initial_success_rate=request.initial_success_rate,
            created_by=current_user.sub,
        )

        logger.info(
            f"Saved prompt from campaign '{request.campaign_id}' as template '{template.id}'"
        )

        return TemplateResponse(
            id=template.id,
            created_at=template.created_at,
            updated_at=template.updated_at,
            template=template,
        )

    except TemplateValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"message": e.message, "details": e.details},
        )
    except PromptLibraryServiceError as e:
        logger.error(f"Error saving from campaign: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"message": "Failed to save prompt from campaign", "error": str(e)},
        )


# =============================================================================
# Rating Endpoints
# =============================================================================


@router.post(
    "/templates/{template_id}/rate",
    response_model=RatingResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Rate a template",
    description="Submit a rating and optional review for a prompt template.",
)
async def rate_template(
    template_id: str,
    request: RateTemplateRequest,
    current_user: TokenPayload = Depends(get_current_user),
    library_service: PromptLibraryService = Depends(get_library_service),
    rating_service: TemplateRatingService = Depends(get_rating_service),
) -> RatingResponse:
    """Rate a template."""
    # First verify template exists and user can access it
    try:
        library_service.get_template(
            template_id=template_id,
            user_id=current_user.sub,
            check_visibility=True,
        )
    except TemplateNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Template '{template_id}' not found",
        )
    except TemplatePermissionError:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Access denied to template '{template_id}'",
        )

    try:
        rating = rating_service.rate_template(
            template_id=template_id,
            user_id=current_user.sub,
            rating=request.rating,
            effectiveness_score=request.effectiveness_score,
            comment=request.comment,
            reported_success=request.reported_success,
            target_model_tested=request.target_model_tested,
        )

        logger.info(
            f"User '{current_user.sub}' rated template '{template_id}' with {request.rating} stars"
        )

        return RatingResponse(
            id=rating.rating_id,
            created_at=rating.created_at,
            updated_at=rating.updated_at,
            rating=rating,
        )

    except RatingValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"message": str(e)},
        )


@router.get(
    "/templates/{template_id}/ratings",
    response_model=RatingListResponse,
    summary="Get template ratings",
    description="Get all ratings for a template with aggregated statistics.",
)
async def get_template_ratings(
    template_id: str,
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Results per page"),
    current_user: TokenPayload = Depends(get_current_user),
    library_service: PromptLibraryService = Depends(get_library_service),
    rating_service: TemplateRatingService = Depends(get_rating_service),
) -> RatingListResponse:
    """Get ratings for a template."""
    # Verify template access
    try:
        library_service.get_template(
            template_id=template_id,
            user_id=current_user.sub,
            check_visibility=True,
        )
    except TemplateNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Template '{template_id}' not found",
        )
    except TemplatePermissionError:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Access denied to template '{template_id}'",
        )

    # Get ratings and statistics
    offset = (page - 1) * page_size
    ratings = rating_service.get_template_ratings(
        template_id=template_id,
        limit=page_size,
        offset=offset,
    )
    statistics = rating_service.get_rating_statistics(template_id)
    total_count = statistics.total_ratings
    total_pages = (total_count + page_size - 1) // page_size if total_count > 0 else 0

    return RatingListResponse(
        template_id=template_id,
        ratings=ratings,
        statistics=statistics,
        total_count=total_count,
        page=page,
        page_size=page_size,
        total_pages=total_pages,
    )


@router.get(
    "/templates/{template_id}/ratings/statistics",
    response_model=RatingStatisticsResponse,
    summary="Get rating statistics",
    description="Get aggregated rating statistics for a template.",
)
async def get_rating_statistics(
    template_id: str,
    current_user: TokenPayload = Depends(get_current_user),
    library_service: PromptLibraryService = Depends(get_library_service),
    rating_service: TemplateRatingService = Depends(get_rating_service),
) -> RatingStatisticsResponse:
    """Get rating statistics for a template."""
    # Verify template access
    try:
        library_service.get_template(
            template_id=template_id,
            user_id=current_user.sub,
            check_visibility=True,
        )
    except TemplateNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Template '{template_id}' not found",
        )
    except TemplatePermissionError:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Access denied to template '{template_id}'",
        )

    statistics = rating_service.get_rating_statistics(template_id)

    return RatingStatisticsResponse(
        template_id=template_id,
        statistics=statistics,
    )


@router.put(
    "/templates/{template_id}/ratings/my-rating",
    response_model=RatingResponse,
    summary="Update my rating",
    description="Update your rating for a template. Only the rating owner can update it.",
)
async def update_rating(
    template_id: str,
    request: UpdateRatingRequest,
    current_user: TokenPayload = Depends(get_current_user),
    rating_service: TemplateRatingService = Depends(get_rating_service),
) -> RatingResponse:
    """Update current user's rating for a template."""
    try:
        rating = rating_service.update_rating(
            template_id=template_id,
            user_id=current_user.sub,
            rating=request.rating,
            effectiveness_score=request.effectiveness_score,
            comment=request.comment,
            reported_success=request.reported_success,
            target_model_tested=request.target_model_tested,
        )

        return RatingResponse(
            id=rating.rating_id,
            created_at=rating.created_at,
            updated_at=rating.updated_at,
            rating=rating,
        )

    except RatingNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No rating found for this template",
        )
    except RatingPermissionError:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Permission denied to update this rating",
        )


@router.delete(
    "/templates/{template_id}/ratings/my-rating",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete my rating",
    description="Delete your rating for a template. Only the rating owner can delete it.",
)
async def delete_rating(
    template_id: str,
    current_user: TokenPayload = Depends(get_current_user),
    rating_service: TemplateRatingService = Depends(get_rating_service),
):
    """Delete current user's rating for a template."""
    try:
        success = rating_service.delete_rating(
            template_id=template_id,
            user_id=current_user.sub,
        )
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No rating found for this template",
            )
        return None

    except RatingNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No rating found for this template",
        )
    except RatingPermissionError:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Permission denied to delete this rating",
        )


@router.get(
    "/templates/top-rated",
    response_model=TopRatedTemplatesResponse,
    summary="Get top-rated templates",
    description="Get the highest-rated templates in the library.",
)
async def get_top_rated_templates(
    limit: int = Query(10, ge=1, le=100, description="Maximum number of templates"),
    min_ratings: int = Query(1, ge=1, description="Minimum number of ratings required"),
    current_user: TokenPayload = Depends(get_current_user),
    service: PromptLibraryService = Depends(get_library_service),
) -> TopRatedTemplatesResponse:
    """Get top-rated templates."""
    templates = service.get_top_rated_templates(
        limit=limit,
        min_ratings=min_ratings,
        sharing_level=SharingLevel.PUBLIC,  # Only public templates
    )

    return TopRatedTemplatesResponse(
        templates=templates,
        time_period="all_time",
        limit=limit,
    )


# =============================================================================
# Version Endpoints
# =============================================================================


@router.get(
    "/templates/{template_id}/versions",
    response_model=TemplateVersionListResponse,
    summary="Get version history",
    description="Get the version history for a template.",
)
async def get_template_versions(
    template_id: str,
    limit: int = Query(100, ge=1, le=1000, description="Maximum versions to return"),
    current_user: TokenPayload = Depends(get_current_user),
    library_service: PromptLibraryService = Depends(get_library_service),
    version_service: TemplateVersionService = Depends(get_version_service),
) -> TemplateVersionListResponse:
    """Get version history for a template."""
    # Verify template access
    try:
        template = library_service.get_template(
            template_id=template_id,
            user_id=current_user.sub,
            check_visibility=True,
        )
    except TemplateNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Template '{template_id}' not found",
        )
    except TemplatePermissionError:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Access denied to template '{template_id}'",
        )

    versions = version_service.get_versions(
        template_id=template_id,
        user_id=current_user.sub,
        limit=limit,
    )

    return TemplateVersionListResponse(
        template_id=template_id,
        versions=versions,
        current_version=template.current_version,
        total_count=len(versions),
    )


@router.post(
    "/templates/{template_id}/versions",
    response_model=TemplateVersionResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new version",
    description="Create a new version of a template with updated content.",
)
async def create_template_version(
    template_id: str,
    request: CreateVersionRequest,
    current_user: TokenPayload = Depends(get_current_user),
    library_service: PromptLibraryService = Depends(get_library_service),
    version_service: TemplateVersionService = Depends(get_version_service),
) -> TemplateVersionResponse:
    """Create a new version for a template."""
    # Verify template access and edit permission
    try:
        library_service.get_template(
            template_id=template_id,
            user_id=current_user.sub,
            check_visibility=True,
        )
    except TemplateNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Template '{template_id}' not found",
        )
    except TemplatePermissionError:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Access denied to template '{template_id}'",
        )

    try:
        version = version_service.create_version(
            template_id=template_id,
            prompt_content=request.prompt_content,
            change_summary=request.change_summary,
            user_id=current_user.sub,
        )

        logger.info(
            f"Created version {version.version_number} for template '{template_id}'"
        )

        return TemplateVersionResponse(
            id=version.version_id,
            created_at=version.created_at,
            updated_at=version.created_at,
            version=version,
        )

    except VersionPermissionError:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Permission denied to create version for template '{template_id}'",
        )
    except VersionValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"message": str(e)},
        )


@router.get(
    "/templates/{template_id}/versions/{version_number}",
    response_model=TemplateVersionResponse,
    summary="Get a specific version",
    description="Get a specific version of a template by version number.",
)
async def get_template_version(
    template_id: str,
    version_number: int,
    current_user: TokenPayload = Depends(get_current_user),
    library_service: PromptLibraryService = Depends(get_library_service),
    version_service: TemplateVersionService = Depends(get_version_service),
) -> TemplateVersionResponse:
    """Get a specific version of a template."""
    # Verify template access
    try:
        library_service.get_template(
            template_id=template_id,
            user_id=current_user.sub,
            check_visibility=True,
        )
    except TemplateNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Template '{template_id}' not found",
        )
    except TemplatePermissionError:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Access denied to template '{template_id}'",
        )

    try:
        version = version_service.get_version(
            template_id=template_id,
            version_number=version_number,
            user_id=current_user.sub,
        )

        if not version:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Version {version_number} not found for template '{template_id}'",
            )

        return TemplateVersionResponse(
            id=version.version_id,
            created_at=version.created_at,
            updated_at=version.created_at,
            version=version,
        )

    except VersionNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Version {version_number} not found for template '{template_id}'",
        )


@router.post(
    "/templates/{template_id}/versions/{version_number}/restore",
    response_model=TemplateResponse,
    summary="Restore a version",
    description="Restore a template to a previous version.",
)
async def restore_template_version(
    template_id: str,
    version_number: int,
    current_user: TokenPayload = Depends(get_current_user),
    library_service: PromptLibraryService = Depends(get_library_service),
) -> TemplateResponse:
    """Restore a template to a previous version."""
    try:
        template = library_service.restore_template_version(
            template_id=template_id,
            version_number=version_number,
            user_id=current_user.sub,
        )

        logger.info(
            f"Restored template '{template_id}' to version {version_number}"
        )

        return TemplateResponse(
            id=template.id,
            created_at=template.created_at,
            updated_at=template.updated_at,
            template=template,
        )

    except TemplateNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e.message),
        )
    except TemplatePermissionError:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Permission denied to restore template '{template_id}'",
        )


# =============================================================================
# Statistics Endpoints
# =============================================================================


@router.get(
    "/templates/stats",
    response_model=TemplateStatsResponse,
    summary="Get library statistics",
    description="Get comprehensive statistics about the prompt library.",
)
async def get_library_stats(
    current_user: TokenPayload = Depends(get_current_user),
    service: PromptLibraryService = Depends(get_library_service),
) -> TemplateStatsResponse:
    """Get library statistics."""
    stats = service.get_library_statistics(
        user_id=current_user.sub,
        include_user_stats=True,
    )

    return TemplateStatsResponse(
        total_templates=stats.get("total_templates", 0),
        public_templates=stats.get("by_sharing_level", {}).get("public", 0),
        private_templates=stats.get("by_sharing_level", {}).get("private", 0),
        team_templates=stats.get("by_sharing_level", {}).get("team", 0),
        techniques_used=stats.get("top_techniques", {}),
        vulnerabilities_targeted=stats.get("vulnerabilities_targeted", {}),
        top_rated_count=stats.get("top_rated_count", 0),
        total_ratings=stats.get("total_ratings", 0),
        average_success_rate=stats.get("average_success_rate"),
    )
