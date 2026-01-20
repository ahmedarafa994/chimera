from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status

from app.core.auth import TokenPayload, get_current_user
from app.domain.prompt_library_models import TemplateVersion
from app.schemas.prompt_library import (
    CreateTemplateRequest,
    CreateVersionRequest,
    RateTemplateRequest,
    RatingListResponse,
    RatingStatistics,
    SaveFromCampaignRequest,
    SearchTemplatesRequest,
    SearchTemplatesResponse,
    TemplateDetailResponse,
    TemplateListItem,
    TemplateStatsResponse,
    TopRatedTemplatesResponse,
    UpdateTemplateRequest,
)
from app.services.prompt_library_service import prompt_library_service
from app.services.template_rating_service import template_rating_service
from app.services.template_version_service import template_version_service

router = APIRouter()


@router.post(
    "/templates",
    response_model=TemplateDetailResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_template(
    request: CreateTemplateRequest,
    current_user: Annotated[TokenPayload, Depends(get_current_user)],
):
    user_id = current_user.sub
    template = await prompt_library_service.create_template(user_id, request)
    return TemplateDetailResponse(
        **template.dict(),
        avg_rating=template.avg_rating,
        total_ratings=template.total_ratings,
        effectiveness_score=template.effectiveness_score,
    )


@router.get("/templates", response_model=SearchTemplatesResponse)
async def list_templates(
    query: str | None = None,
    limit: int = 20,
    offset: int = 0,
    current_user: TokenPayload = Depends(get_current_user),
):
    search_request = SearchTemplatesRequest(query=query, limit=limit, offset=offset)
    items, total = await prompt_library_service.search_templates(search_request)

    list_items = [
        TemplateListItem(
            id=t.id,
            title=t.title,
            description=t.description,
            technique_types=t.metadata.technique_types,
            vulnerability_types=t.metadata.vulnerability_types,
            sharing_level=t.sharing_level,
            status=t.status,
            avg_rating=t.avg_rating,
            total_ratings=t.total_ratings,
            effectiveness_score=t.effectiveness_score,
            tags=t.metadata.tags,
            created_at=t.created_at,
            owner_id=t.owner_id,
        )
        for t in items
    ]

    return SearchTemplatesResponse(items=list_items, total=total, limit=limit, offset=offset)


@router.get("/templates/{template_id}", response_model=TemplateDetailResponse)
async def get_template(
    template_id: str, current_user: Annotated[TokenPayload, Depends(get_current_user)]
):
    template = await prompt_library_service.get_template(template_id)
    if not template:
        raise HTTPException(status_code=404, detail="Template not found")
    return TemplateDetailResponse(
        **template.dict(),
        avg_rating=template.avg_rating,
        total_ratings=template.total_ratings,
        effectiveness_score=template.effectiveness_score,
    )


@router.put("/templates/{template_id}", response_model=TemplateDetailResponse)
async def update_template(
    template_id: str,
    request: UpdateTemplateRequest,
    current_user: Annotated[TokenPayload, Depends(get_current_user)],
):
    template = await prompt_library_service.update_template(template_id, request)
    if not template:
        raise HTTPException(status_code=404, detail="Template not found")
    return TemplateDetailResponse(
        **template.dict(),
        avg_rating=template.avg_rating,
        total_ratings=template.total_ratings,
        effectiveness_score=template.effectiveness_score,
    )


@router.delete("/templates/{template_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_template(
    template_id: str, current_user: Annotated[TokenPayload, Depends(get_current_user)]
) -> None:
    success = await prompt_library_service.delete_template(template_id)
    if not success:
        raise HTTPException(status_code=404, detail="Template not found")


@router.post("/templates/search", response_model=SearchTemplatesResponse)
async def search_templates(
    request: SearchTemplatesRequest,
    current_user: Annotated[TokenPayload, Depends(get_current_user)],
):
    items, total = await prompt_library_service.search_templates(request)

    list_items = [
        TemplateListItem(
            id=t.id,
            title=t.title,
            description=t.description,
            technique_types=t.metadata.technique_types,
            vulnerability_types=t.metadata.vulnerability_types,
            sharing_level=t.sharing_level,
            status=t.status,
            avg_rating=t.avg_rating,
            total_ratings=t.total_ratings,
            effectiveness_score=t.effectiveness_score,
            tags=t.metadata.tags,
            created_at=t.created_at,
            owner_id=t.owner_id,
        )
        for t in items
    ]

    return SearchTemplatesResponse(
        items=list_items,
        total=total,
        limit=request.limit,
        offset=request.offset,
    )


@router.post("/templates/{template_id}/rate", response_model=TemplateDetailResponse)
async def rate_template(
    template_id: str,
    request: RateTemplateRequest,
    current_user: Annotated[TokenPayload, Depends(get_current_user)],
):
    user_id = current_user.sub
    template = await template_rating_service.rate_template(template_id, user_id, request)
    if not template:
        raise HTTPException(status_code=404, detail="Template not found")
    return TemplateDetailResponse(
        **template.dict(),
        avg_rating=template.avg_rating,
        total_ratings=template.total_ratings,
        effectiveness_score=template.effectiveness_score,
    )


@router.get("/templates/{template_id}/versions", response_model=list[TemplateVersion])
async def list_versions(
    template_id: str, current_user: Annotated[TokenPayload, Depends(get_current_user)]
):
    template = await prompt_library_service.get_template(template_id)
    if not template:
        raise HTTPException(status_code=404, detail="Template not found")
    return template.versions


@router.post("/templates/{template_id}/versions", response_model=TemplateDetailResponse)
async def create_version(
    template_id: str,
    request: CreateVersionRequest,
    current_user: Annotated[TokenPayload, Depends(get_current_user)],
):
    user_id = current_user.sub
    template = await template_version_service.create_version(template_id, user_id, request)
    if not template:
        raise HTTPException(status_code=404, detail="Template not found")
    return TemplateDetailResponse(
        **template.dict(),
        avg_rating=template.avg_rating,
        total_ratings=template.total_ratings,
        effectiveness_score=template.effectiveness_score,
    )


@router.get("/statistics", response_model=TemplateStatsResponse)
async def get_statistics(current_user: Annotated[TokenPayload, Depends(get_current_user)]):
    """Get library-wide statistics."""
    return await prompt_library_service.get_statistics()


@router.get("/top-rated", response_model=TopRatedTemplatesResponse)
async def get_top_rated(
    limit: Annotated[int, Query(ge=1, le=50)] = 5,
    current_user: TokenPayload = Depends(get_current_user),
):
    """Get top-rated templates."""
    items = await prompt_library_service.get_top_rated(limit=limit)
    list_items = [
        TemplateListItem(
            id=t.id,
            title=t.title,
            description=t.description,
            technique_types=t.metadata.technique_types,
            vulnerability_types=t.metadata.vulnerability_types,
            sharing_level=t.sharing_level,
            status=t.status,
            avg_rating=t.avg_rating,
            total_ratings=t.total_ratings,
            effectiveness_score=t.effectiveness_score,
            tags=t.metadata.tags,
            created_at=t.created_at,
            owner_id=t.owner_id,
        )
        for t in items
    ]
    return TopRatedTemplatesResponse(items=list_items)


@router.post(
    "/save-from-campaign",
    response_model=TemplateDetailResponse,
    status_code=status.HTTP_201_CREATED,
)
async def save_from_campaign(
    request: SaveFromCampaignRequest,
    current_user: Annotated[TokenPayload, Depends(get_current_user)],
):
    """Save a successful attack from a campaign as a template."""
    user_id = current_user.sub
    template = await prompt_library_service.save_from_campaign(user_id, request)
    if not template:
        raise HTTPException(status_code=404, detail="Campaign or attack not found")
    return TemplateDetailResponse(
        **template.dict(),
        avg_rating=template.avg_rating,
        total_ratings=template.total_ratings,
        effectiveness_score=template.effectiveness_score,
    )


@router.get("/templates/{template_id}/ratings", response_model=RatingListResponse)
async def get_template_ratings(
    template_id: str,
    current_user: Annotated[TokenPayload, Depends(get_current_user)],
):
    """Get all ratings for a template."""
    template = await prompt_library_service.get_template(template_id)
    if not template:
        raise HTTPException(status_code=404, detail="Template not found")

    ratings = await template_rating_service.get_ratings(template_id)
    stats = await template_rating_service.get_rating_statistics(template_id)

    return RatingListResponse(ratings=ratings, statistics=stats)


@router.get("/templates/{template_id}/ratings/stats", response_model=RatingStatistics)
async def get_rating_stats(
    template_id: str,
    current_user: Annotated[TokenPayload, Depends(get_current_user)],
):
    """Get rating statistics for a template."""
    template = await prompt_library_service.get_template(template_id)
    if not template:
        raise HTTPException(status_code=404, detail="Template not found")

    return await template_rating_service.get_rating_statistics(template_id)


@router.put("/templates/{template_id}/rate", response_model=TemplateDetailResponse)
async def update_rating(
    template_id: str,
    request: RateTemplateRequest,
    current_user: Annotated[TokenPayload, Depends(get_current_user)],
):
    """Update an existing rating."""
    user_id = current_user.sub
    template = await template_rating_service.update_rating(template_id, user_id, request)
    if not template:
        raise HTTPException(status_code=404, detail="Template or rating not found")
    return TemplateDetailResponse(
        **template.dict(),
        avg_rating=template.avg_rating,
        total_ratings=template.total_ratings,
        effectiveness_score=template.effectiveness_score,
    )


@router.delete("/templates/{template_id}/rate", status_code=status.HTTP_204_NO_CONTENT)
async def delete_rating(
    template_id: str, current_user: Annotated[TokenPayload, Depends(get_current_user)]
) -> None:
    """Delete user's rating for a template."""
    user_id = current_user.sub
    success = await template_rating_service.delete_rating(template_id, user_id)
    if not success:
        raise HTTPException(status_code=404, detail="Rating not found")


@router.post(
    "/templates/{template_id}/versions/{version_id}/restore",
    response_model=TemplateDetailResponse,
)
async def restore_version(
    template_id: str,
    version_id: str,
    current_user: Annotated[TokenPayload, Depends(get_current_user)],
):
    """Restore a previous version of a template."""
    user_id = current_user.sub
    template = await template_version_service.restore_version(template_id, version_id, user_id)
    if not template:
        raise HTTPException(status_code=404, detail="Template or version not found")
    return TemplateDetailResponse(
        **template.dict(),
        avg_rating=template.avg_rating,
        total_ratings=template.total_ratings,
        effectiveness_score=template.effectiveness_score,
    )


@router.get("/templates/{template_id}/versions/{version_id}", response_model=TemplateVersion)
async def get_template_version(
    template_id: str,
    version_id: str,
    current_user: Annotated[TokenPayload, Depends(get_current_user)],
):
    """Get a specific version of a template."""
    version = await template_version_service.get_version(template_id, version_id)
    if not version:
        raise HTTPException(status_code=404, detail="Version not found")
    return version
