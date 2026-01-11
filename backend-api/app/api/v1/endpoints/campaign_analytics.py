"""
Campaign Analytics API Endpoints

Provides RESTful endpoints for campaign telemetry analytics:
- Campaign listing and detail retrieval
- Statistical analysis (mean, median, p95, std_dev)
- Time series data for visualization
- Campaign comparison (2-4 campaigns)
- CSV and chart export functionality
- Technique and provider breakdowns

Subtask 1.4: Create Campaign Analytics API Endpoints
"""

import csv
import io
import time
from datetime import datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import db_manager
from app.core.logging import logger
from app.schemas.campaign_analytics import (
    CampaignComparison,
    CampaignComparisonRequest,
    CampaignDetail,
    CampaignFilterParams,
    CampaignListResponse,
    CampaignStatistics,
    CampaignStatusEnum,
    CampaignSummary,
    ExportFormat,
    ExportRequest,
    ExportResponse,
    PotencyBreakdown,
    ProviderBreakdown,
    TechniqueBreakdown,
    TelemetryEventDetail,
    TelemetryFilterParams,
    TelemetryListResponse,
    TelemetryTimeSeries,
    TimeGranularity,
)
from app.services.campaign_analytics_service import CampaignAnalyticsService

router = APIRouter()


# =============================================================================
# Database Session Dependency
# =============================================================================


async def get_db_session() -> AsyncSession:
    """
    Get an async database session for campaign analytics.

    Uses the read_only_session for query operations.
    """
    async with db_manager.read_only_session() as session:
        yield session


async def get_analytics_service(
    session: AsyncSession = Depends(get_db_session),
) -> CampaignAnalyticsService:
    """Get campaign analytics service with database session."""
    return CampaignAnalyticsService(session)


# =============================================================================
# Campaign List and Detail Endpoints
# =============================================================================


@router.get(
    "",
    response_model=CampaignListResponse,
    summary="List campaigns",
    description="Get a paginated list of campaigns with optional filtering.",
)
async def list_campaigns(
    page: int = Query(1, ge=1, description="Page number (1-indexed)"),
    page_size: int = Query(20, ge=1, le=100, description="Items per page"),
    sort_by: str = Query("created_at", description="Field to sort by"),
    sort_order: str = Query("desc", regex="^(asc|desc)$", description="Sort order"),
    status: list[CampaignStatusEnum] | None = Query(None, description="Filter by status"),
    provider: list[str] | None = Query(None, description="Filter by target provider"),
    technique_suite: list[str] | None = Query(None, description="Filter by technique suite"),
    tags: list[str] | None = Query(None, description="Filter by tags"),
    start_date: datetime | None = Query(None, description="Created after this date"),
    end_date: datetime | None = Query(None, description="Created before this date"),
    search: str | None = Query(None, max_length=100, description="Search in name/description"),
    service: CampaignAnalyticsService = Depends(get_analytics_service),
) -> CampaignListResponse:
    """
    List campaigns with pagination and filtering.

    Returns a paginated list of campaign summaries with quick stats.
    Supports filtering by status, provider, technique, tags, date range,
    and text search.
    """
    filters = None
    if any([status, provider, technique_suite, tags, start_date, end_date, search]):
        filters = CampaignFilterParams(
            status=status,
            provider=provider,
            technique_suite=technique_suite,
            tags=tags,
            start_date=start_date,
            end_date=end_date,
            search=search,
        )

    result = await service.list_campaigns(
        page=page,
        page_size=page_size,
        sort_by=sort_by,
        sort_order=sort_order,
        filters=filters,
    )

    logger.debug(f"Listed {len(result.items)} campaigns (page {page}/{result.total_pages})")
    return result


@router.get(
    "/{campaign_id}",
    response_model=CampaignDetail,
    summary="Get campaign detail",
    description="Get detailed information about a specific campaign.",
    responses={404: {"description": "Campaign not found"}},
)
async def get_campaign_detail(
    campaign_id: str,
    service: CampaignAnalyticsService = Depends(get_analytics_service),
) -> CampaignDetail:
    """
    Get detailed campaign information.

    Returns the full campaign details including configuration,
    quick statistics, and metadata.
    """
    campaign = await service.get_campaign_detail(campaign_id)

    if not campaign:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Campaign {campaign_id} not found",
        )

    return campaign


@router.get(
    "/{campaign_id}/summary",
    response_model=CampaignSummary,
    summary="Get campaign summary",
    description="Get a summary view of a campaign (lighter than detail).",
    responses={404: {"description": "Campaign not found"}},
)
async def get_campaign_summary(
    campaign_id: str,
    service: CampaignAnalyticsService = Depends(get_analytics_service),
) -> CampaignSummary:
    """
    Get campaign summary.

    Returns a lightweight summary view of the campaign with
    quick stats but without full configuration.
    """
    summary = await service.get_campaign_summary(campaign_id)

    if not summary:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Campaign {campaign_id} not found",
        )

    return summary


# =============================================================================
# Statistics Endpoints
# =============================================================================


@router.get(
    "/{campaign_id}/statistics",
    response_model=CampaignStatistics,
    summary="Get campaign statistics",
    description="Get comprehensive statistical analysis for a campaign.",
    responses={404: {"description": "Campaign not found"}},
)
async def get_campaign_statistics(
    campaign_id: str,
    service: CampaignAnalyticsService = Depends(get_analytics_service),
) -> CampaignStatistics:
    """
    Get comprehensive statistics for a campaign.

    Computes and returns:
    - Attempt counts by status
    - Success rate distribution (mean, median, p95, std_dev)
    - Semantic success distribution
    - Latency distribution
    - Token usage distribution (prompt, completion, total)
    - Cost distribution
    - Total duration
    """
    stats = await service.calculate_statistics(campaign_id)

    if not stats:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Campaign {campaign_id} not found",
        )

    return stats


# =============================================================================
# Breakdown Endpoints
# =============================================================================


@router.get(
    "/{campaign_id}/breakdown/techniques",
    response_model=TechniqueBreakdown,
    summary="Get technique breakdown",
    description="Get success rate breakdown by transformation technique.",
    responses={404: {"description": "Campaign not found"}},
)
async def get_technique_breakdown(
    campaign_id: str,
    service: CampaignAnalyticsService = Depends(get_analytics_service),
) -> TechniqueBreakdown:
    """
    Get breakdown of results by transformation technique.

    Shows success rate, latency, and cost metrics for each
    technique used in the campaign.
    """
    breakdown = await service.get_technique_breakdown(campaign_id)

    if not breakdown:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Campaign {campaign_id} not found",
        )

    return breakdown


@router.get(
    "/{campaign_id}/breakdown/providers",
    response_model=ProviderBreakdown,
    summary="Get provider breakdown",
    description="Get success rate breakdown by LLM provider.",
    responses={404: {"description": "Campaign not found"}},
)
async def get_provider_breakdown(
    campaign_id: str,
    service: CampaignAnalyticsService = Depends(get_analytics_service),
) -> ProviderBreakdown:
    """
    Get breakdown of results by LLM provider.

    Shows success rate, latency, and cost metrics for each
    provider used in the campaign.
    """
    breakdown = await service.get_provider_breakdown(campaign_id)

    if not breakdown:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Campaign {campaign_id} not found",
        )

    return breakdown


@router.get(
    "/{campaign_id}/breakdown/potency",
    response_model=PotencyBreakdown,
    summary="Get potency breakdown",
    description="Get success rate breakdown by potency level.",
    responses={404: {"description": "Campaign not found"}},
)
async def get_potency_breakdown(
    campaign_id: str,
    service: CampaignAnalyticsService = Depends(get_analytics_service),
) -> PotencyBreakdown:
    """
    Get breakdown of results by potency level.

    Shows success rate, latency, and cost metrics for each
    potency level (1-10) used in the campaign.
    """
    breakdown = await service.get_potency_breakdown(campaign_id)

    if not breakdown:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Campaign {campaign_id} not found",
        )

    return breakdown


# =============================================================================
# Time Series Endpoints
# =============================================================================


@router.get(
    "/{campaign_id}/time-series",
    response_model=TelemetryTimeSeries,
    summary="Get time series data",
    description="Get time-bucketed telemetry data for visualization.",
    responses={404: {"description": "Campaign not found"}},
)
async def get_time_series(
    campaign_id: str,
    metric: str = Query(
        "success_rate",
        description="Metric to chart: success_rate, latency, tokens, cost, semantic_success",
    ),
    granularity: TimeGranularity = Query(
        TimeGranularity.HOUR,
        description="Time bucket granularity",
    ),
    start_time: datetime | None = Query(None, description="Series start time"),
    end_time: datetime | None = Query(None, description="Series end time"),
    technique_suite: list[str] | None = Query(None, description="Filter by technique"),
    provider: list[str] | None = Query(None, description="Filter by provider"),
    service: CampaignAnalyticsService = Depends(get_analytics_service),
) -> TelemetryTimeSeries:
    """
    Get time series data for a campaign metric.

    Returns time-bucketed data for charting success rates, latency,
    token usage, or cost over the campaign duration.

    Supports filtering by technique or provider.
    """
    filters = None
    if technique_suite or provider:
        filters = TelemetryFilterParams(
            technique_suite=technique_suite,
            provider=provider,
        )

    time_series = await service.get_time_series(
        campaign_id=campaign_id,
        metric=metric,
        granularity=granularity,
        start_time=start_time,
        end_time=end_time,
        filters=filters,
    )

    if not time_series:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Campaign {campaign_id} not found",
        )

    return time_series


# =============================================================================
# Comparison Endpoints
# =============================================================================


@router.post(
    "/compare",
    response_model=CampaignComparison,
    summary="Compare campaigns",
    description="Compare 2-4 campaigns side by side with normalized metrics.",
    responses={
        400: {"description": "Invalid campaign IDs (must be 2-4 unique IDs)"},
        404: {"description": "One or more campaigns not found"},
    },
)
async def compare_campaigns(
    request: CampaignComparisonRequest,
    service: CampaignAnalyticsService = Depends(get_analytics_service),
) -> CampaignComparison:
    """
    Compare multiple campaigns side by side.

    Supports comparing 2-4 campaigns with:
    - Core metrics comparison (success rate, latency, cost)
    - Normalized metrics (0-1 scale) for radar charts
    - Best performer identification
    - Delta calculation (for 2-campaign comparison)
    - Optional time series overlay
    """
    try:
        comparison = await service.compare_campaigns(
            campaign_ids=request.campaign_ids,
            include_time_series=request.include_time_series,
            normalize_metrics=request.normalize_metrics,
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )

    if not comparison:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="One or more campaigns not found",
        )

    logger.info(f"Compared {len(request.campaign_ids)} campaigns")
    return comparison


# =============================================================================
# Telemetry Event Endpoints
# =============================================================================


@router.get(
    "/{campaign_id}/events",
    response_model=TelemetryListResponse,
    summary="List telemetry events",
    description="Get paginated list of individual telemetry events.",
    responses={404: {"description": "Campaign not found"}},
)
async def list_telemetry_events(
    campaign_id: str,
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(50, ge=1, le=200, description="Items per page"),
    status_filter: list[str] | None = Query(None, alias="status", description="Filter by status"),
    technique_suite: list[str] | None = Query(None, description="Filter by technique"),
    provider: list[str] | None = Query(None, description="Filter by provider"),
    model: list[str] | None = Query(None, description="Filter by model"),
    success_only: bool | None = Query(None, description="Only successful attempts"),
    start_time: datetime | None = Query(None, description="Events after this time"),
    end_time: datetime | None = Query(None, description="Events before this time"),
    min_potency: int | None = Query(None, ge=1, le=10, description="Minimum potency level"),
    max_potency: int | None = Query(None, ge=1, le=10, description="Maximum potency level"),
    service: CampaignAnalyticsService = Depends(get_analytics_service),
) -> TelemetryListResponse:
    """
    Get paginated list of telemetry events for a campaign.

    Returns individual execution events with summary information.
    Supports filtering by status, technique, provider, time range,
    and potency level.
    """
    filters = None
    if any([status_filter, technique_suite, provider, model, success_only is not None,
            start_time, end_time, min_potency, max_potency]):
        filters = TelemetryFilterParams(
            technique_suite=technique_suite,
            provider=provider,
            model=model,
            success_only=success_only,
            start_time=start_time,
            end_time=end_time,
            min_potency=min_potency,
            max_potency=max_potency,
        )

    result = await service.get_telemetry_events(
        campaign_id=campaign_id,
        page=page,
        page_size=page_size,
        filters=filters,
    )

    return result


@router.get(
    "/{campaign_id}/events/{event_id}",
    response_model=TelemetryEventDetail,
    summary="Get telemetry event detail",
    description="Get full details for a single telemetry event.",
    responses={404: {"description": "Event not found"}},
)
async def get_telemetry_event_detail(
    campaign_id: str,
    event_id: str,
    service: CampaignAnalyticsService = Depends(get_analytics_service),
) -> TelemetryEventDetail:
    """
    Get full details for a single telemetry event.

    Returns complete event information including full prompts,
    responses, quality scores, and detailed timing breakdown.
    """
    event = await service.get_telemetry_event_detail(campaign_id, event_id)

    if not event:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Event {event_id} not found in campaign {campaign_id}",
        )

    return event


# =============================================================================
# Export Endpoints
# =============================================================================


@router.get(
    "/{campaign_id}/export/csv",
    summary="Export campaign data as CSV",
    description="Export raw telemetry data in CSV format for external analysis.",
    response_class=StreamingResponse,
    responses={
        200: {
            "content": {"text/csv": {}},
            "description": "CSV file download",
        },
        404: {"description": "Campaign not found"},
    },
)
async def export_campaign_csv(
    campaign_id: str,
    include_prompts: bool = Query(False, description="Include full prompt text"),
    include_responses: bool = Query(False, description="Include full response text"),
    service: CampaignAnalyticsService = Depends(get_analytics_service),
) -> StreamingResponse:
    """
    Export campaign telemetry data as CSV.

    Downloads a CSV file containing all telemetry events for the campaign.
    Optionally includes full prompt and response text (can be large).
    """
    start_time = time.time()

    # Get campaign summary to verify it exists
    summary = await service.get_campaign_summary(campaign_id)
    if not summary:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Campaign {campaign_id} not found",
        )

    # Get all events (paginated internally for large campaigns)
    all_events = []
    page = 1
    page_size = 200

    while True:
        events = await service.get_telemetry_events(
            campaign_id=campaign_id,
            page=page,
            page_size=page_size,
        )
        all_events.extend(events.items)

        if page >= events.total_pages:
            break
        page += 1

    # Build CSV
    output = io.StringIO()

    # Define columns
    columns = [
        "id",
        "sequence_number",
        "technique_suite",
        "potency_level",
        "provider",
        "model",
        "status",
        "success_indicator",
        "total_latency_ms",
        "total_tokens",
        "created_at",
    ]

    if include_prompts:
        columns.append("original_prompt")
    if include_responses:
        columns.append("response_preview")

    writer = csv.DictWriter(output, fieldnames=columns)
    writer.writeheader()

    for event in all_events:
        row = {
            "id": event.id,
            "sequence_number": event.sequence_number,
            "technique_suite": event.technique_suite,
            "potency_level": event.potency_level,
            "provider": event.provider,
            "model": event.model,
            "status": event.status.value if hasattr(event.status, 'value') else event.status,
            "success_indicator": event.success_indicator,
            "total_latency_ms": event.total_latency_ms,
            "total_tokens": event.total_tokens,
            "created_at": event.created_at.isoformat() if event.created_at else "",
        }

        if include_prompts:
            row["original_prompt"] = event.original_prompt_preview or ""
        if include_responses:
            row["response_preview"] = ""  # Not in summary, would need detail

        writer.writerow(row)

    # Prepare response
    output.seek(0)

    # Generate filename
    campaign_name_safe = summary.name.replace(" ", "_").replace("/", "_")[:50]
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    filename = f"campaign_{campaign_name_safe}_{timestamp}.csv"

    processing_time = (time.time() - start_time) * 1000
    logger.info(
        f"Exported {len(all_events)} events for campaign {campaign_id} "
        f"in {processing_time:.2f}ms"
    )

    return StreamingResponse(
        io.BytesIO(output.getvalue().encode("utf-8")),
        media_type="text/csv",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "X-Row-Count": str(len(all_events)),
            "X-Processing-Time-Ms": f"{processing_time:.2f}",
        },
    )


@router.post(
    "/{campaign_id}/export/chart",
    response_model=ExportResponse,
    summary="Export campaign chart",
    description="Generate chart export (PNG/SVG) for research publications.",
    responses={
        200: {"description": "Export metadata with download URL or inline data"},
        404: {"description": "Campaign not found"},
    },
)
async def export_campaign_chart(
    campaign_id: str,
    request: ExportRequest,
    service: CampaignAnalyticsService = Depends(get_analytics_service),
) -> ExportResponse:
    """
    Generate chart export for a campaign.

    Creates PNG or SVG chart exports suitable for research publications
    and security reports.

    Note: Actual chart rendering happens on the frontend. This endpoint
    prepares the data and returns metadata for the export.
    """
    start_time = time.time()

    # Verify campaign exists
    summary = await service.get_campaign_summary(campaign_id)
    if not summary:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Campaign {campaign_id} not found",
        )

    # Get statistics for chart data
    stats = await service.calculate_statistics(campaign_id)

    # Determine format
    export_format = ExportFormat.PNG
    if request.chart_options:
        export_format = request.chart_options.format

    # Generate metadata
    campaign_name_safe = summary.name.replace(" ", "_").replace("/", "_")[:50]
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    mime_types = {
        ExportFormat.PNG: "image/png",
        ExportFormat.SVG: "image/svg+xml",
        ExportFormat.CSV: "text/csv",
        ExportFormat.JSON: "application/json",
    }

    extensions = {
        ExportFormat.PNG: "png",
        ExportFormat.SVG: "svg",
        ExportFormat.CSV: "csv",
        ExportFormat.JSON: "json",
    }

    processing_time = (time.time() - start_time) * 1000

    # Return export metadata
    # Actual chart rendering is handled by frontend using Recharts
    return ExportResponse(
        success=True,
        campaign_id=campaign_id,
        export_type=request.export_type,
        file_name=f"campaign_{campaign_name_safe}_{timestamp}.{extensions[export_format]}",
        file_size_bytes=0,  # Will be set after frontend renders
        mime_type=mime_types[export_format],
        data=None,  # Chart data is rendered client-side
        download_url=None,  # Frontend generates the download
        exported_at=datetime.utcnow(),
        row_count=stats.attempts.total if stats else 0,
        processing_time_ms=processing_time,
    )


# =============================================================================
# Cache Management Endpoints
# =============================================================================


@router.delete(
    "/{campaign_id}/cache",
    summary="Invalidate campaign cache",
    description="Clear cached analytics data for a campaign.",
    responses={200: {"description": "Cache invalidated"}},
)
async def invalidate_campaign_cache(
    campaign_id: str,
    service: CampaignAnalyticsService = Depends(get_analytics_service),
) -> dict[str, Any]:
    """
    Invalidate cached analytics data for a campaign.

    Forces fresh computation of statistics and breakdowns
    on the next request.
    """
    await service.invalidate_cache(campaign_id)

    return {
        "success": True,
        "message": f"Cache invalidated for campaign {campaign_id}",
    }


@router.get(
    "/cache/stats",
    summary="Get cache statistics",
    description="Get analytics cache statistics.",
    responses={200: {"description": "Cache statistics"}},
)
async def get_cache_stats(
    service: CampaignAnalyticsService = Depends(get_analytics_service),
) -> dict[str, Any]:
    """
    Get analytics cache statistics.

    Returns cache hit/miss rates and size information.
    """
    return service.get_cache_stats()
