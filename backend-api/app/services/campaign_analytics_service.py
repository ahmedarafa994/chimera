"""Campaign Analytics Service.

Provides comprehensive analytics for campaign telemetry including:
- Campaign summaries and statistics (mean, median, p95, std_dev)
- Time series data for visualization
- Campaign comparison with normalized metrics
- Technique and provider breakdowns
- Caching for expensive aggregations

Subtask 1.3: Create CampaignAnalyticsService
"""

import asyncio
import hashlib
import statistics
import time
from datetime import datetime
from typing import Any

from sqlalchemy import and_, case, desc, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import logger
from app.infrastructure.database.campaign_models import (
    Campaign,
    CampaignResult,
    CampaignTelemetryEvent,
    ExecutionStatus,
)
from app.schemas.campaign_analytics import (
    AttemptCounts,
    BreakdownItem,
    CampaignComparison,
    CampaignComparisonItem,
    CampaignDetail,
    CampaignFilterParams,
    CampaignListResponse,
    CampaignStatistics,
    CampaignStatusEnum,
    CampaignSummary,
    DistributionStats,
    PercentileStats,
    PotencyBreakdown,
    ProviderBreakdown,
    TechniqueBreakdown,
    TelemetryEventDetail,
    TelemetryEventSummary,
    TelemetryFilterParams,
    TelemetryListResponse,
    TelemetryTimeSeries,
    TimeGranularity,
    TimeSeriesDataPoint,
)

# =============================================================================
# Analytics Cache
# =============================================================================


class AnalyticsCache:
    """LRU cache for expensive analytics computations.

    Provides TTL-based caching for campaign statistics and aggregations
    to reduce database load for frequently accessed campaigns.
    """

    def __init__(self, max_size: int = 200, default_ttl: int = 300) -> None:
        """Initialize analytics cache.

        Args:
            max_size: Maximum number of cached entries
            default_ttl: Default time-to-live in seconds

        """
        self._cache: dict[str, dict[str, Any]] = {}
        self._max_size = max_size
        self._default_ttl = default_ttl
        self._hits = 0
        self._misses = 0
        self._lock = asyncio.Lock()

    def _generate_key(self, prefix: str, *args: Any) -> str:
        """Generate cache key from prefix and arguments."""
        key_data = f"{prefix}:{':'.join(str(a) for a in args)}"
        return hashlib.sha256(key_data.encode()).hexdigest()

    async def get(self, prefix: str, *args: Any) -> Any | None:
        """Get cached value for key."""
        key = self._generate_key(prefix, *args)

        async with self._lock:
            if key not in self._cache:
                self._misses += 1
                return None

            entry = self._cache[key]
            if time.time() > entry["expires_at"]:
                del self._cache[key]
                self._misses += 1
                return None

            self._hits += 1
            logger.debug(f"Analytics cache hit for key: {prefix}")
            return entry["value"]

    async def set(self, value: Any, prefix: str, *args: Any, ttl: int | None = None) -> None:
        """Cache value with key."""
        key = self._generate_key(prefix, *args)

        async with self._lock:
            # Evict oldest entries if cache is full
            if len(self._cache) >= self._max_size:
                oldest_key = min(self._cache.keys(), key=lambda k: self._cache[k]["created_at"])
                del self._cache[oldest_key]

            self._cache[key] = {
                "value": value,
                "created_at": time.time(),
                "expires_at": time.time() + (ttl or self._default_ttl),
            }

    async def invalidate(self, prefix: str, *args: Any) -> None:
        """Invalidate a specific cache entry."""
        key = self._generate_key(prefix, *args)
        async with self._lock:
            self._cache.pop(key, None)

    async def invalidate_campaign(self, campaign_id: str) -> None:
        """Invalidate all cache entries for a campaign."""
        async with self._lock:
            keys_to_delete = [
                k for k, v in self._cache.items() if campaign_id in str(v.get("value", ""))
            ]
            for key in keys_to_delete:
                del self._cache[key]

    async def clear(self) -> None:
        """Clear all cached entries."""
        async with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        total = self._hits + self._misses
        return {
            "size": len(self._cache),
            "max_size": self._max_size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self._hits / total if total > 0 else 0,
        }


# =============================================================================
# Campaign Analytics Service
# =============================================================================


class CampaignAnalyticsService:
    """Service for campaign telemetry analytics.

    Provides comprehensive analytics including:
    - Campaign summaries and detailed statistics
    - Time series data for charting
    - Side-by-side campaign comparison
    - Technique and provider breakdowns

    Includes caching for expensive aggregations.
    """

    # Cache configuration
    _CACHE_TTL_SUMMARY = 60  # 1 minute for summaries
    _CACHE_TTL_STATISTICS = 120  # 2 minutes for statistics
    _CACHE_TTL_TIME_SERIES = 180  # 3 minutes for time series
    _CACHE_TTL_BREAKDOWN = 120  # 2 minutes for breakdowns

    def __init__(self, session: AsyncSession) -> None:
        """Initialize campaign analytics service.

        Args:
            session: SQLAlchemy async session for database operations

        """
        self.session = session
        self._cache = AnalyticsCache(max_size=200, default_ttl=120)

    # =========================================================================
    # Campaign Summary Methods
    # =========================================================================

    async def get_campaign_summary(self, campaign_id: str) -> CampaignSummary | None:
        """Get summary view of a campaign.

        Args:
            campaign_id: Campaign identifier

        Returns:
            CampaignSummary or None if not found

        """
        # Check cache first
        cached = await self._cache.get("summary", campaign_id)
        if cached:
            return cached

        # Query campaign with basic info
        stmt = select(Campaign).where(Campaign.id == campaign_id)
        result = await self.session.execute(stmt)
        campaign = result.scalar_one_or_none()

        if not campaign:
            return None

        # Get quick stats from results or calculate
        summary = await self._build_campaign_summary(campaign)

        # Cache the result
        await self._cache.set(summary, "summary", campaign_id, ttl=self._CACHE_TTL_SUMMARY)

        return summary

    async def get_campaign_detail(self, campaign_id: str) -> CampaignDetail | None:
        """Get detailed campaign information.

        Args:
            campaign_id: Campaign identifier

        Returns:
            CampaignDetail or None if not found

        """
        stmt = select(Campaign).where(Campaign.id == campaign_id)
        result = await self.session.execute(stmt)
        campaign = result.scalar_one_or_none()

        if not campaign:
            return None

        # Build base summary
        summary = await self._build_campaign_summary(campaign)

        # Extend with detail fields
        return CampaignDetail(
            **summary.model_dump(),
            transformation_config=campaign.transformation_config or {},
            config=campaign.config or {},
            user_id=campaign.user_id,
            session_id=campaign.session_id,
        )

    async def list_campaigns(
        self,
        page: int = 1,
        page_size: int = 20,
        sort_by: str = "created_at",
        sort_order: str = "desc",
        filters: CampaignFilterParams | None = None,
    ) -> CampaignListResponse:
        """List campaigns with pagination and filtering.

        Args:
            page: Page number (1-indexed)
            page_size: Items per page
            sort_by: Field to sort by
            sort_order: 'asc' or 'desc'
            filters: Optional filter parameters

        Returns:
            Paginated campaign list response

        """
        # Base query
        stmt = select(Campaign)

        # Apply filters
        if filters:
            stmt = self._apply_campaign_filters(stmt, filters)

        # Get total count
        count_stmt = select(func.count()).select_from(stmt.subquery())
        total_result = await self.session.execute(count_stmt)
        total = total_result.scalar() or 0

        # Apply sorting
        sort_column = getattr(Campaign, sort_by, Campaign.created_at)
        if sort_order == "desc":
            stmt = stmt.order_by(desc(sort_column))
        else:
            stmt = stmt.order_by(sort_column)

        # Apply pagination
        offset = (page - 1) * page_size
        stmt = stmt.offset(offset).limit(page_size)

        result = await self.session.execute(stmt)
        campaigns = result.scalars().all()

        # Build summaries
        items = []
        for campaign in campaigns:
            summary = await self._build_campaign_summary(campaign)
            items.append(summary)

        total_pages = (total + page_size - 1) // page_size

        return CampaignListResponse(
            items=items,
            total=total,
            page=page,
            page_size=page_size,
            total_pages=total_pages,
            has_next=page < total_pages,
            has_prev=page > 1,
        )

    # =========================================================================
    # Statistics Methods
    # =========================================================================

    async def calculate_statistics(self, campaign_id: str) -> CampaignStatistics | None:
        """Calculate comprehensive statistics for a campaign.

        Computes mean, median, p95, std_dev for success rates, latency,
        token usage, and cost metrics.

        Args:
            campaign_id: Campaign identifier

        Returns:
            CampaignStatistics or None if campaign not found

        """
        # Check cache first
        cached = await self._cache.get("statistics", campaign_id)
        if cached:
            return cached

        # Verify campaign exists
        campaign_stmt = select(Campaign.id).where(Campaign.id == campaign_id)
        campaign_result = await self.session.execute(campaign_stmt)
        if not campaign_result.scalar_one_or_none():
            return None

        # Get all telemetry events for calculation
        events_stmt = select(CampaignTelemetryEvent).where(
            CampaignTelemetryEvent.campaign_id == campaign_id,
        )
        events_result = await self.session.execute(events_stmt)
        events = list(events_result.scalars().all())

        if not events:
            # Return empty statistics
            return CampaignStatistics(
                campaign_id=campaign_id,
                attempts=AttemptCounts(),
                success_rate=DistributionStats(),
                semantic_success=DistributionStats(),
                latency_ms=DistributionStats(),
                prompt_tokens=DistributionStats(),
                completion_tokens=DistributionStats(),
                total_tokens=DistributionStats(),
                cost_cents=DistributionStats(),
                computed_at=datetime.utcnow(),
            )

        # Calculate attempt counts
        attempts = self._calculate_attempt_counts(events)

        # Calculate success rate distribution (per-event success indicator)
        success_values = [1.0 if e.success_indicator else 0.0 for e in events]
        success_rate_stats = self._calculate_distribution_stats(success_values)

        # Calculate semantic success distribution
        semantic_values = [
            e.semantic_success_score for e in events if e.semantic_success_score is not None
        ]
        semantic_success_stats = self._calculate_distribution_stats(semantic_values)

        # Calculate latency distribution
        latency_values = [e.total_latency_ms for e in events if e.total_latency_ms > 0]
        latency_stats = self._calculate_distribution_stats(latency_values)

        # Calculate token distributions
        prompt_token_values = [float(e.prompt_tokens) for e in events if e.prompt_tokens > 0]
        completion_token_values = [
            float(e.completion_tokens) for e in events if e.completion_tokens > 0
        ]
        total_token_values = [float(e.total_tokens) for e in events if e.total_tokens > 0]

        prompt_token_stats = self._calculate_distribution_stats(prompt_token_values)
        completion_token_stats = self._calculate_distribution_stats(completion_token_values)
        total_token_stats = self._calculate_distribution_stats(total_token_values)

        # Calculate cost distribution
        cost_values = [e.estimated_cost_cents for e in events if e.estimated_cost_cents > 0]
        cost_stats = self._calculate_distribution_stats(cost_values)

        # Calculate totals
        total_prompt_tokens = sum(e.prompt_tokens for e in events)
        total_completion_tokens = sum(e.completion_tokens for e in events)
        total_tokens_used = sum(e.total_tokens for e in events)
        total_cost = sum(e.estimated_cost_cents for e in events)

        # Calculate duration if campaign has timestamps
        campaign = await self.session.get(Campaign, campaign_id)
        total_duration = None
        if campaign and campaign.started_at and campaign.completed_at:
            total_duration = (campaign.completed_at - campaign.started_at).total_seconds()

        stats = CampaignStatistics(
            campaign_id=campaign_id,
            attempts=attempts,
            success_rate=success_rate_stats,
            semantic_success=semantic_success_stats,
            latency_ms=latency_stats,
            prompt_tokens=prompt_token_stats,
            completion_tokens=completion_token_stats,
            total_tokens=total_token_stats,
            total_prompt_tokens=total_prompt_tokens,
            total_completion_tokens=total_completion_tokens,
            total_tokens_used=total_tokens_used,
            cost_cents=cost_stats,
            total_cost_cents=total_cost,
            total_duration_seconds=total_duration,
            computed_at=datetime.utcnow(),
        )

        # Cache the result
        await self._cache.set(stats, "statistics", campaign_id, ttl=self._CACHE_TTL_STATISTICS)

        return stats

    # =========================================================================
    # Time Series Methods
    # =========================================================================

    async def get_time_series(
        self,
        campaign_id: str,
        metric: str = "success_rate",
        granularity: TimeGranularity = TimeGranularity.HOUR,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        filters: TelemetryFilterParams | None = None,
    ) -> TelemetryTimeSeries | None:
        """Get time series data for a campaign metric.

        Args:
            campaign_id: Campaign identifier
            metric: Metric name ('success_rate', 'latency', 'tokens', 'cost')
            granularity: Time bucket granularity
            start_time: Start of time range
            end_time: End of time range
            filters: Optional telemetry filters

        Returns:
            TelemetryTimeSeries or None if campaign not found

        """
        # Cache key includes all parameters
        cache_key_parts = (campaign_id, metric, granularity.value, str(start_time), str(end_time))
        cached = await self._cache.get("time_series", *cache_key_parts)
        if cached:
            return cached

        # Verify campaign exists
        campaign_stmt = select(Campaign).where(Campaign.id == campaign_id)
        campaign_result = await self.session.execute(campaign_stmt)
        campaign = campaign_result.scalar_one_or_none()
        if not campaign:
            return None

        # Build query with time bucketing
        time_bucket = self._get_time_bucket_expression(granularity)

        # Base query
        stmt = (
            select(
                time_bucket.label("bucket"),
                func.count(CampaignTelemetryEvent.id).label("count"),
            )
            .where(CampaignTelemetryEvent.campaign_id == campaign_id)
            .group_by(time_bucket)
            .order_by(time_bucket)
        )

        # Add metric-specific aggregation
        if metric == "success_rate":
            stmt = stmt.add_columns(
                func.avg(case((CampaignTelemetryEvent.success_indicator, 1.0), else_=0.0)).label(
                    "value",
                ),
            )
        elif metric == "latency":
            stmt = stmt.add_columns(
                func.avg(CampaignTelemetryEvent.total_latency_ms).label("value"),
            )
        elif metric == "tokens":
            stmt = stmt.add_columns(func.avg(CampaignTelemetryEvent.total_tokens).label("value"))
        elif metric == "cost":
            stmt = stmt.add_columns(
                func.sum(CampaignTelemetryEvent.estimated_cost_cents).label("value"),
            )
        elif metric == "semantic_success":
            stmt = stmt.add_columns(
                func.avg(CampaignTelemetryEvent.semantic_success_score).label("value"),
            )
        else:
            # Default to success rate
            stmt = stmt.add_columns(
                func.avg(case((CampaignTelemetryEvent.success_indicator, 1.0), else_=0.0)).label(
                    "value",
                ),
            )

        # Apply time range filters
        if start_time:
            stmt = stmt.where(CampaignTelemetryEvent.created_at >= start_time)
        if end_time:
            stmt = stmt.where(CampaignTelemetryEvent.created_at <= end_time)

        # Apply additional filters
        if filters:
            stmt = self._apply_telemetry_filters(stmt, filters)

        result = await self.session.execute(stmt)
        rows = result.all()

        # Build data points
        data_points = []
        for row in rows:
            data_points.append(
                TimeSeriesDataPoint(
                    timestamp=row.bucket,
                    value=float(row.value) if row.value is not None else 0.0,
                    count=row.count,
                ),
            )

        # Determine time range
        series_start = start_time or (campaign.started_at if campaign else None)
        series_end = end_time or (campaign.completed_at if campaign else None)

        if data_points:
            series_start = series_start or data_points[0].timestamp
            series_end = series_end or data_points[-1].timestamp

        time_series = TelemetryTimeSeries(
            campaign_id=campaign_id,
            metric=metric,
            granularity=granularity,
            data_points=data_points,
            start_time=series_start,
            end_time=series_end,
            total_points=len(data_points),
        )

        # Cache the result
        await self._cache.set(
            time_series,
            "time_series",
            *cache_key_parts,
            ttl=self._CACHE_TTL_TIME_SERIES,
        )

        return time_series

    # =========================================================================
    # Comparison Methods
    # =========================================================================

    async def compare_campaigns(
        self,
        campaign_ids: list[str],
        include_time_series: bool = False,
        normalize_metrics: bool = True,
    ) -> CampaignComparison | None:
        """Compare multiple campaigns side-by-side.

        Supports up to 4 campaigns with normalized metrics for radar charts.

        Args:
            campaign_ids: List of 2-4 campaign IDs
            include_time_series: Include time series data
            normalize_metrics: Include normalized (0-1) metrics

        Returns:
            CampaignComparison or None if any campaign not found

        """
        if len(campaign_ids) < 2 or len(campaign_ids) > 4:
            msg = "Compare requires 2-4 campaign IDs"
            raise ValueError(msg)

        # Get all campaigns
        campaigns_data: list[CampaignComparisonItem] = []
        stats_list: list[CampaignStatistics] = []

        for cid in campaign_ids:
            summary = await self.get_campaign_summary(cid)
            if not summary:
                return None

            stats = await self.calculate_statistics(cid)
            if not stats:
                return None

            stats_list.append(stats)

            # Build comparison item
            item = CampaignComparisonItem(
                campaign_id=summary.id,
                campaign_name=summary.name,
                status=summary.status,
                total_attempts=summary.total_attempts,
                success_rate=summary.success_rate,
                semantic_success_mean=stats.semantic_success.mean,
                latency_mean=stats.latency_ms.mean,
                latency_p95=(
                    stats.latency_ms.percentiles.p95 if stats.latency_ms.percentiles else None
                ),
                avg_tokens=stats.total_tokens.mean,
                total_tokens=stats.total_tokens_used,
                total_cost_cents=stats.total_cost_cents,
                avg_cost_per_attempt=(
                    stats.total_cost_cents / stats.attempts.total
                    if stats.attempts.total > 0
                    else None
                ),
                duration_seconds=stats.total_duration_seconds,
                best_technique=None,  # Will be populated from breakdown
                best_provider=None,  # Will be populated from breakdown
            )
            campaigns_data.append(item)

        # Normalize metrics if requested
        if normalize_metrics:
            campaigns_data = self._normalize_comparison_metrics(campaigns_data)

        # Identify best performers
        best_success_rate = max(campaigns_data, key=lambda c: c.success_rate or 0)
        best_latency = min(
            campaigns_data,
            key=lambda c: c.latency_mean if c.latency_mean is not None else float("inf"),
        )
        best_cost = min(
            campaigns_data,
            key=lambda c: (
                c.avg_cost_per_attempt if c.avg_cost_per_attempt is not None else float("inf")
            ),
        )

        # Calculate deltas for 2-campaign comparison
        delta_success = None
        delta_latency = None
        delta_cost = None

        if len(campaigns_data) == 2:
            c1, c2 = campaigns_data[0], campaigns_data[1]
            if c1.success_rate is not None and c2.success_rate is not None:
                delta_success = c1.success_rate - c2.success_rate
            if c1.latency_mean is not None and c2.latency_mean is not None:
                delta_latency = c1.latency_mean - c2.latency_mean
            if c1.total_cost_cents is not None and c2.total_cost_cents is not None:
                delta_cost = c1.total_cost_cents - c2.total_cost_cents

        # Include time series if requested
        time_series_data = None
        if include_time_series:
            time_series_data = []
            for cid in campaign_ids:
                ts = await self.get_time_series(cid, metric="success_rate")
                if ts:
                    time_series_data.append(ts)

        return CampaignComparison(
            campaigns=campaigns_data,
            compared_at=datetime.utcnow(),
            best_success_rate_campaign=best_success_rate.campaign_id,
            best_latency_campaign=best_latency.campaign_id,
            best_cost_efficiency_campaign=best_cost.campaign_id,
            delta_success_rate=delta_success,
            delta_latency_ms=delta_latency,
            delta_cost_cents=delta_cost,
            time_series=time_series_data,
        )

    # =========================================================================
    # Breakdown Methods
    # =========================================================================

    async def get_technique_breakdown(self, campaign_id: str) -> TechniqueBreakdown | None:
        """Get success rate breakdown by transformation technique.

        Args:
            campaign_id: Campaign identifier

        Returns:
            TechniqueBreakdown or None if campaign not found

        """
        # Check cache
        cached = await self._cache.get("technique_breakdown", campaign_id)
        if cached:
            return cached

        # Verify campaign exists
        campaign_stmt = select(Campaign.id).where(Campaign.id == campaign_id)
        campaign_result = await self.session.execute(campaign_stmt)
        if not campaign_result.scalar_one_or_none():
            return None

        # Aggregate by technique
        stmt = (
            select(
                CampaignTelemetryEvent.technique_suite,
                func.count(CampaignTelemetryEvent.id).label("attempts"),
                func.sum(case((CampaignTelemetryEvent.success_indicator, 1), else_=0)).label(
                    "successes",
                ),
                func.avg(CampaignTelemetryEvent.total_latency_ms).label("avg_latency"),
                func.avg(CampaignTelemetryEvent.total_tokens).label("avg_tokens"),
                func.sum(CampaignTelemetryEvent.estimated_cost_cents).label("total_cost"),
            )
            .where(CampaignTelemetryEvent.campaign_id == campaign_id)
            .group_by(CampaignTelemetryEvent.technique_suite)
            .order_by(desc("successes"))
        )

        result = await self.session.execute(stmt)
        rows = result.all()

        items = []
        best_technique = None
        worst_technique = None
        best_rate = -1
        worst_rate = 2

        for row in rows:
            success_rate = row.successes / row.attempts if row.attempts > 0 else 0.0

            item = BreakdownItem(
                name=row.technique_suite,
                attempts=row.attempts,
                successes=row.successes,
                success_rate=success_rate,
                avg_latency_ms=row.avg_latency,
                avg_tokens=row.avg_tokens,
                total_cost_cents=row.total_cost,
            )
            items.append(item)

            if success_rate > best_rate:
                best_rate = success_rate
                best_technique = row.technique_suite
            if success_rate < worst_rate:
                worst_rate = success_rate
                worst_technique = row.technique_suite

        breakdown = TechniqueBreakdown(
            campaign_id=campaign_id,
            items=items,
            best_technique=best_technique,
            worst_technique=worst_technique,
        )

        # Cache the result
        await self._cache.set(
            breakdown,
            "technique_breakdown",
            campaign_id,
            ttl=self._CACHE_TTL_BREAKDOWN,
        )

        return breakdown

    async def get_provider_breakdown(self, campaign_id: str) -> ProviderBreakdown | None:
        """Get success rate breakdown by LLM provider.

        Args:
            campaign_id: Campaign identifier

        Returns:
            ProviderBreakdown or None if campaign not found

        """
        # Check cache
        cached = await self._cache.get("provider_breakdown", campaign_id)
        if cached:
            return cached

        # Verify campaign exists
        campaign_stmt = select(Campaign.id).where(Campaign.id == campaign_id)
        campaign_result = await self.session.execute(campaign_stmt)
        if not campaign_result.scalar_one_or_none():
            return None

        # Aggregate by provider
        stmt = (
            select(
                CampaignTelemetryEvent.provider,
                func.count(CampaignTelemetryEvent.id).label("attempts"),
                func.sum(case((CampaignTelemetryEvent.success_indicator, 1), else_=0)).label(
                    "successes",
                ),
                func.avg(CampaignTelemetryEvent.total_latency_ms).label("avg_latency"),
                func.avg(CampaignTelemetryEvent.total_tokens).label("avg_tokens"),
                func.sum(CampaignTelemetryEvent.estimated_cost_cents).label("total_cost"),
            )
            .where(CampaignTelemetryEvent.campaign_id == campaign_id)
            .group_by(CampaignTelemetryEvent.provider)
            .order_by(desc("successes"))
        )

        result = await self.session.execute(stmt)
        rows = result.all()

        items = []
        best_provider = None
        best_rate = -1

        for row in rows:
            success_rate = row.successes / row.attempts if row.attempts > 0 else 0.0

            item = BreakdownItem(
                name=row.provider,
                attempts=row.attempts,
                successes=row.successes,
                success_rate=success_rate,
                avg_latency_ms=row.avg_latency,
                avg_tokens=row.avg_tokens,
                total_cost_cents=row.total_cost,
            )
            items.append(item)

            if success_rate > best_rate:
                best_rate = success_rate
                best_provider = row.provider

        breakdown = ProviderBreakdown(
            campaign_id=campaign_id,
            items=items,
            best_provider=best_provider,
        )

        # Cache the result
        await self._cache.set(
            breakdown,
            "provider_breakdown",
            campaign_id,
            ttl=self._CACHE_TTL_BREAKDOWN,
        )

        return breakdown

    async def get_potency_breakdown(self, campaign_id: str) -> PotencyBreakdown | None:
        """Get success rate breakdown by potency level.

        Args:
            campaign_id: Campaign identifier

        Returns:
            PotencyBreakdown or None if campaign not found

        """
        # Check cache
        cached = await self._cache.get("potency_breakdown", campaign_id)
        if cached:
            return cached

        # Verify campaign exists
        campaign_stmt = select(Campaign.id).where(Campaign.id == campaign_id)
        campaign_result = await self.session.execute(campaign_stmt)
        if not campaign_result.scalar_one_or_none():
            return None

        # Aggregate by potency level
        stmt = (
            select(
                CampaignTelemetryEvent.potency_level,
                func.count(CampaignTelemetryEvent.id).label("attempts"),
                func.sum(case((CampaignTelemetryEvent.success_indicator, 1), else_=0)).label(
                    "successes",
                ),
                func.avg(CampaignTelemetryEvent.total_latency_ms).label("avg_latency"),
                func.avg(CampaignTelemetryEvent.total_tokens).label("avg_tokens"),
                func.sum(CampaignTelemetryEvent.estimated_cost_cents).label("total_cost"),
            )
            .where(CampaignTelemetryEvent.campaign_id == campaign_id)
            .group_by(CampaignTelemetryEvent.potency_level)
            .order_by(CampaignTelemetryEvent.potency_level)
        )

        result = await self.session.execute(stmt)
        rows = result.all()

        items = []
        best_potency = None
        best_rate = -1

        for row in rows:
            success_rate = row.successes / row.attempts if row.attempts > 0 else 0.0

            item = BreakdownItem(
                name=str(row.potency_level),
                attempts=row.attempts,
                successes=row.successes,
                success_rate=success_rate,
                avg_latency_ms=row.avg_latency,
                avg_tokens=row.avg_tokens,
                total_cost_cents=row.total_cost,
            )
            items.append(item)

            if success_rate > best_rate:
                best_rate = success_rate
                best_potency = row.potency_level

        breakdown = PotencyBreakdown(
            campaign_id=campaign_id,
            items=items,
            best_potency_level=best_potency,
        )

        # Cache the result
        await self._cache.set(
            breakdown,
            "potency_breakdown",
            campaign_id,
            ttl=self._CACHE_TTL_BREAKDOWN,
        )

        return breakdown

    # =========================================================================
    # Telemetry Event Methods
    # =========================================================================

    async def get_telemetry_events(
        self,
        campaign_id: str,
        page: int = 1,
        page_size: int = 50,
        filters: TelemetryFilterParams | None = None,
    ) -> TelemetryListResponse:
        """Get paginated telemetry events for a campaign.

        Args:
            campaign_id: Campaign identifier
            page: Page number
            page_size: Items per page
            filters: Optional filter parameters

        Returns:
            Paginated telemetry event list

        """
        # Base query
        stmt = select(CampaignTelemetryEvent).where(
            CampaignTelemetryEvent.campaign_id == campaign_id,
        )

        # Apply filters
        if filters:
            stmt = self._apply_telemetry_filters(stmt, filters)

        # Get total count
        count_stmt = select(func.count()).select_from(stmt.subquery())
        total_result = await self.session.execute(count_stmt)
        total = total_result.scalar() or 0

        # Apply sorting and pagination
        stmt = stmt.order_by(desc(CampaignTelemetryEvent.created_at))
        offset = (page - 1) * page_size
        stmt = stmt.offset(offset).limit(page_size)

        result = await self.session.execute(stmt)
        events = result.scalars().all()

        # Build summaries
        items = []
        for event in events:
            summary = TelemetryEventSummary(
                id=event.id,
                campaign_id=event.campaign_id,
                sequence_number=event.sequence_number,
                original_prompt_preview=(
                    event.original_prompt[:200] if event.original_prompt else None
                ),
                technique_suite=event.technique_suite,
                potency_level=event.potency_level,
                provider=event.provider,
                model=event.model,
                status=event.status,
                success_indicator=event.success_indicator,
                total_latency_ms=event.total_latency_ms,
                total_tokens=event.total_tokens,
                created_at=event.created_at,
            )
            items.append(summary)

        total_pages = (total + page_size - 1) // page_size

        return TelemetryListResponse(
            items=items,
            total=total,
            page=page,
            page_size=page_size,
            total_pages=total_pages,
        )

    async def get_telemetry_event_detail(
        self,
        campaign_id: str,
        event_id: str,
    ) -> TelemetryEventDetail | None:
        """Get full details for a single telemetry event.

        Args:
            campaign_id: Campaign identifier
            event_id: Telemetry event identifier

        Returns:
            TelemetryEventDetail or None if not found

        """
        stmt = select(CampaignTelemetryEvent).where(
            and_(
                CampaignTelemetryEvent.campaign_id == campaign_id,
                CampaignTelemetryEvent.id == event_id,
            ),
        )

        result = await self.session.execute(stmt)
        event = result.scalar_one_or_none()

        if not event:
            return None

        return TelemetryEventDetail(
            id=event.id,
            campaign_id=event.campaign_id,
            sequence_number=event.sequence_number,
            original_prompt_preview=(
                event.original_prompt[:200] if event.original_prompt else None
            ),
            technique_suite=event.technique_suite,
            potency_level=event.potency_level,
            provider=event.provider,
            model=event.model,
            status=event.status,
            success_indicator=event.success_indicator,
            total_latency_ms=event.total_latency_ms,
            total_tokens=event.total_tokens,
            created_at=event.created_at,
            original_prompt=event.original_prompt,
            transformed_prompt=event.transformed_prompt,
            response_text=event.response_text,
            applied_techniques=event.applied_techniques or [],
            execution_time_ms=event.execution_time_ms,
            transformation_time_ms=event.transformation_time_ms,
            prompt_tokens=event.prompt_tokens,
            completion_tokens=event.completion_tokens,
            semantic_success_score=event.semantic_success_score,
            effectiveness_score=event.effectiveness_score,
            naturalness_score=event.naturalness_score,
            detectability_score=event.detectability_score,
            bypass_indicators=event.bypass_indicators or [],
            safety_trigger_detected=event.safety_trigger_detected,
            error_message=event.error_message,
            error_code=event.error_code,
            metadata=event.event_metadata or {},
        )

    # =========================================================================
    # Cache Management
    # =========================================================================

    async def invalidate_cache(self, campaign_id: str | None = None) -> None:
        """Invalidate cached analytics data.

        Args:
            campaign_id: Specific campaign to invalidate, or None for all

        """
        if campaign_id:
            await self._cache.invalidate_campaign(campaign_id)
            logger.info(f"Invalidated analytics cache for campaign: {campaign_id}")
        else:
            await self._cache.clear()
            logger.info("Cleared all analytics cache")

    def get_cache_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        return self._cache.get_stats()

    # =========================================================================
    # Private Helper Methods
    # =========================================================================

    async def _build_campaign_summary(self, campaign: Campaign) -> CampaignSummary:
        """Build campaign summary from campaign entity."""
        # Try to get pre-computed results first
        results_stmt = select(CampaignResult).where(CampaignResult.campaign_id == campaign.id)
        results_result = await self.session.execute(results_stmt)
        results = results_result.scalar_one_or_none()

        total_attempts = 0
        success_rate = None
        avg_latency = None

        if results:
            total_attempts = results.total_attempts
            success_rate = results.success_rate
            avg_latency = results.latency_mean
        else:
            # Calculate from events
            count_stmt = select(
                func.count(CampaignTelemetryEvent.id).label("total"),
                func.avg(case((CampaignTelemetryEvent.success_indicator, 1.0), else_=0.0)).label(
                    "success_rate",
                ),
                func.avg(CampaignTelemetryEvent.total_latency_ms).label("avg_latency"),
            ).where(CampaignTelemetryEvent.campaign_id == campaign.id)
            count_result = await self.session.execute(count_stmt)
            counts = count_result.one()
            total_attempts = counts.total or 0
            success_rate = float(counts.success_rate) if counts.success_rate is not None else None
            avg_latency = float(counts.avg_latency) if counts.avg_latency is not None else None

        # Calculate duration
        duration_seconds = None
        if campaign.started_at and campaign.completed_at:
            duration_seconds = (campaign.completed_at - campaign.started_at).total_seconds()

        return CampaignSummary(
            id=campaign.id,
            name=campaign.name,
            description=campaign.description,
            objective=campaign.objective,
            status=CampaignStatusEnum(campaign.status),
            target_provider=campaign.target_provider,
            target_model=campaign.target_model,
            technique_suites=campaign.technique_suites or [],
            tags=campaign.tags or [],
            total_attempts=total_attempts,
            success_rate=success_rate,
            avg_latency_ms=avg_latency,
            started_at=campaign.started_at,
            completed_at=campaign.completed_at,
            duration_seconds=duration_seconds,
            created_at=campaign.created_at,
            updated_at=campaign.updated_at,
        )

    def _calculate_attempt_counts(self, events: list[CampaignTelemetryEvent]) -> AttemptCounts:
        """Calculate attempt counts by status."""
        counts = {
            "total": len(events),
            "successful": 0,
            "failed": 0,
            "partial_success": 0,
            "timeout": 0,
            "skipped": 0,
        }

        for event in events:
            status = event.status
            if status == ExecutionStatus.SUCCESS.value:
                counts["successful"] += 1
            elif status == ExecutionStatus.FAILURE.value:
                counts["failed"] += 1
            elif status == ExecutionStatus.PARTIAL_SUCCESS.value:
                counts["partial_success"] += 1
            elif status == ExecutionStatus.TIMEOUT.value:
                counts["timeout"] += 1
            elif status == ExecutionStatus.SKIPPED.value:
                counts["skipped"] += 1

        return AttemptCounts(**counts)

    def _calculate_distribution_stats(self, values: list[float]) -> DistributionStats:
        """Calculate distribution statistics for a list of values."""
        if not values:
            return DistributionStats()

        sorted_values = sorted(values)
        n = len(sorted_values)

        mean = statistics.mean(values)
        median = statistics.median(values)
        std_dev = statistics.stdev(values) if n > 1 else 0.0
        min_value = min(values)
        max_value = max(values)

        # Calculate percentiles
        p50 = sorted_values[int(n * 0.50)] if n > 0 else None
        p90 = sorted_values[int(n * 0.90)] if n > 0 else None
        p95 = sorted_values[int(n * 0.95)] if n > 0 else None
        p99 = sorted_values[int(n * 0.99)] if n > 0 else None

        return DistributionStats(
            mean=mean,
            median=median,
            std_dev=std_dev,
            min_value=min_value,
            max_value=max_value,
            percentiles=PercentileStats(
                p50=p50,
                p90=p90,
                p95=p95,
                p99=p99,
            ),
        )

    def _normalize_comparison_metrics(
        self,
        campaigns: list[CampaignComparisonItem],
    ) -> list[CampaignComparisonItem]:
        """Normalize metrics to 0-1 scale for radar chart comparison."""
        if not campaigns:
            return campaigns

        # Find min/max for each metric
        success_rates = [c.success_rate for c in campaigns if c.success_rate is not None]
        latencies = [c.latency_mean for c in campaigns if c.latency_mean is not None]
        costs = [c.avg_cost_per_attempt for c in campaigns if c.avg_cost_per_attempt is not None]

        max_success = max(success_rates) if success_rates else 1.0
        min_latency = min(latencies) if latencies else 1.0
        max_latency = max(latencies) if latencies else 1.0
        min_cost = min(costs) if costs else 1.0
        max_cost = max(costs) if costs else 1.0

        for campaign in campaigns:
            # Normalize success rate (higher is better)
            if campaign.success_rate is not None and max_success > 0:
                campaign.normalized_success_rate = campaign.success_rate / max_success

            # Normalize latency (lower is better, so invert)
            if campaign.latency_mean is not None and max_latency > min_latency:
                campaign.normalized_latency = 1.0 - (
                    (campaign.latency_mean - min_latency) / (max_latency - min_latency)
                )
            elif campaign.latency_mean is not None:
                campaign.normalized_latency = 1.0

            # Normalize cost (lower is better, so invert)
            if campaign.avg_cost_per_attempt is not None and max_cost > min_cost:
                campaign.normalized_cost = 1.0 - (
                    (campaign.avg_cost_per_attempt - min_cost) / (max_cost - min_cost)
                )
            elif campaign.avg_cost_per_attempt is not None:
                campaign.normalized_cost = 1.0

            # Overall effectiveness (average of normalized metrics)
            normalized_values = [
                v
                for v in [
                    campaign.normalized_success_rate,
                    campaign.normalized_latency,
                    campaign.normalized_cost,
                ]
                if v is not None
            ]
            if normalized_values:
                campaign.normalized_effectiveness = statistics.mean(normalized_values)

        return campaigns

    def _get_time_bucket_expression(self, granularity: TimeGranularity):
        """Get SQL expression for time bucketing based on granularity."""
        # SQLite-compatible time bucketing
        if granularity == TimeGranularity.MINUTE:
            return func.strftime("%Y-%m-%d %H:%M:00", CampaignTelemetryEvent.created_at)
        if granularity == TimeGranularity.HOUR:
            return func.strftime("%Y-%m-%d %H:00:00", CampaignTelemetryEvent.created_at)
        # DAY
        return func.strftime("%Y-%m-%d 00:00:00", CampaignTelemetryEvent.created_at)

    def _apply_campaign_filters(self, stmt, filters: CampaignFilterParams):
        """Apply filters to campaign query."""
        if filters.status:
            status_values = [s.value for s in filters.status]
            stmt = stmt.where(Campaign.status.in_(status_values))

        if filters.provider:
            stmt = stmt.where(Campaign.target_provider.in_(filters.provider))

        if filters.technique_suite:
            # JSON array contains check (SQLite compatible)
            for technique in filters.technique_suite:
                stmt = stmt.where(
                    func.json_extract(Campaign.technique_suites, "$").contains(technique),
                )

        if filters.tags:
            for tag in filters.tags:
                stmt = stmt.where(func.json_extract(Campaign.tags, "$").contains(tag))

        if filters.start_date:
            stmt = stmt.where(Campaign.created_at >= filters.start_date)

        if filters.end_date:
            stmt = stmt.where(Campaign.created_at <= filters.end_date)

        if filters.search:
            search_pattern = f"%{filters.search}%"
            stmt = stmt.where(
                (Campaign.name.ilike(search_pattern))
                | (Campaign.description.ilike(search_pattern)),
            )

        return stmt

    def _apply_telemetry_filters(self, stmt, filters: TelemetryFilterParams):
        """Apply filters to telemetry event query."""
        if filters.status:
            status_values = [s.value for s in filters.status]
            stmt = stmt.where(CampaignTelemetryEvent.status.in_(status_values))

        if filters.technique_suite:
            stmt = stmt.where(CampaignTelemetryEvent.technique_suite.in_(filters.technique_suite))

        if filters.provider:
            stmt = stmt.where(CampaignTelemetryEvent.provider.in_(filters.provider))

        if filters.model:
            stmt = stmt.where(CampaignTelemetryEvent.model.in_(filters.model))

        if filters.success_only is not None:
            stmt = stmt.where(CampaignTelemetryEvent.success_indicator == filters.success_only)

        if filters.start_time:
            stmt = stmt.where(CampaignTelemetryEvent.created_at >= filters.start_time)

        if filters.end_time:
            stmt = stmt.where(CampaignTelemetryEvent.created_at <= filters.end_time)

        if filters.min_potency is not None:
            stmt = stmt.where(CampaignTelemetryEvent.potency_level >= filters.min_potency)

        if filters.max_potency is not None:
            stmt = stmt.where(CampaignTelemetryEvent.potency_level <= filters.max_potency)

        return stmt


# =============================================================================
# Service Factory and Dependencies
# =============================================================================


async def get_campaign_analytics_service(session: AsyncSession) -> CampaignAnalyticsService:
    """Get campaign analytics service instance.

    Args:
        session: SQLAlchemy async session

    Returns:
        CampaignAnalyticsService instance

    """
    return CampaignAnalyticsService(session)


# Global service instance for non-request scoped usage
_analytics_service: CampaignAnalyticsService | None = None


async def get_analytics_service_singleton(session: AsyncSession) -> CampaignAnalyticsService:
    """Get singleton analytics service instance.

    Useful for background tasks and non-request contexts.

    Args:
        session: SQLAlchemy async session

    Returns:
        CampaignAnalyticsService singleton

    """
    global _analytics_service
    if _analytics_service is None:
        _analytics_service = CampaignAnalyticsService(session)
    return _analytics_service
