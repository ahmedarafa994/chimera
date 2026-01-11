"""
Backend API Integration Tests for Campaign Analytics Endpoints.

Tests cover:
- Campaign listing with pagination and filtering
- Campaign detail retrieval
- Campaign statistics endpoints
- Campaign comparison endpoint (2-4 campaigns)
- Time series data endpoints
- Breakdown endpoints (technique, provider, potency)
- Telemetry event endpoints
- Export endpoints (CSV, chart)
- Cache management endpoints
- Error responses and edge cases

Subtask 9.2: Create Backend API Integration Tests
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient

from app.main import app
from app.schemas.campaign_analytics import (
    CampaignComparison,
    CampaignComparisonRequest,
    CampaignDetail,
    CampaignListResponse,
    CampaignStatistics,
    CampaignStatusEnum,
    CampaignSummary,
    ExportFormat,
    ExportRequest,
    PotencyBreakdown,
    ProviderBreakdown,
    TechniqueBreakdown,
    TelemetryEventDetail,
    TelemetryListResponse,
    TelemetryTimeSeries,
    TimeGranularity,
)
from app.services.campaign_analytics_service import CampaignAnalyticsService


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def client():
    """Create a test client for the FastAPI application."""
    return TestClient(app)


@pytest.fixture
def auth_headers():
    """Return authentication headers for testing."""
    return {"X-API-Key": "chimera_default_key_change_in_production"}


@pytest.fixture
def mock_campaign_summary():
    """Create a mock campaign summary."""
    return CampaignSummary(
        id="test-campaign-001",
        name="Test Campaign",
        description="A test campaign for API testing",
        objective="Test jailbreak techniques",
        status=CampaignStatusEnum.COMPLETED,
        target_provider="openai",
        target_model="gpt-4",
        technique_suites=["dan_persona", "cognitive_hacking"],
        tags=["test", "benchmark"],
        total_attempts=100,
        success_rate=0.75,
        avg_latency_ms=1250.5,
        started_at=datetime(2024, 1, 1, 10, 0, 0),
        completed_at=datetime(2024, 1, 1, 12, 0, 0),
        duration_seconds=7200.0,
        created_at=datetime(2024, 1, 1, 9, 0, 0),
        updated_at=datetime(2024, 1, 1, 12, 0, 0),
    )


@pytest.fixture
def mock_campaign_detail(mock_campaign_summary):
    """Create a mock campaign detail."""
    return CampaignDetail(
        id=mock_campaign_summary.id,
        name=mock_campaign_summary.name,
        description=mock_campaign_summary.description,
        objective=mock_campaign_summary.objective,
        status=mock_campaign_summary.status,
        target_provider=mock_campaign_summary.target_provider,
        target_model=mock_campaign_summary.target_model,
        technique_suites=mock_campaign_summary.technique_suites,
        tags=mock_campaign_summary.tags,
        total_attempts=mock_campaign_summary.total_attempts,
        success_rate=mock_campaign_summary.success_rate,
        avg_latency_ms=mock_campaign_summary.avg_latency_ms,
        started_at=mock_campaign_summary.started_at,
        completed_at=mock_campaign_summary.completed_at,
        duration_seconds=mock_campaign_summary.duration_seconds,
        transformation_config={"potency": 5},
        config={"max_iterations": 100},
        user_id="user-001",
        session_id="session-001",
        created_at=mock_campaign_summary.created_at,
        updated_at=mock_campaign_summary.updated_at,
    )


@pytest.fixture
def mock_campaign_list_response(mock_campaign_summary):
    """Create a mock paginated campaign list response."""
    return CampaignListResponse(
        items=[mock_campaign_summary],
        total=1,
        page=1,
        page_size=20,
        total_pages=1,
        has_next=False,
        has_previous=False,
    )


@pytest.fixture
def mock_campaign_statistics():
    """Create mock campaign statistics."""
    from app.schemas.campaign_analytics import (
        AttemptCounts,
        DistributionStats,
        PercentileStats,
    )

    return CampaignStatistics(
        campaign_id="test-campaign-001",
        attempts=AttemptCounts(total=100, successful=75, failed=20, partial=5),
        success_rate=DistributionStats(
            mean=0.75,
            median=0.78,
            std_dev=0.12,
            min_value=0.0,
            max_value=1.0,
            percentiles=PercentileStats(p50=0.78, p90=0.92, p95=0.95, p99=0.98),
        ),
        semantic_success_rate=DistributionStats(
            mean=0.72,
            median=0.75,
            std_dev=0.15,
            min_value=0.0,
            max_value=1.0,
            percentiles=None,
        ),
        latency=DistributionStats(
            mean=1250.5,
            median=1100.0,
            std_dev=450.0,
            min_value=500.0,
            max_value=3000.0,
            percentiles=PercentileStats(p50=1100.0, p90=2000.0, p95=2500.0, p99=2900.0),
        ),
        prompt_tokens=DistributionStats(
            mean=150.0,
            median=140.0,
            std_dev=30.0,
            min_value=80.0,
            max_value=250.0,
            percentiles=None,
        ),
        completion_tokens=DistributionStats(
            mean=75.0,
            median=70.0,
            std_dev=20.0,
            min_value=30.0,
            max_value=150.0,
            percentiles=None,
        ),
        total_tokens=DistributionStats(
            mean=225.0,
            median=210.0,
            std_dev=45.0,
            min_value=110.0,
            max_value=400.0,
            percentiles=None,
        ),
        cost=DistributionStats(
            mean=1.5,
            median=1.4,
            std_dev=0.5,
            min_value=0.5,
            max_value=3.0,
            percentiles=None,
        ),
        total_duration_seconds=7200.0,
        computed_at=datetime(2024, 1, 1, 12, 0, 0),
    )


@pytest.fixture
def mock_technique_breakdown():
    """Create mock technique breakdown."""
    from app.schemas.campaign_analytics import BreakdownItem

    return TechniqueBreakdown(
        campaign_id="test-campaign-001",
        items=[
            BreakdownItem(
                name="dan_persona",
                count=50,
                success_count=40,
                success_rate=0.8,
                avg_latency_ms=1200.0,
                avg_tokens=200.0,
                avg_cost_cents=1.2,
                metadata={},
            ),
            BreakdownItem(
                name="cognitive_hacking",
                count=50,
                success_count=35,
                success_rate=0.7,
                avg_latency_ms=1300.0,
                avg_tokens=220.0,
                avg_cost_cents=1.5,
                metadata={},
            ),
        ],
        best_technique="dan_persona",
        computed_at=datetime(2024, 1, 1, 12, 0, 0),
    )


@pytest.fixture
def mock_provider_breakdown():
    """Create mock provider breakdown."""
    from app.schemas.campaign_analytics import BreakdownItem

    return ProviderBreakdown(
        campaign_id="test-campaign-001",
        items=[
            BreakdownItem(
                name="openai",
                count=70,
                success_count=55,
                success_rate=0.785,
                avg_latency_ms=1100.0,
                avg_tokens=190.0,
                avg_cost_cents=1.3,
                metadata={},
            ),
            BreakdownItem(
                name="anthropic",
                count=30,
                success_count=20,
                success_rate=0.667,
                avg_latency_ms=1500.0,
                avg_tokens=240.0,
                avg_cost_cents=1.8,
                metadata={},
            ),
        ],
        best_provider="openai",
        computed_at=datetime(2024, 1, 1, 12, 0, 0),
    )


@pytest.fixture
def mock_potency_breakdown():
    """Create mock potency breakdown."""
    from app.schemas.campaign_analytics import BreakdownItem

    return PotencyBreakdown(
        campaign_id="test-campaign-001",
        items=[
            BreakdownItem(
                name="5",
                count=40,
                success_count=30,
                success_rate=0.75,
                avg_latency_ms=1150.0,
                avg_tokens=180.0,
                avg_cost_cents=1.1,
                metadata={},
            ),
            BreakdownItem(
                name="7",
                count=60,
                success_count=45,
                success_rate=0.75,
                avg_latency_ms=1350.0,
                avg_tokens=230.0,
                avg_cost_cents=1.6,
                metadata={},
            ),
        ],
        best_potency=7,
        computed_at=datetime(2024, 1, 1, 12, 0, 0),
    )


@pytest.fixture
def mock_time_series():
    """Create mock time series data."""
    from app.schemas.campaign_analytics import TimeSeriesDataPoint

    return TelemetryTimeSeries(
        campaign_id="test-campaign-001",
        metric="success_rate",
        granularity=TimeGranularity.HOUR,
        start_time=datetime(2024, 1, 1, 10, 0, 0),
        end_time=datetime(2024, 1, 1, 12, 0, 0),
        data_points=[
            TimeSeriesDataPoint(
                timestamp=datetime(2024, 1, 1, 10, 0, 0),
                value=0.7,
                count=40,
                metadata={},
            ),
            TimeSeriesDataPoint(
                timestamp=datetime(2024, 1, 1, 11, 0, 0),
                value=0.8,
                count=60,
                metadata={},
            ),
        ],
        computed_at=datetime(2024, 1, 1, 12, 0, 0),
    )


@pytest.fixture
def mock_telemetry_list():
    """Create mock telemetry events list."""
    from app.schemas.campaign_analytics import TelemetryEventSummary, ExecutionStatusEnum

    return TelemetryListResponse(
        items=[
            TelemetryEventSummary(
                id="event-001",
                campaign_id="test-campaign-001",
                sequence_number=0,
                technique_suite="dan_persona",
                potency_level=5,
                provider="openai",
                model="gpt-4",
                status=ExecutionStatusEnum.SUCCESS,
                success_indicator=True,
                total_latency_ms=1100.0,
                total_tokens=200,
                original_prompt_preview="Test prompt...",
                created_at=datetime(2024, 1, 1, 10, 0, 0),
            ),
        ],
        total=1,
        page=1,
        page_size=50,
        total_pages=1,
        has_next=False,
        has_previous=False,
    )


@pytest.fixture
def mock_telemetry_detail():
    """Create mock telemetry event detail."""
    from app.schemas.campaign_analytics import ExecutionStatusEnum

    return TelemetryEventDetail(
        id="event-001",
        campaign_id="test-campaign-001",
        sequence_number=0,
        original_prompt="Test original prompt",
        transformed_prompt="Test transformed prompt",
        response_text="Test response text",
        technique_suite="dan_persona",
        potency_level=5,
        provider="openai",
        model="gpt-4",
        status=ExecutionStatusEnum.SUCCESS,
        success_indicator=True,
        execution_time_ms=1000.0,
        transformation_time_ms=100.0,
        total_latency_ms=1100.0,
        prompt_tokens=100,
        completion_tokens=100,
        total_tokens=200,
        estimated_cost_cents=1.5,
        semantic_success_score=0.85,
        effectiveness_score=0.8,
        naturalness_score=0.9,
        detectability_score=0.2,
        bypass_indicators=["indicator_1"],
        safety_trigger_detected=False,
        applied_techniques=["technique_1", "technique_2"],
        metadata={},
        created_at=datetime(2024, 1, 1, 10, 0, 0),
    )


@pytest.fixture
def mock_campaign_comparison():
    """Create mock campaign comparison."""
    from app.schemas.campaign_analytics import CampaignComparisonItem, NormalizedMetrics

    return CampaignComparison(
        campaign_ids=["campaign-001", "campaign-002"],
        items=[
            CampaignComparisonItem(
                campaign_id="campaign-001",
                campaign_name="Campaign 1",
                status=CampaignStatusEnum.COMPLETED,
                total_attempts=100,
                success_rate=0.8,
                avg_latency_ms=1000.0,
                avg_tokens=200.0,
                avg_cost_cents=1.5,
                total_cost_cents=150.0,
                best_technique="dan_persona",
                best_provider="openai",
                duration_seconds=3600.0,
                normalized=NormalizedMetrics(
                    success_rate=1.0,
                    latency_score=1.0,
                    cost_efficiency=0.9,
                    effectiveness=0.85,
                    semantic_success=0.82,
                    throughput=0.75,
                ),
            ),
            CampaignComparisonItem(
                campaign_id="campaign-002",
                campaign_name="Campaign 2",
                status=CampaignStatusEnum.COMPLETED,
                total_attempts=80,
                success_rate=0.7,
                avg_latency_ms=1200.0,
                avg_tokens=220.0,
                avg_cost_cents=1.8,
                total_cost_cents=144.0,
                best_technique="cognitive_hacking",
                best_provider="anthropic",
                duration_seconds=4000.0,
                normalized=NormalizedMetrics(
                    success_rate=0.875,
                    latency_score=0.833,
                    cost_efficiency=0.833,
                    effectiveness=0.75,
                    semantic_success=0.72,
                    throughput=0.6,
                ),
            ),
        ],
        best_overall="campaign-001",
        winners={
            "success_rate": "campaign-001",
            "latency": "campaign-001",
            "cost": "campaign-002",
        },
        computed_at=datetime(2024, 1, 1, 12, 0, 0),
    )


# =============================================================================
# Campaign List Endpoint Tests
# =============================================================================


class TestCampaignListEndpoint:
    """Tests for GET /api/v1/campaigns endpoint."""

    @pytest.mark.integration
    def test_list_campaigns_success(
        self, client, auth_headers, mock_campaign_list_response
    ):
        """Test successful campaign list retrieval."""
        with patch(
            "app.api.v1.endpoints.campaign_analytics.get_analytics_service"
        ) as mock_get_service:
            mock_service = AsyncMock(spec=CampaignAnalyticsService)
            mock_service.list_campaigns.return_value = mock_campaign_list_response
            mock_get_service.return_value = mock_service

            response = client.get("/api/v1/campaigns", headers=auth_headers)

            assert response.status_code == 200
            data = response.json()
            assert "items" in data
            assert "total" in data
            assert "page" in data
            assert "page_size" in data
            assert data["total"] == 1

    @pytest.mark.integration
    def test_list_campaigns_with_pagination(
        self, client, auth_headers, mock_campaign_list_response
    ):
        """Test campaign list with pagination parameters."""
        with patch(
            "app.api.v1.endpoints.campaign_analytics.get_analytics_service"
        ) as mock_get_service:
            mock_service = AsyncMock(spec=CampaignAnalyticsService)
            mock_service.list_campaigns.return_value = mock_campaign_list_response
            mock_get_service.return_value = mock_service

            response = client.get(
                "/api/v1/campaigns?page=1&page_size=10", headers=auth_headers
            )

            assert response.status_code == 200
            mock_service.list_campaigns.assert_called_once()
            call_kwargs = mock_service.list_campaigns.call_args
            assert call_kwargs.kwargs["page"] == 1
            assert call_kwargs.kwargs["page_size"] == 10

    @pytest.mark.integration
    def test_list_campaigns_with_status_filter(
        self, client, auth_headers, mock_campaign_list_response
    ):
        """Test campaign list filtered by status."""
        with patch(
            "app.api.v1.endpoints.campaign_analytics.get_analytics_service"
        ) as mock_get_service:
            mock_service = AsyncMock(spec=CampaignAnalyticsService)
            mock_service.list_campaigns.return_value = mock_campaign_list_response
            mock_get_service.return_value = mock_service

            response = client.get(
                "/api/v1/campaigns?status=completed", headers=auth_headers
            )

            assert response.status_code == 200

    @pytest.mark.integration
    def test_list_campaigns_with_provider_filter(
        self, client, auth_headers, mock_campaign_list_response
    ):
        """Test campaign list filtered by provider."""
        with patch(
            "app.api.v1.endpoints.campaign_analytics.get_analytics_service"
        ) as mock_get_service:
            mock_service = AsyncMock(spec=CampaignAnalyticsService)
            mock_service.list_campaigns.return_value = mock_campaign_list_response
            mock_get_service.return_value = mock_service

            response = client.get(
                "/api/v1/campaigns?provider=openai", headers=auth_headers
            )

            assert response.status_code == 200

    @pytest.mark.integration
    def test_list_campaigns_with_search(
        self, client, auth_headers, mock_campaign_list_response
    ):
        """Test campaign list with search query."""
        with patch(
            "app.api.v1.endpoints.campaign_analytics.get_analytics_service"
        ) as mock_get_service:
            mock_service = AsyncMock(spec=CampaignAnalyticsService)
            mock_service.list_campaigns.return_value = mock_campaign_list_response
            mock_get_service.return_value = mock_service

            response = client.get(
                "/api/v1/campaigns?search=jailbreak", headers=auth_headers
            )

            assert response.status_code == 200

    @pytest.mark.integration
    def test_list_campaigns_with_date_filter(
        self, client, auth_headers, mock_campaign_list_response
    ):
        """Test campaign list filtered by date range."""
        with patch(
            "app.api.v1.endpoints.campaign_analytics.get_analytics_service"
        ) as mock_get_service:
            mock_service = AsyncMock(spec=CampaignAnalyticsService)
            mock_service.list_campaigns.return_value = mock_campaign_list_response
            mock_get_service.return_value = mock_service

            response = client.get(
                "/api/v1/campaigns?start_date=2024-01-01T00:00:00Z&end_date=2024-01-31T23:59:59Z",
                headers=auth_headers,
            )

            assert response.status_code == 200

    @pytest.mark.integration
    def test_list_campaigns_invalid_page(self, client, auth_headers):
        """Test campaign list with invalid page number."""
        response = client.get("/api/v1/campaigns?page=0", headers=auth_headers)

        # FastAPI validation should reject page < 1
        assert response.status_code == 422

    @pytest.mark.integration
    def test_list_campaigns_invalid_page_size(self, client, auth_headers):
        """Test campaign list with invalid page size."""
        response = client.get("/api/v1/campaigns?page_size=200", headers=auth_headers)

        # FastAPI validation should reject page_size > 100
        assert response.status_code == 422


# =============================================================================
# Campaign Detail Endpoint Tests
# =============================================================================


class TestCampaignDetailEndpoint:
    """Tests for GET /api/v1/campaigns/{campaign_id} endpoint."""

    @pytest.mark.integration
    def test_get_campaign_detail_success(
        self, client, auth_headers, mock_campaign_detail
    ):
        """Test successful campaign detail retrieval."""
        with patch(
            "app.api.v1.endpoints.campaign_analytics.get_analytics_service"
        ) as mock_get_service:
            mock_service = AsyncMock(spec=CampaignAnalyticsService)
            mock_service.get_campaign_detail.return_value = mock_campaign_detail
            mock_get_service.return_value = mock_service

            response = client.get(
                "/api/v1/campaigns/test-campaign-001", headers=auth_headers
            )

            assert response.status_code == 200
            data = response.json()
            assert data["id"] == "test-campaign-001"
            assert data["name"] == "Test Campaign"
            assert data["status"] == "completed"

    @pytest.mark.integration
    def test_get_campaign_detail_not_found(self, client, auth_headers):
        """Test campaign detail for non-existent campaign."""
        with patch(
            "app.api.v1.endpoints.campaign_analytics.get_analytics_service"
        ) as mock_get_service:
            mock_service = AsyncMock(spec=CampaignAnalyticsService)
            mock_service.get_campaign_detail.return_value = None
            mock_get_service.return_value = mock_service

            response = client.get(
                "/api/v1/campaigns/nonexistent-campaign", headers=auth_headers
            )

            assert response.status_code == 404
            data = response.json()
            assert "detail" in data
            assert "not found" in data["detail"].lower()


# =============================================================================
# Campaign Summary Endpoint Tests
# =============================================================================


class TestCampaignSummaryEndpoint:
    """Tests for GET /api/v1/campaigns/{campaign_id}/summary endpoint."""

    @pytest.mark.integration
    def test_get_campaign_summary_success(
        self, client, auth_headers, mock_campaign_summary
    ):
        """Test successful campaign summary retrieval."""
        with patch(
            "app.api.v1.endpoints.campaign_analytics.get_analytics_service"
        ) as mock_get_service:
            mock_service = AsyncMock(spec=CampaignAnalyticsService)
            mock_service.get_campaign_summary.return_value = mock_campaign_summary
            mock_get_service.return_value = mock_service

            response = client.get(
                "/api/v1/campaigns/test-campaign-001/summary", headers=auth_headers
            )

            assert response.status_code == 200
            data = response.json()
            assert data["id"] == "test-campaign-001"
            assert data["total_attempts"] == 100
            assert data["success_rate"] == 0.75

    @pytest.mark.integration
    def test_get_campaign_summary_not_found(self, client, auth_headers):
        """Test campaign summary for non-existent campaign."""
        with patch(
            "app.api.v1.endpoints.campaign_analytics.get_analytics_service"
        ) as mock_get_service:
            mock_service = AsyncMock(spec=CampaignAnalyticsService)
            mock_service.get_campaign_summary.return_value = None
            mock_get_service.return_value = mock_service

            response = client.get(
                "/api/v1/campaigns/nonexistent-campaign/summary", headers=auth_headers
            )

            assert response.status_code == 404


# =============================================================================
# Campaign Statistics Endpoint Tests
# =============================================================================


class TestCampaignStatisticsEndpoint:
    """Tests for GET /api/v1/campaigns/{campaign_id}/statistics endpoint."""

    @pytest.mark.integration
    def test_get_statistics_success(
        self, client, auth_headers, mock_campaign_statistics
    ):
        """Test successful statistics retrieval."""
        with patch(
            "app.api.v1.endpoints.campaign_analytics.get_analytics_service"
        ) as mock_get_service:
            mock_service = AsyncMock(spec=CampaignAnalyticsService)
            mock_service.calculate_statistics.return_value = mock_campaign_statistics
            mock_get_service.return_value = mock_service

            response = client.get(
                "/api/v1/campaigns/test-campaign-001/statistics", headers=auth_headers
            )

            assert response.status_code == 200
            data = response.json()
            assert data["campaign_id"] == "test-campaign-001"
            assert "attempts" in data
            assert "success_rate" in data
            assert "latency" in data
            assert data["attempts"]["total"] == 100

    @pytest.mark.integration
    def test_get_statistics_not_found(self, client, auth_headers):
        """Test statistics for non-existent campaign."""
        with patch(
            "app.api.v1.endpoints.campaign_analytics.get_analytics_service"
        ) as mock_get_service:
            mock_service = AsyncMock(spec=CampaignAnalyticsService)
            mock_service.calculate_statistics.return_value = None
            mock_get_service.return_value = mock_service

            response = client.get(
                "/api/v1/campaigns/nonexistent-campaign/statistics", headers=auth_headers
            )

            assert response.status_code == 404


# =============================================================================
# Breakdown Endpoint Tests
# =============================================================================


class TestBreakdownEndpoints:
    """Tests for breakdown endpoints (technique, provider, potency)."""

    @pytest.mark.integration
    def test_get_technique_breakdown_success(
        self, client, auth_headers, mock_technique_breakdown
    ):
        """Test successful technique breakdown retrieval."""
        with patch(
            "app.api.v1.endpoints.campaign_analytics.get_analytics_service"
        ) as mock_get_service:
            mock_service = AsyncMock(spec=CampaignAnalyticsService)
            mock_service.get_technique_breakdown.return_value = mock_technique_breakdown
            mock_get_service.return_value = mock_service

            response = client.get(
                "/api/v1/campaigns/test-campaign-001/breakdown/techniques",
                headers=auth_headers,
            )

            assert response.status_code == 200
            data = response.json()
            assert data["campaign_id"] == "test-campaign-001"
            assert "items" in data
            assert data["best_technique"] == "dan_persona"

    @pytest.mark.integration
    def test_get_technique_breakdown_not_found(self, client, auth_headers):
        """Test technique breakdown for non-existent campaign."""
        with patch(
            "app.api.v1.endpoints.campaign_analytics.get_analytics_service"
        ) as mock_get_service:
            mock_service = AsyncMock(spec=CampaignAnalyticsService)
            mock_service.get_technique_breakdown.return_value = None
            mock_get_service.return_value = mock_service

            response = client.get(
                "/api/v1/campaigns/nonexistent-campaign/breakdown/techniques",
                headers=auth_headers,
            )

            assert response.status_code == 404

    @pytest.mark.integration
    def test_get_provider_breakdown_success(
        self, client, auth_headers, mock_provider_breakdown
    ):
        """Test successful provider breakdown retrieval."""
        with patch(
            "app.api.v1.endpoints.campaign_analytics.get_analytics_service"
        ) as mock_get_service:
            mock_service = AsyncMock(spec=CampaignAnalyticsService)
            mock_service.get_provider_breakdown.return_value = mock_provider_breakdown
            mock_get_service.return_value = mock_service

            response = client.get(
                "/api/v1/campaigns/test-campaign-001/breakdown/providers",
                headers=auth_headers,
            )

            assert response.status_code == 200
            data = response.json()
            assert data["best_provider"] == "openai"

    @pytest.mark.integration
    def test_get_potency_breakdown_success(
        self, client, auth_headers, mock_potency_breakdown
    ):
        """Test successful potency breakdown retrieval."""
        with patch(
            "app.api.v1.endpoints.campaign_analytics.get_analytics_service"
        ) as mock_get_service:
            mock_service = AsyncMock(spec=CampaignAnalyticsService)
            mock_service.get_potency_breakdown.return_value = mock_potency_breakdown
            mock_get_service.return_value = mock_service

            response = client.get(
                "/api/v1/campaigns/test-campaign-001/breakdown/potency",
                headers=auth_headers,
            )

            assert response.status_code == 200
            data = response.json()
            assert data["best_potency"] == 7


# =============================================================================
# Time Series Endpoint Tests
# =============================================================================


class TestTimeSeriesEndpoint:
    """Tests for GET /api/v1/campaigns/{campaign_id}/time-series endpoint."""

    @pytest.mark.integration
    def test_get_time_series_success(self, client, auth_headers, mock_time_series):
        """Test successful time series retrieval."""
        with patch(
            "app.api.v1.endpoints.campaign_analytics.get_analytics_service"
        ) as mock_get_service:
            mock_service = AsyncMock(spec=CampaignAnalyticsService)
            mock_service.get_time_series.return_value = mock_time_series
            mock_get_service.return_value = mock_service

            response = client.get(
                "/api/v1/campaigns/test-campaign-001/time-series", headers=auth_headers
            )

            assert response.status_code == 200
            data = response.json()
            assert data["campaign_id"] == "test-campaign-001"
            assert data["metric"] == "success_rate"
            assert "data_points" in data

    @pytest.mark.integration
    def test_get_time_series_with_granularity(
        self, client, auth_headers, mock_time_series
    ):
        """Test time series with specific granularity."""
        with patch(
            "app.api.v1.endpoints.campaign_analytics.get_analytics_service"
        ) as mock_get_service:
            mock_service = AsyncMock(spec=CampaignAnalyticsService)
            mock_service.get_time_series.return_value = mock_time_series
            mock_get_service.return_value = mock_service

            response = client.get(
                "/api/v1/campaigns/test-campaign-001/time-series?granularity=minute",
                headers=auth_headers,
            )

            assert response.status_code == 200

    @pytest.mark.integration
    def test_get_time_series_with_metric(self, client, auth_headers, mock_time_series):
        """Test time series with specific metric."""
        with patch(
            "app.api.v1.endpoints.campaign_analytics.get_analytics_service"
        ) as mock_get_service:
            mock_service = AsyncMock(spec=CampaignAnalyticsService)
            mock_service.get_time_series.return_value = mock_time_series
            mock_get_service.return_value = mock_service

            response = client.get(
                "/api/v1/campaigns/test-campaign-001/time-series?metric=latency",
                headers=auth_headers,
            )

            assert response.status_code == 200

    @pytest.mark.integration
    def test_get_time_series_with_filters(self, client, auth_headers, mock_time_series):
        """Test time series with technique and provider filters."""
        with patch(
            "app.api.v1.endpoints.campaign_analytics.get_analytics_service"
        ) as mock_get_service:
            mock_service = AsyncMock(spec=CampaignAnalyticsService)
            mock_service.get_time_series.return_value = mock_time_series
            mock_get_service.return_value = mock_service

            response = client.get(
                "/api/v1/campaigns/test-campaign-001/time-series?technique_suite=dan_persona&provider=openai",
                headers=auth_headers,
            )

            assert response.status_code == 200

    @pytest.mark.integration
    def test_get_time_series_not_found(self, client, auth_headers):
        """Test time series for non-existent campaign."""
        with patch(
            "app.api.v1.endpoints.campaign_analytics.get_analytics_service"
        ) as mock_get_service:
            mock_service = AsyncMock(spec=CampaignAnalyticsService)
            mock_service.get_time_series.return_value = None
            mock_get_service.return_value = mock_service

            response = client.get(
                "/api/v1/campaigns/nonexistent-campaign/time-series",
                headers=auth_headers,
            )

            assert response.status_code == 404


# =============================================================================
# Campaign Comparison Endpoint Tests
# =============================================================================


class TestCampaignComparisonEndpoint:
    """Tests for POST /api/v1/campaigns/compare endpoint."""

    @pytest.mark.integration
    def test_compare_campaigns_success(
        self, client, auth_headers, mock_campaign_comparison
    ):
        """Test successful campaign comparison."""
        with patch(
            "app.api.v1.endpoints.campaign_analytics.get_analytics_service"
        ) as mock_get_service:
            mock_service = AsyncMock(spec=CampaignAnalyticsService)
            mock_service.compare_campaigns.return_value = mock_campaign_comparison
            mock_get_service.return_value = mock_service

            response = client.post(
                "/api/v1/campaigns/compare",
                json={"campaign_ids": ["campaign-001", "campaign-002"]},
                headers=auth_headers,
            )

            assert response.status_code == 200
            data = response.json()
            assert "campaign_ids" in data
            assert len(data["items"]) == 2
            assert data["best_overall"] == "campaign-001"

    @pytest.mark.integration
    def test_compare_campaigns_with_options(
        self, client, auth_headers, mock_campaign_comparison
    ):
        """Test campaign comparison with optional parameters."""
        with patch(
            "app.api.v1.endpoints.campaign_analytics.get_analytics_service"
        ) as mock_get_service:
            mock_service = AsyncMock(spec=CampaignAnalyticsService)
            mock_service.compare_campaigns.return_value = mock_campaign_comparison
            mock_get_service.return_value = mock_service

            response = client.post(
                "/api/v1/campaigns/compare",
                json={
                    "campaign_ids": ["campaign-001", "campaign-002"],
                    "include_time_series": True,
                    "normalize_metrics": True,
                },
                headers=auth_headers,
            )

            assert response.status_code == 200

    @pytest.mark.integration
    def test_compare_campaigns_four_campaigns(
        self, client, auth_headers, mock_campaign_comparison
    ):
        """Test comparison with maximum allowed campaigns (4)."""
        with patch(
            "app.api.v1.endpoints.campaign_analytics.get_analytics_service"
        ) as mock_get_service:
            mock_service = AsyncMock(spec=CampaignAnalyticsService)
            mock_service.compare_campaigns.return_value = mock_campaign_comparison
            mock_get_service.return_value = mock_service

            response = client.post(
                "/api/v1/campaigns/compare",
                json={
                    "campaign_ids": [
                        "campaign-001",
                        "campaign-002",
                        "campaign-003",
                        "campaign-004",
                    ]
                },
                headers=auth_headers,
            )

            assert response.status_code == 200

    @pytest.mark.integration
    def test_compare_campaigns_single_campaign_invalid(self, client, auth_headers):
        """Test comparison with only one campaign (should fail validation)."""
        response = client.post(
            "/api/v1/campaigns/compare",
            json={"campaign_ids": ["campaign-001"]},
            headers=auth_headers,
        )

        # Validation should fail for less than 2 campaigns
        assert response.status_code == 422

    @pytest.mark.integration
    def test_compare_campaigns_too_many_invalid(self, client, auth_headers):
        """Test comparison with too many campaigns (should fail validation)."""
        response = client.post(
            "/api/v1/campaigns/compare",
            json={
                "campaign_ids": [
                    "campaign-001",
                    "campaign-002",
                    "campaign-003",
                    "campaign-004",
                    "campaign-005",
                ]
            },
            headers=auth_headers,
        )

        # Validation should fail for more than 4 campaigns
        assert response.status_code == 422

    @pytest.mark.integration
    def test_compare_campaigns_not_found(self, client, auth_headers):
        """Test comparison with non-existent campaign."""
        with patch(
            "app.api.v1.endpoints.campaign_analytics.get_analytics_service"
        ) as mock_get_service:
            mock_service = AsyncMock(spec=CampaignAnalyticsService)
            mock_service.compare_campaigns.return_value = None
            mock_get_service.return_value = mock_service

            response = client.post(
                "/api/v1/campaigns/compare",
                json={"campaign_ids": ["campaign-001", "nonexistent-campaign"]},
                headers=auth_headers,
            )

            assert response.status_code == 404

    @pytest.mark.integration
    def test_compare_campaigns_invalid_request(self, client, auth_headers):
        """Test comparison with invalid request body."""
        with patch(
            "app.api.v1.endpoints.campaign_analytics.get_analytics_service"
        ) as mock_get_service:
            mock_service = AsyncMock(spec=CampaignAnalyticsService)
            mock_service.compare_campaigns.side_effect = ValueError(
                "Invalid campaign IDs"
            )
            mock_get_service.return_value = mock_service

            response = client.post(
                "/api/v1/campaigns/compare",
                json={"campaign_ids": ["campaign-001", "campaign-001"]},  # Duplicates
                headers=auth_headers,
            )

            assert response.status_code == 400


# =============================================================================
# Telemetry Events Endpoint Tests
# =============================================================================


class TestTelemetryEventsEndpoint:
    """Tests for telemetry event endpoints."""

    @pytest.mark.integration
    def test_list_events_success(self, client, auth_headers, mock_telemetry_list):
        """Test successful telemetry events list retrieval."""
        with patch(
            "app.api.v1.endpoints.campaign_analytics.get_analytics_service"
        ) as mock_get_service:
            mock_service = AsyncMock(spec=CampaignAnalyticsService)
            mock_service.get_telemetry_events.return_value = mock_telemetry_list
            mock_get_service.return_value = mock_service

            response = client.get(
                "/api/v1/campaigns/test-campaign-001/events", headers=auth_headers
            )

            assert response.status_code == 200
            data = response.json()
            assert "items" in data
            assert "total" in data

    @pytest.mark.integration
    def test_list_events_with_pagination(
        self, client, auth_headers, mock_telemetry_list
    ):
        """Test telemetry events with pagination."""
        with patch(
            "app.api.v1.endpoints.campaign_analytics.get_analytics_service"
        ) as mock_get_service:
            mock_service = AsyncMock(spec=CampaignAnalyticsService)
            mock_service.get_telemetry_events.return_value = mock_telemetry_list
            mock_get_service.return_value = mock_service

            response = client.get(
                "/api/v1/campaigns/test-campaign-001/events?page=1&page_size=25",
                headers=auth_headers,
            )

            assert response.status_code == 200

    @pytest.mark.integration
    def test_list_events_with_filters(self, client, auth_headers, mock_telemetry_list):
        """Test telemetry events with filters."""
        with patch(
            "app.api.v1.endpoints.campaign_analytics.get_analytics_service"
        ) as mock_get_service:
            mock_service = AsyncMock(spec=CampaignAnalyticsService)
            mock_service.get_telemetry_events.return_value = mock_telemetry_list
            mock_get_service.return_value = mock_service

            response = client.get(
                "/api/v1/campaigns/test-campaign-001/events?technique_suite=dan_persona&provider=openai&success_only=true",
                headers=auth_headers,
            )

            assert response.status_code == 200

    @pytest.mark.integration
    def test_get_event_detail_success(
        self, client, auth_headers, mock_telemetry_detail
    ):
        """Test successful telemetry event detail retrieval."""
        with patch(
            "app.api.v1.endpoints.campaign_analytics.get_analytics_service"
        ) as mock_get_service:
            mock_service = AsyncMock(spec=CampaignAnalyticsService)
            mock_service.get_telemetry_event_detail.return_value = mock_telemetry_detail
            mock_get_service.return_value = mock_service

            response = client.get(
                "/api/v1/campaigns/test-campaign-001/events/event-001",
                headers=auth_headers,
            )

            assert response.status_code == 200
            data = response.json()
            assert data["id"] == "event-001"
            assert data["original_prompt"] == "Test original prompt"

    @pytest.mark.integration
    def test_get_event_detail_not_found(self, client, auth_headers):
        """Test telemetry event detail for non-existent event."""
        with patch(
            "app.api.v1.endpoints.campaign_analytics.get_analytics_service"
        ) as mock_get_service:
            mock_service = AsyncMock(spec=CampaignAnalyticsService)
            mock_service.get_telemetry_event_detail.return_value = None
            mock_get_service.return_value = mock_service

            response = client.get(
                "/api/v1/campaigns/test-campaign-001/events/nonexistent-event",
                headers=auth_headers,
            )

            assert response.status_code == 404


# =============================================================================
# Export Endpoint Tests
# =============================================================================


class TestExportEndpoints:
    """Tests for export endpoints (CSV, chart)."""

    @pytest.mark.integration
    def test_export_csv_success(
        self, client, auth_headers, mock_campaign_summary, mock_telemetry_list
    ):
        """Test successful CSV export."""
        with patch(
            "app.api.v1.endpoints.campaign_analytics.get_analytics_service"
        ) as mock_get_service:
            mock_service = AsyncMock(spec=CampaignAnalyticsService)
            mock_service.get_campaign_summary.return_value = mock_campaign_summary
            mock_service.get_telemetry_events.return_value = mock_telemetry_list
            mock_get_service.return_value = mock_service

            response = client.get(
                "/api/v1/campaigns/test-campaign-001/export/csv", headers=auth_headers
            )

            assert response.status_code == 200
            assert response.headers["content-type"] == "text/csv; charset=utf-8"
            assert "content-disposition" in response.headers

    @pytest.mark.integration
    def test_export_csv_with_prompts(
        self, client, auth_headers, mock_campaign_summary, mock_telemetry_list
    ):
        """Test CSV export including prompt text."""
        with patch(
            "app.api.v1.endpoints.campaign_analytics.get_analytics_service"
        ) as mock_get_service:
            mock_service = AsyncMock(spec=CampaignAnalyticsService)
            mock_service.get_campaign_summary.return_value = mock_campaign_summary
            mock_service.get_telemetry_events.return_value = mock_telemetry_list
            mock_get_service.return_value = mock_service

            response = client.get(
                "/api/v1/campaigns/test-campaign-001/export/csv?include_prompts=true",
                headers=auth_headers,
            )

            assert response.status_code == 200

    @pytest.mark.integration
    def test_export_csv_not_found(self, client, auth_headers):
        """Test CSV export for non-existent campaign."""
        with patch(
            "app.api.v1.endpoints.campaign_analytics.get_analytics_service"
        ) as mock_get_service:
            mock_service = AsyncMock(spec=CampaignAnalyticsService)
            mock_service.get_campaign_summary.return_value = None
            mock_get_service.return_value = mock_service

            response = client.get(
                "/api/v1/campaigns/nonexistent-campaign/export/csv",
                headers=auth_headers,
            )

            assert response.status_code == 404

    @pytest.mark.integration
    def test_export_chart_success(
        self, client, auth_headers, mock_campaign_summary, mock_campaign_statistics
    ):
        """Test successful chart export."""
        with patch(
            "app.api.v1.endpoints.campaign_analytics.get_analytics_service"
        ) as mock_get_service:
            mock_service = AsyncMock(spec=CampaignAnalyticsService)
            mock_service.get_campaign_summary.return_value = mock_campaign_summary
            mock_service.calculate_statistics.return_value = mock_campaign_statistics
            mock_get_service.return_value = mock_service

            response = client.post(
                "/api/v1/campaigns/test-campaign-001/export/chart",
                json={"export_type": "chart", "chart_options": {"format": "png"}},
                headers=auth_headers,
            )

            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert data["campaign_id"] == "test-campaign-001"

    @pytest.mark.integration
    def test_export_chart_svg_format(
        self, client, auth_headers, mock_campaign_summary, mock_campaign_statistics
    ):
        """Test chart export in SVG format."""
        with patch(
            "app.api.v1.endpoints.campaign_analytics.get_analytics_service"
        ) as mock_get_service:
            mock_service = AsyncMock(spec=CampaignAnalyticsService)
            mock_service.get_campaign_summary.return_value = mock_campaign_summary
            mock_service.calculate_statistics.return_value = mock_campaign_statistics
            mock_get_service.return_value = mock_service

            response = client.post(
                "/api/v1/campaigns/test-campaign-001/export/chart",
                json={"export_type": "chart", "chart_options": {"format": "svg"}},
                headers=auth_headers,
            )

            assert response.status_code == 200
            data = response.json()
            assert "svg" in data["file_name"]


# =============================================================================
# Cache Management Endpoint Tests
# =============================================================================


class TestCacheManagementEndpoints:
    """Tests for cache management endpoints."""

    @pytest.mark.integration
    def test_invalidate_cache_success(self, client, auth_headers):
        """Test successful cache invalidation."""
        with patch(
            "app.api.v1.endpoints.campaign_analytics.get_analytics_service"
        ) as mock_get_service:
            mock_service = AsyncMock(spec=CampaignAnalyticsService)
            mock_service.invalidate_cache.return_value = None
            mock_get_service.return_value = mock_service

            response = client.delete(
                "/api/v1/campaigns/test-campaign-001/cache", headers=auth_headers
            )

            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True

    @pytest.mark.integration
    def test_get_cache_stats_success(self, client, auth_headers):
        """Test successful cache stats retrieval."""
        with patch(
            "app.api.v1.endpoints.campaign_analytics.get_analytics_service"
        ) as mock_get_service:
            mock_service = AsyncMock(spec=CampaignAnalyticsService)
            mock_service.get_cache_stats.return_value = {
                "size": 10,
                "max_size": 500,
                "hits": 100,
                "misses": 20,
                "hit_rate": 0.833,
            }
            mock_get_service.return_value = mock_service

            response = client.get(
                "/api/v1/campaigns/cache/stats", headers=auth_headers
            )

            assert response.status_code == 200
            data = response.json()
            assert "size" in data
            assert "hits" in data
            assert "hit_rate" in data


# =============================================================================
# Error Response Tests
# =============================================================================


class TestErrorResponses:
    """Tests for error responses and edge cases."""

    @pytest.mark.integration
    def test_unauthenticated_request(self, client):
        """Test request without authentication."""
        response = client.get("/api/v1/campaigns")

        # Should require authentication
        assert response.status_code in [401, 403]

    @pytest.mark.integration
    def test_invalid_api_key(self, client):
        """Test request with invalid API key."""
        response = client.get(
            "/api/v1/campaigns", headers={"X-API-Key": "invalid-key"}
        )

        # Should reject invalid API key
        assert response.status_code in [401, 403]

    @pytest.mark.integration
    def test_invalid_campaign_id_format(self, client, auth_headers):
        """Test with invalid campaign ID format."""
        with patch(
            "app.api.v1.endpoints.campaign_analytics.get_analytics_service"
        ) as mock_get_service:
            mock_service = AsyncMock(spec=CampaignAnalyticsService)
            mock_service.get_campaign_detail.return_value = None
            mock_get_service.return_value = mock_service

            # Very long ID (potential DoS)
            long_id = "a" * 1000
            response = client.get(f"/api/v1/campaigns/{long_id}", headers=auth_headers)

            # Should either reject or return 404
            assert response.status_code in [404, 400, 422]

    @pytest.mark.integration
    def test_service_error_handling(self, client, auth_headers):
        """Test handling of service-level errors."""
        with patch(
            "app.api.v1.endpoints.campaign_analytics.get_analytics_service"
        ) as mock_get_service:
            mock_service = AsyncMock(spec=CampaignAnalyticsService)
            mock_service.get_campaign_detail.side_effect = Exception("Database error")
            mock_get_service.return_value = mock_service

            response = client.get(
                "/api/v1/campaigns/test-campaign-001", headers=auth_headers
            )

            # Should return 500 for internal errors
            assert response.status_code == 500


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    @pytest.mark.integration
    def test_empty_campaign_list(self, client, auth_headers):
        """Test listing with no campaigns."""
        empty_response = CampaignListResponse(
            items=[],
            total=0,
            page=1,
            page_size=20,
            total_pages=0,
            has_next=False,
            has_previous=False,
        )

        with patch(
            "app.api.v1.endpoints.campaign_analytics.get_analytics_service"
        ) as mock_get_service:
            mock_service = AsyncMock(spec=CampaignAnalyticsService)
            mock_service.list_campaigns.return_value = empty_response
            mock_get_service.return_value = mock_service

            response = client.get("/api/v1/campaigns", headers=auth_headers)

            assert response.status_code == 200
            data = response.json()
            assert data["items"] == []
            assert data["total"] == 0

    @pytest.mark.integration
    def test_special_characters_in_search(
        self, client, auth_headers, mock_campaign_list_response
    ):
        """Test search with special characters."""
        with patch(
            "app.api.v1.endpoints.campaign_analytics.get_analytics_service"
        ) as mock_get_service:
            mock_service = AsyncMock(spec=CampaignAnalyticsService)
            mock_service.list_campaigns.return_value = mock_campaign_list_response
            mock_get_service.return_value = mock_service

            response = client.get(
                "/api/v1/campaigns?search=%3Cscript%3Ealert%28%27xss%27%29%3C%2Fscript%3E",
                headers=auth_headers,
            )

            # Should handle safely, not error
            assert response.status_code == 200

    @pytest.mark.integration
    def test_unicode_in_campaign_name(self, client, auth_headers):
        """Test campaign with unicode characters in name."""
        unicode_summary = CampaignSummary(
            id="test-unicode-001",
            name="Test Campaign 日本語 🚀",
            description="Unicode test",
            objective="Test unicode handling",
            status=CampaignStatusEnum.COMPLETED,
            total_attempts=10,
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )

        with patch(
            "app.api.v1.endpoints.campaign_analytics.get_analytics_service"
        ) as mock_get_service:
            mock_service = AsyncMock(spec=CampaignAnalyticsService)
            mock_service.get_campaign_summary.return_value = unicode_summary
            mock_get_service.return_value = mock_service

            response = client.get(
                "/api/v1/campaigns/test-unicode-001/summary", headers=auth_headers
            )

            assert response.status_code == 200
            data = response.json()
            assert "日本語" in data["name"]
            assert "🚀" in data["name"]

    @pytest.mark.integration
    def test_large_page_size(self, client, auth_headers):
        """Test pagination with maximum allowed page size."""
        with patch(
            "app.api.v1.endpoints.campaign_analytics.get_analytics_service"
        ) as mock_get_service:
            mock_service = AsyncMock(spec=CampaignAnalyticsService)
            mock_service.list_campaigns.return_value = CampaignListResponse(
                items=[],
                total=0,
                page=1,
                page_size=100,
                total_pages=0,
                has_next=False,
                has_previous=False,
            )
            mock_get_service.return_value = mock_service

            response = client.get(
                "/api/v1/campaigns?page_size=100", headers=auth_headers
            )

            assert response.status_code == 200

    @pytest.mark.integration
    def test_boundary_potency_values(self, client, auth_headers, mock_telemetry_list):
        """Test filtering with boundary potency values (1 and 10)."""
        with patch(
            "app.api.v1.endpoints.campaign_analytics.get_analytics_service"
        ) as mock_get_service:
            mock_service = AsyncMock(spec=CampaignAnalyticsService)
            mock_service.get_telemetry_events.return_value = mock_telemetry_list
            mock_get_service.return_value = mock_service

            response = client.get(
                "/api/v1/campaigns/test-campaign-001/events?min_potency=1&max_potency=10",
                headers=auth_headers,
            )

            assert response.status_code == 200

    @pytest.mark.integration
    def test_invalid_potency_values(self, client, auth_headers):
        """Test filtering with invalid potency values."""
        response = client.get(
            "/api/v1/campaigns/test-campaign-001/events?min_potency=0",
            headers=auth_headers,
        )

        # Should reject potency < 1
        assert response.status_code == 422

        response = client.get(
            "/api/v1/campaigns/test-campaign-001/events?max_potency=11",
            headers=auth_headers,
        )

        # Should reject potency > 10
        assert response.status_code == 422
