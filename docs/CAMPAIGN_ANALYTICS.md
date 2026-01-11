# Campaign Telemetry Analytics Panel

Post-campaign analytics view with detailed breakdowns, statistical analysis, comparative metrics across campaigns, and exportable charts/graphs for research papers and security reports.

**Version**: 1.0.0
**Last Updated**: 2026-01-11
**Status**: Complete

---

## Table of Contents

1. [Feature Overview](#feature-overview)
2. [Architecture](#architecture)
3. [API Reference](#api-reference)
4. [Component Library](#component-library)
5. [Export Formats](#export-formats)
6. [Usage Examples](#usage-examples)
7. [Configuration](#configuration)
8. [Troubleshooting](#troubleshooting)

---

## Feature Overview

The Campaign Telemetry Analytics Panel provides comprehensive post-mortem analysis capabilities for adversarial prompt campaigns. It addresses the pain point of manual tracking of campaign telemetry and results, enabling researchers to:

- **Analyze Performance**: View statistical summaries including mean, median, p95 success rates
- **Compare Campaigns**: Side-by-side comparison of up to 4 campaigns with normalized metrics
- **Visualize Trends**: Time-series charts showing prompt evolution and success rate correlation
- **Export Results**: Generate PNG/SVG charts and CSV data for research publications
- **Drill Down**: Filter by technique, provider, potency level, or time range

### Target Users

| User Type | Primary Use Case |
|-----------|-----------------|
| Academic Researcher | Compare campaign results for publications on technique effectiveness |
| Security Consultant | Export telemetry charts for client audit reports |
| AI Safety Team | Analyze attack pattern trends to recommend guardrail improvements |

### Key Metrics

The analytics panel tracks and displays:

- **Success Rate**: Overall and per-technique/provider success rates
- **Latency Distribution**: P50, P90, P95, P99 latency percentiles
- **Token Usage**: Prompt, completion, and total token consumption
- **Cost Analysis**: Per-attempt and total cost metrics
- **Semantic Success**: Quality scores for successful bypasses
- **Effectiveness Score**: Composite effectiveness metric

---

## Architecture

### System Architecture Diagram

```
┌──────────────────────────────────────────────────────────────────────────┐
│                            Frontend (Next.js)                             │
├──────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────────────────────┐ │
│  │                  CampaignAnalyticsDashboard                          │ │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────────────┐ │ │
│  │  │CampaignSelector │  │StatsSummaryPanel│  │    FilterBar         │ │ │
│  │  └─────────────────┘  └─────────────────┘  └──────────────────────┘ │ │
│  │  ┌──────────────────────────────────────────────────────────────┐   │ │
│  │  │                    Tab Views                                   │   │ │
│  │  │  ┌──────────┬──────────┬────────────┬──────────────────────┐ │   │ │
│  │  │  │ Overview │  Charts  │ Comparison │     Raw Data         │ │   │ │
│  │  │  └──────────┴──────────┴────────────┴──────────────────────┘ │   │ │
│  │  └──────────────────────────────────────────────────────────────┘   │ │
│  └─────────────────────────────────────────────────────────────────────┘ │
│                                   │                                       │
│                        TanStack Query Hooks                               │
│                                   │                                       │
└───────────────────────────────────┼───────────────────────────────────────┘
                                    │ HTTP/REST
┌───────────────────────────────────┼───────────────────────────────────────┐
│                           FastAPI Backend                                 │
├───────────────────────────────────┼───────────────────────────────────────┤
│  ┌───────────────────────────────────────────────────────────────────┐   │
│  │                  Campaign Analytics Router                         │   │
│  │                   /api/v1/campaigns/*                              │   │
│  └───────────────────────────────────────────────────────────────────┘   │
│                                   │                                       │
│  ┌───────────────────────────────────────────────────────────────────┐   │
│  │              CampaignAnalyticsService                              │   │
│  │  ┌──────────────────────┐  ┌────────────────────────────────────┐ │   │
│  │  │   AnalyticsCache     │  │     Statistics Calculator         │ │   │
│  │  │   (TTL-based LRU)    │  │  (mean, median, p95, std_dev)    │ │   │
│  │  └──────────────────────┘  └────────────────────────────────────┘ │   │
│  └───────────────────────────────────────────────────────────────────┘   │
│                                   │                                       │
│  ┌───────────────────────────────────────────────────────────────────┐   │
│  │                      SQLAlchemy ORM                                │   │
│  │  Campaign | CampaignTelemetryEvent | CampaignResult                │   │
│  └───────────────────────────────────────────────────────────────────┘   │
└───────────────────────────────────────────────────────────────────────────┘
```

### Backend Components

| Component | File | Description |
|-----------|------|-------------|
| **Campaign Models** | `backend-api/app/infrastructure/database/campaign_models.py` | SQLAlchemy models for campaigns, telemetry events, and results |
| **Pydantic Schemas** | `backend-api/app/schemas/campaign_analytics.py` | Request/response models with validation |
| **Analytics Service** | `backend-api/app/services/campaign_analytics_service.py` | Business logic with caching |
| **API Endpoints** | `backend-api/app/api/v1/endpoints/campaign_analytics.py` | REST API endpoints |

### Frontend Components

| Component | File | Description |
|-----------|------|-------------|
| **Dashboard** | `frontend/src/components/campaign-analytics/CampaignAnalyticsDashboard.tsx` | Main analytics view |
| **Query Hooks** | `frontend/src/lib/api/query/campaign-queries.ts` | TanStack Query data fetching |
| **TypeScript Types** | `frontend/src/types/campaign-analytics.ts` | Type definitions |
| **Charts** | `frontend/src/components/campaign-analytics/charts/` | Recharts visualizations |

---

## API Reference

All endpoints are prefixed with `/api/v1/campaigns`.

### Campaign List & Detail

#### List Campaigns
```http
GET /api/v1/campaigns
```

**Query Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `page` | integer | 1 | Page number (1-indexed) |
| `page_size` | integer | 20 | Items per page (max: 100) |
| `sort_by` | string | "created_at" | Sort field |
| `sort_order` | string | "desc" | Sort order (asc/desc) |
| `status` | array | - | Filter by status (draft, running, completed, etc.) |
| `provider` | array | - | Filter by LLM provider |
| `technique_suite` | array | - | Filter by technique suite |
| `tags` | array | - | Filter by tags |
| `start_date` | datetime | - | Created after this date |
| `end_date` | datetime | - | Created before this date |
| `search` | string | - | Search in name/description (max 100 chars) |

**Response:**
```json
{
  "items": [
    {
      "id": "550e8400-e29b-41d4-a716-446655440000",
      "name": "GPT-4 Jailbreak Benchmark",
      "objective": "Evaluate jailbreak techniques",
      "status": "completed",
      "target_provider": "openai",
      "target_model": "gpt-4",
      "technique_suites": ["dan_persona"],
      "total_attempts": 100,
      "success_rate": 0.75,
      "avg_latency_ms": 1250.5,
      "started_at": "2024-01-15T10:00:00Z",
      "completed_at": "2024-01-15T12:30:00Z"
    }
  ],
  "total": 42,
  "page": 1,
  "page_size": 20,
  "total_pages": 3
}
```

#### Get Campaign Detail
```http
GET /api/v1/campaigns/{campaign_id}
```

#### Get Campaign Summary
```http
GET /api/v1/campaigns/{campaign_id}/summary
```

### Statistics & Analytics

#### Get Campaign Statistics
```http
GET /api/v1/campaigns/{campaign_id}/statistics
```

**Response:**
```json
{
  "campaign_id": "550e8400-e29b-41d4-a716-446655440000",
  "attempts": {
    "total": 100,
    "successes": 75,
    "failures": 20,
    "partial_successes": 5
  },
  "success_rate": {
    "mean": 0.75,
    "median": 0.78,
    "std_dev": 0.15,
    "p50": 0.78,
    "p90": 0.92,
    "p95": 0.95,
    "p99": 0.98
  },
  "latency_ms": {
    "mean": 1250.5,
    "median": 1180.0,
    "std_dev": 320.4,
    "p50": 1180.0,
    "p90": 1650.0,
    "p95": 1920.0,
    "p99": 2450.0
  },
  "token_usage": {
    "prompt": { "mean": 450, "median": 420 },
    "completion": { "mean": 280, "median": 250 },
    "total": { "mean": 730, "median": 680 }
  },
  "cost": {
    "mean": 0.0023,
    "median": 0.0021,
    "total": 0.23
  }
}
```

### Breakdown Endpoints

#### Technique Breakdown
```http
GET /api/v1/campaigns/{campaign_id}/breakdown/techniques
```

**Response:**
```json
{
  "campaign_id": "550e8400-e29b-41d4-a716-446655440000",
  "items": [
    {
      "name": "dan_persona",
      "attempts": 50,
      "successes": 42,
      "success_rate": 0.84,
      "avg_latency_ms": 1150.5,
      "avg_tokens": 680,
      "avg_cost": 0.0021
    },
    {
      "name": "cognitive_hacking",
      "attempts": 50,
      "successes": 33,
      "success_rate": 0.66,
      "avg_latency_ms": 1350.2,
      "avg_tokens": 780,
      "avg_cost": 0.0025
    }
  ]
}
```

#### Provider Breakdown
```http
GET /api/v1/campaigns/{campaign_id}/breakdown/providers
```

#### Potency Breakdown
```http
GET /api/v1/campaigns/{campaign_id}/breakdown/potency
```

### Time Series Data

```http
GET /api/v1/campaigns/{campaign_id}/time-series
```

**Query Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `metric` | string | "success_rate" | Metric to chart (success_rate, latency, tokens, cost) |
| `granularity` | string | "hour" | Time bucket (minute, hour, day) |
| `start_time` | datetime | - | Series start time |
| `end_time` | datetime | - | Series end time |
| `technique_suite` | array | - | Filter by technique |
| `provider` | array | - | Filter by provider |

**Response:**
```json
{
  "campaign_id": "550e8400-e29b-41d4-a716-446655440000",
  "metric": "success_rate",
  "granularity": "hour",
  "data_points": [
    {
      "timestamp": "2024-01-15T10:00:00Z",
      "value": 0.65,
      "count": 25
    },
    {
      "timestamp": "2024-01-15T11:00:00Z",
      "value": 0.78,
      "count": 35
    }
  ]
}
```

### Campaign Comparison

```http
POST /api/v1/campaigns/compare
```

**Request Body:**
```json
{
  "campaign_ids": [
    "550e8400-e29b-41d4-a716-446655440000",
    "660f9400-f39c-52e5-b827-557766550001"
  ],
  "include_time_series": true,
  "normalize_metrics": true
}
```

**Response:**
```json
{
  "campaigns": [
    {
      "id": "550e8400-e29b-41d4-a716-446655440000",
      "name": "GPT-4 Benchmark",
      "success_rate": 0.75,
      "avg_latency_ms": 1250.5,
      "total_cost": 0.23,
      "normalized": {
        "success_rate": 0.85,
        "latency_score": 0.72,
        "cost_efficiency": 0.88
      }
    }
  ],
  "delta": {
    "success_rate": 0.15,
    "latency_ms": -180.3,
    "cost": -0.05
  },
  "best_performers": {
    "success_rate": "550e8400-e29b-41d4-a716-446655440000",
    "latency": "660f9400-f39c-52e5-b827-557766550001",
    "cost": "660f9400-f39c-52e5-b827-557766550001"
  }
}
```

### Telemetry Events

#### List Telemetry Events
```http
GET /api/v1/campaigns/{campaign_id}/events
```

**Query Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `page` | integer | 1 | Page number |
| `page_size` | integer | 50 | Items per page (max: 200) |
| `status` | array | - | Filter by event status |
| `technique_suite` | array | - | Filter by technique |
| `provider` | array | - | Filter by provider |
| `model` | array | - | Filter by model |
| `success_only` | boolean | - | Only successful attempts |
| `start_time` | datetime | - | Events after this time |
| `end_time` | datetime | - | Events before this time |
| `min_potency` | integer | - | Minimum potency (1-10) |
| `max_potency` | integer | - | Maximum potency (1-10) |

#### Get Event Detail
```http
GET /api/v1/campaigns/{campaign_id}/events/{event_id}
```

### Export Endpoints

#### CSV Export
```http
GET /api/v1/campaigns/{campaign_id}/export/csv
```

**Query Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `include_prompts` | boolean | false | Include full prompt text |
| `include_responses` | boolean | false | Include full response text |

**Response:** Streaming CSV file download

#### Chart Export
```http
POST /api/v1/campaigns/{campaign_id}/export/chart
```

**Request Body:**
```json
{
  "export_type": "chart",
  "chart_options": {
    "format": "png",
    "width": 1920,
    "height": 1080,
    "scale": 2
  }
}
```

### Cache Management

#### Invalidate Cache
```http
DELETE /api/v1/campaigns/{campaign_id}/cache
```

#### Get Cache Stats
```http
GET /api/v1/campaigns/cache/stats
```

---

## Component Library

The Campaign Analytics feature exports 80+ React components with full TypeScript support.

### Core Components

#### CampaignAnalyticsDashboard

The main dashboard component combining all analytics features.

```tsx
import { CampaignAnalyticsDashboard } from '@/components/campaign-analytics';

function AnalyticsPage() {
  return (
    <CampaignAnalyticsDashboard
      initialCampaignId="550e8400-e29b-41d4-a716-446655440000"
      defaultTab="overview"
      showFilters={true}
      showExport={true}
    />
  );
}
```

**Props:**

| Prop | Type | Default | Description |
|------|------|---------|-------------|
| `initialCampaignId` | string | - | Pre-selected campaign ID |
| `defaultTab` | DashboardTab | "overview" | Default active tab |
| `showFilters` | boolean | true | Show filter bar |
| `showExport` | boolean | true | Show export panel |
| `compact` | boolean | false | Compact layout mode |

**Variants:**
- `CompactCampaignAnalyticsDashboard` - For embedded widgets
- `SimpleCampaignAnalyticsDashboard` - Without sidebar/fullscreen

#### StatisticsCard

Displays statistical metrics with support for trends and deltas.

```tsx
import { StatisticsCard, SuccessRateCard, LatencyCard } from '@/components/campaign-analytics';

// Basic usage
<StatisticsCard
  label="Success Rate"
  value={0.75}
  format="percentage"
  metricType="mean"
  delta={0.05}
  deltaPositiveIsGood={true}
/>

// Preset cards
<SuccessRateCard stats={statistics.success_rate} />
<LatencyCard stats={statistics.latency_ms} showP95={true} />
```

**Props:**

| Prop | Type | Description |
|------|------|-------------|
| `label` | string | Metric label |
| `value` | number \| null | Metric value |
| `format` | ValueFormat | Display format (number, percentage, duration, currency, compact) |
| `metricType` | StatisticMetricType | Type badge (mean, median, p95, etc.) |
| `delta` | number | Change from baseline |
| `trend` | TrendDirection | Trend direction (up, down, neutral) |
| `color` | ColorVariant | Card color variant |

**Preset Cards:**
- `SuccessRateCard` - Pre-configured for success rate metrics
- `LatencyCard` - Pre-configured for latency with P95 display
- `TokenUsageCard` - Pre-configured for token counts
- `CostCard` - Pre-configured for cost metrics

#### CampaignSelector

Searchable dropdown for campaign selection with multi-select support.

```tsx
import {
  CampaignSelector,
  CampaignComparisonSelector
} from '@/components/campaign-analytics';

// Single selection
<CampaignSelector
  selectedId={selectedCampaign}
  onSelect={(id) => setSelectedCampaign(id)}
  statusFilter={['completed']}
/>

// Multi-select for comparison (2-4 campaigns)
<CampaignComparisonSelector
  selectedIds={comparisonIds}
  onChange={(ids) => setComparisonIds(ids)}
/>
```

**Props:**

| Prop | Type | Description |
|------|------|-------------|
| `mode` | SelectionMode | "single" or "multi" |
| `selectedId` | string | Selected campaign ID (single mode) |
| `selectedIds` | string[] | Selected campaign IDs (multi mode) |
| `onSelect` | function | Selection callback |
| `maxSelections` | number | Maximum selections (default: 4) |
| `statusFilter` | CampaignStatusEnum[] | Filter by campaign status |

**Variants:**
- `CampaignSelectorSingle` - Pre-configured single select
- `CampaignSelectorMulti` - Pre-configured multi-select
- `CampaignComparisonSelector` - 2-4 completed campaigns only

### Chart Components

All chart components use Recharts and support export functionality.

#### SuccessRateTimeSeriesChart

Line chart showing success rate evolution over time.

```tsx
import { SuccessRateTimeSeriesChart } from '@/components/campaign-analytics';

<SuccessRateTimeSeriesChart
  campaignId={campaignId}
  granularity="hour"
  showZoomControls={true}
  onExport={(format) => handleExport(format)}
/>
```

**Features:**
- Zoom controls (in/out/reset)
- Granularity selector (minute/hour/day)
- Custom tooltips with count badges
- Multiple series for comparison
- Trend indicator

**Variants:**
- `SimpleSuccessRateChart` - Without controls
- `ComparisonSuccessRateChart` - Multiple campaign overlay

#### TechniqueEffectivenessChart

Bar/radar chart comparing technique success rates.

```tsx
import {
  TechniqueEffectivenessChart,
  TechniqueRadarChart
} from '@/components/campaign-analytics';

<TechniqueEffectivenessChart
  campaignId={campaignId}
  viewMode="bar" // or "radar"
  sortBy="success_rate"
  onDrillDown={(technique) => handleDrillDown(technique)}
/>
```

**Features:**
- Bar/radar view toggle
- Sorting by name, success rate, attempts, latency
- Color-coded success tiers (excellent, good, moderate, poor, critical)
- Drill-down callback

**Variants:**
- `SimpleTechniqueChart` - Minimal controls
- `TechniqueRadarChart` - Radar view only
- `CompactTechniqueChart` - For dashboards

#### ProviderComparisonChart

Grouped bar chart comparing LLM provider performance.

```tsx
import { ProviderComparisonChart } from '@/components/campaign-analytics';

<ProviderComparisonChart
  campaignId={campaignId}
  showLatencyOverlay={true}
  showCostMetrics={true}
  displayMode="grouped"
/>
```

**Features:**
- Latency overlay line chart
- Cost metrics display
- Provider-specific colors (OpenAI, Anthropic, Google, etc.)
- Three display modes: success_rate, grouped, stacked

**Variants:**
- `SimpleProviderChart` - Without controls
- `ProviderLatencyChart` - Latency focus
- `ProviderCostChart` - Cost focus
- `CompactProviderChart` - Top 5 only

#### PromptEvolutionChart

Scatter plot showing prompt iteration vs success rate correlation.

```tsx
import { PromptEvolutionChart } from '@/components/campaign-analytics';

<PromptEvolutionChart
  campaignId={campaignId}
  events={telemetryEvents}
  showTrendLine={true}
  showCorrelation={true}
/>
```

**Features:**
- Linear regression trend line
- Pearson correlation coefficient (r value)
- Interactive tooltips with prompt snippets
- Cumulative success rate toggle
- Point sizing by token count

**Variants:**
- `SimplePromptEvolutionChart` - Without controls
- `CompactPromptEvolutionChart` - For widgets
- `DetailedPromptEvolutionChart` - Full-featured

#### LatencyDistributionChart

Histogram/box plot showing latency distribution.

```tsx
import { LatencyDistributionChart } from '@/components/campaign-analytics';

<LatencyDistributionChart
  campaignId={campaignId}
  events={telemetryEvents}
  binCount={20}
  showPercentileMarkers={true}
  viewMode="histogram" // or "boxplot", "combined"
/>
```

**Features:**
- Configurable bin count
- Percentile markers (P50, P90, P95, P99)
- Technique/provider filtering
- Latency tier color-coding (Fast, Normal, Slow, Very Slow)
- Statistics summary panel

**Variants:**
- `SimpleLatencyChart` - Minimal controls
- `CompactLatencyChart` - For dashboards
- `DetailedLatencyChart` - Full-featured
- `PercentileLatencyChart` - Percentile focus

### Comparison Components

#### CampaignComparisonPanel

Main comparison view for 2-4 campaigns.

```tsx
import { CampaignComparisonPanel } from '@/components/campaign-analytics';

<CampaignComparisonPanel
  selectedCampaignIds={comparisonIds}
  onSelectionChange={(ids) => setComparisonIds(ids)}
  viewMode="combined" // or "table", "chart"
  onExport={handleExport}
/>
```

**Features:**
- Multi-select campaign selector (2-4 campaigns)
- Comparison table with rankings
- Normalized metrics radar chart
- View mode toggle
- Export support
- Fullscreen mode

**Variants:**
- `SimpleComparisonPanel` - Without header controls
- `CompactComparisonPanel` - Chart-only view
- `DetailedComparisonPanel` - Full-featured

#### ComparisonTable

Metrics table with campaign columns.

```tsx
import { ComparisonTable } from '@/components/campaign-analytics';

<ComparisonTable
  comparison={comparisonData}
  highlightBest={true}
  showDeltas={true}
  expandable={true}
/>
```

**Features:**
- Best/worst value highlighting
- Rank badges (1st, 2nd, 3rd)
- Expandable metric groups
- Delta indicators
- Custom value formatters

#### NormalizedMetricsChart

Radar chart with normalized (0-1) metrics.

```tsx
import { NormalizedMetricsChart } from '@/components/campaign-analytics';

<NormalizedMetricsChart
  comparison={comparisonData}
  metrics={['success_rate', 'latency_score', 'cost_efficiency']}
  onMetricVisibilityChange={handleVisibilityChange}
/>
```

**Features:**
- Campaign overlay with color-coded series
- Interactive legend
- Metric visibility controls
- Custom tooltips with descriptions

#### DeltaIndicator

Compact delta display component.

```tsx
import {
  DeltaIndicator,
  SuccessRateDelta,
  LatencyDelta
} from '@/components/campaign-analytics';

// Generic
<DeltaIndicator
  current={0.85}
  baseline={0.75}
  mode="percentage"
  direction="higher_is_better"
/>

// Presets
<SuccessRateDelta current={0.85} baseline={0.75} />
<LatencyDelta current={1200} baseline={1500} />
```

**Variants:**
- `PercentageDeltaIndicator` - Percentage mode
- `AbsoluteDeltaIndicator` - Absolute mode
- `SuccessRateDelta` - Success rate preset
- `LatencyDelta` - Latency preset
- `CostDelta` - Cost preset
- `TokenDelta` - Token preset
- `InlineDelta` - Extra-small inline
- `LargeDelta` - Large with tooltip
- `DeltaComparison` - Full comparison display
- `DeltaBadge` - Badge-style indicator

### Filter Components

#### FilterBar

Horizontal filter controls for telemetry data.

```tsx
import { FilterBar, createDefaultFilterState } from '@/components/campaign-analytics';

const [filters, setFilters] = useState(createDefaultFilterState());

<FilterBar
  campaignId={campaignId}
  value={filters}
  onChange={setFilters}
  showActiveCount={true}
/>
```

**Features:**
- Technique multi-select
- Provider multi-select
- Date range picker with presets
- Status filter (success/failure/partial)
- Clear all button
- Active filter chips

**Variants:**
- `CompactFilterBar` - Reduced size
- `InlineFilterBar` - Minimal inline

#### DateRangePicker

Date range selector with presets.

```tsx
import { DateRangePicker, AnalyticsDateRangePicker } from '@/components/campaign-analytics';

<DateRangePicker
  value={dateRange}
  onChange={setDateRange}
  presets={['today', 'last_7_days', 'last_30_days', 'custom']}
  showCalendar={true}
/>

// Analytics-optimized
<AnalyticsDateRangePicker
  value={dateRange}
  onChange={setDateRange}
/>
```

**Presets:**
- today, yesterday
- last_7_days, last_30_days, last_90_days
- this_week, last_week
- this_month, last_month
- this_year, last_year
- all_time

**Variants:**
- `SimpleDateRangePicker` - Presets only
- `CompactDateRangePicker` - Smaller size
- `CalendarOnlyDateRangePicker` - No presets
- `AnalyticsDateRangePicker` - Dashboard optimized

### Detail Components

#### TelemetryDetailModal

Modal showing detailed telemetry for a single execution.

```tsx
import { TelemetryDetailModal } from '@/components/campaign-analytics';

<TelemetryDetailModal
  campaignId={campaignId}
  eventId={selectedEventId}
  open={isOpen}
  onOpenChange={setIsOpen}
  onNavigate={(direction) => navigateEvents(direction)}
/>
```

**Features:**
- 4 tabbed views: Prompt, Response, Timing, Quality
- Keyboard navigation (arrow keys)
- Copy-to-clipboard functionality
- Status badges

**Sub-components:**
- `PromptDisplay` - Original/transformed toggle
- `ResponseDisplay` - Response with bypass indicators
- `TimingBreakdown` - Visual percentage bar
- `QualityScores` - Score cards

#### TelemetryTable

Paginated table of telemetry events.

```tsx
import { TelemetryTable } from '@/components/campaign-analytics';

<TelemetryTable
  campaignId={campaignId}
  filters={filters}
  onRowClick={(event) => openDetailModal(event)}
  sortable={true}
  exportable={true}
/>
```

**Features:**
- Paginated with configurable page size
- Sortable columns
- Row click opens detail modal
- CSV export button
- Loading/error/empty states

**Columns:**
- Timestamp
- Technique
- Provider
- Status
- Latency
- Tokens

**Variants:**
- `SimpleTelemetryTable` - No export/modal
- `CompactTelemetryTable` - For dashboards
- `DetailedTelemetryTable` - Full-featured

### Export Components

#### ExportButton

Dropdown button with export options.

```tsx
import {
  ExportButton,
  ChartExportButton,
  DataExportButton
} from '@/components/campaign-analytics';

// Full export button
<ExportButton
  chartRef={chartRef}
  data={telemetryEvents}
  filename="campaign-analytics"
  onExportComplete={(result) => handleComplete(result)}
/>

// Chart-only
<ChartExportButton chartRef={chartRef} />

// Data-only
<DataExportButton data={events} columns={csvColumns} />
```

**Variants:**
- `ChartExportButton` - PNG/SVG only
- `DataExportButton` - CSV only
- `CompactExportButton` - Icon-only
- `FullExportButton` - All options

#### ExportPanel

Bulk export panel for multiple items.

```tsx
import { ExportPanel, ExportPanelSheet } from '@/components/campaign-analytics';

<ExportPanel
  charts={[
    { id: 'success', label: 'Success Rate', ref: successChartRef },
    { id: 'latency', label: 'Latency', ref: latencyChartRef }
  ]}
  data={[
    { id: 'events', label: 'Telemetry Events', data: events }
  ]}
  onExportComplete={handleComplete}
/>

// Sheet variant
<ExportPanelSheet trigger={<Button>Export All</Button>} />
```

**Features:**
- Chart selection with checkboxes
- Format selection (PNG/SVG)
- Resolution slider (1x-4x)
- Background color presets
- ZIP archive generation
- Progress indicator

**Variants:**
- `ExportPanelSheet` - Slide-out sheet
- `ExportPanelCompact` - Compact layout
- `ExportPanelTrigger` - Button with badge

---

## Export Formats

### Chart Export (PNG/SVG)

Charts can be exported in two formats:

| Format | Use Case | Features |
|--------|----------|----------|
| **PNG** | Publications, reports | Raster image, configurable resolution (1x-4x), transparent or colored background |
| **SVG** | Print, vector editing | Vector graphics, infinite scaling, editable in Illustrator/Inkscape |

**PNG Export Options:**
- Scale: 1x to 4x (default: 2x for high-DPI displays)
- Background: White, gray, dark, or transparent
- Padding: Configurable margin
- Quality: 0.92 (high quality JPEG compression for PNG fallback)

**SVG Export Options:**
- Inline styles for portability
- Custom CSS injection
- Optimization (remove unused elements)

### CSV Export

Telemetry data can be exported as CSV for external analysis tools.

**Default Columns:**
- `id` - Event unique identifier
- `sequence_number` - Execution order
- `technique_suite` - Applied technique
- `potency_level` - Potency level (1-10)
- `provider` - LLM provider
- `model` - Model name
- `status` - Execution status
- `success_indicator` - Boolean success flag
- `total_latency_ms` - Total latency
- `total_tokens` - Total token usage
- `created_at` - Timestamp (ISO 8601)

**Optional Columns:**
- `original_prompt` - Full original prompt text
- `response_preview` - Response preview

**Column Presets:**
```typescript
import { CAMPAIGN_CSV_COLUMNS } from '@/lib/utils/csv-export';

// Available presets
CAMPAIGN_CSV_COLUMNS.campaignSummary    // Campaign overview
CAMPAIGN_CSV_COLUMNS.telemetryEvent     // Individual events
CAMPAIGN_CSV_COLUMNS.techniqueBreakdown // By technique
CAMPAIGN_CSV_COLUMNS.providerBreakdown  // By provider
CAMPAIGN_CSV_COLUMNS.timeSeries         // Time-bucketed data
CAMPAIGN_CSV_COLUMNS.campaignComparison // Comparison data
```

### ZIP Archive Export

Bulk exports generate a ZIP archive containing:

```
campaign-export-2024-01-15-103000/
├── charts/
│   ├── success-rate-chart.png
│   ├── technique-effectiveness.svg
│   └── latency-distribution.png
├── data/
│   ├── telemetry-events.csv
│   └── technique-breakdown.csv
└── metadata.json
```

**Metadata JSON:**
```json
{
  "export_date": "2024-01-15T10:30:00Z",
  "campaign_id": "550e8400-e29b-41d4-a716-446655440000",
  "campaign_name": "GPT-4 Jailbreak Benchmark",
  "charts_exported": 3,
  "data_files_exported": 2,
  "total_events": 100,
  "export_format": "zip",
  "export_version": "1.0.0"
}
```

---

## Usage Examples

### Basic Dashboard Integration

```tsx
// pages/dashboard/analytics/page.tsx
import { CampaignAnalyticsDashboard } from '@/components/campaign-analytics';

export default function AnalyticsPage() {
  return (
    <main className="container mx-auto p-6">
      <h1 className="text-3xl font-bold mb-6">Campaign Analytics</h1>
      <CampaignAnalyticsDashboard />
    </main>
  );
}
```

### Pre-selected Campaign with Filters

```tsx
import {
  CampaignAnalyticsDashboard,
  createDefaultFilterState
} from '@/components/campaign-analytics';

function AnalyticsWithFilters({ campaignId }: { campaignId: string }) {
  const [filters, setFilters] = useState({
    ...createDefaultFilterState(),
    techniques: ['dan_persona', 'cognitive_hacking'],
    dateRange: {
      start: subDays(new Date(), 7),
      end: new Date()
    }
  });

  return (
    <CampaignAnalyticsDashboard
      initialCampaignId={campaignId}
      defaultTab="charts"
      initialFilters={filters}
    />
  );
}
```

### Campaign Comparison Widget

```tsx
import {
  CampaignComparisonSelector,
  CampaignComparisonPanel
} from '@/components/campaign-analytics';

function ComparisonWidget() {
  const [selectedIds, setSelectedIds] = useState<string[]>([]);

  return (
    <Card>
      <CardHeader>
        <CardTitle>Compare Campaigns</CardTitle>
        <CampaignComparisonSelector
          selectedIds={selectedIds}
          onChange={setSelectedIds}
        />
      </CardHeader>
      <CardContent>
        {selectedIds.length >= 2 && (
          <CompactComparisonPanel
            selectedCampaignIds={selectedIds}
          />
        )}
      </CardContent>
    </Card>
  );
}
```

### Custom Statistics Display

```tsx
import {
  useCampaignStatistics,
  StatisticsCard,
  DistributionStatCard
} from '@/components/campaign-analytics';

function CustomStatsDisplay({ campaignId }: { campaignId: string }) {
  const { data: stats, isLoading } = useCampaignStatistics(campaignId);

  if (isLoading) return <StatisticsCardSkeleton />;

  return (
    <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
      <DistributionStatCard
        label="Success Rate"
        stats={stats.success_rate}
        format="percentage"
        color="success"
      />
      <DistributionStatCard
        label="Latency"
        stats={stats.latency_ms}
        format="duration"
      />
      <StatisticsCard
        label="Total Attempts"
        value={stats.attempts.total}
        format="compact"
      />
      <StatisticsCard
        label="Total Cost"
        value={stats.cost.total}
        format="currency"
      />
    </div>
  );
}
```

### Chart Export Integration

```tsx
import { useRef } from 'react';
import {
  SuccessRateTimeSeriesChart,
  ChartExportButton,
  exportChartAsPNG
} from '@/components/campaign-analytics';

function ExportableChart({ campaignId }: { campaignId: string }) {
  const chartRef = useRef<HTMLDivElement>(null);

  const handleCustomExport = async () => {
    if (!chartRef.current) return;

    const result = await exportChartAsPNG(chartRef, {
      scale: 4,       // 4x resolution for print
      backgroundColor: '#ffffff',
      padding: 20
    });

    if (result.success && result.blob) {
      // Custom handling - upload to server, etc.
      await uploadChart(result.blob);
    }
  };

  return (
    <Card>
      <CardHeader className="flex flex-row justify-between">
        <CardTitle>Success Rate Over Time</CardTitle>
        <div className="flex gap-2">
          <ChartExportButton chartRef={chartRef} />
          <Button onClick={handleCustomExport}>
            Upload to Report
          </Button>
        </div>
      </CardHeader>
      <CardContent>
        <div ref={chartRef}>
          <SuccessRateTimeSeriesChart campaignId={campaignId} />
        </div>
      </CardContent>
    </Card>
  );
}
```

### Telemetry Event Drill-Down

```tsx
import { useState } from 'react';
import {
  TelemetryTable,
  TelemetryDetailModal,
  FilterBar,
  createDefaultFilterState,
  filterStateToParams
} from '@/components/campaign-analytics';

function TelemetryExplorer({ campaignId }: { campaignId: string }) {
  const [filters, setFilters] = useState(createDefaultFilterState());
  const [selectedEventId, setSelectedEventId] = useState<string | null>(null);

  return (
    <div className="space-y-4">
      <FilterBar
        campaignId={campaignId}
        value={filters}
        onChange={setFilters}
      />

      <TelemetryTable
        campaignId={campaignId}
        filters={filterStateToParams(filters)}
        onRowClick={(event) => setSelectedEventId(event.id)}
      />

      <TelemetryDetailModal
        campaignId={campaignId}
        eventId={selectedEventId}
        open={!!selectedEventId}
        onOpenChange={(open) => !open && setSelectedEventId(null)}
      />
    </div>
  );
}
```

### Using TanStack Query Hooks Directly

```tsx
import {
  useCampaigns,
  useCampaignStatistics,
  useCampaignTimeSeries,
  useCampaignComparison,
  campaignQueryKeys
} from '@/lib/api/query';

function AnalyticsData({ campaignId }: { campaignId: string }) {
  // List campaigns with filtering
  const { data: campaigns } = useCampaigns({
    page: 1,
    page_size: 20,
    status: ['completed'],
    sort_by: 'created_at',
    sort_order: 'desc'
  });

  // Get statistics
  const { data: stats } = useCampaignStatistics(campaignId, {
    enabled: !!campaignId
  });

  // Get time series data
  const { data: timeSeries } = useCampaignTimeSeries(campaignId, {
    metric: 'success_rate',
    granularity: 'hour'
  });

  // Compare multiple campaigns
  const { data: comparison } = useCampaignComparison(
    [campaignId, 'other-campaign-id'],
    { include_time_series: true }
  );

  // Invalidate cache
  const queryClient = useQueryClient();
  const invalidate = () => {
    queryClient.invalidateQueries({
      queryKey: campaignQueryKeys.detail(campaignId)
    });
  };

  return (/* ... */);
}
```

---

## Configuration

### Backend Configuration

The analytics service uses TTL-based caching for expensive computations.

```python
# Default cache TTL settings (in seconds)
CACHE_TTL = {
    "statistics": 300,      # 5 minutes
    "time_series": 180,     # 3 minutes
    "breakdown": 300,       # 5 minutes
    "comparison": 300,      # 5 minutes
    "summary": 600,         # 10 minutes
}
```

### Frontend Configuration

TanStack Query cache settings:

```typescript
// Query stale times (milliseconds)
const STALE_TIMES = {
  campaigns: 30_000,      // 30 seconds
  statistics: 60_000,     // 1 minute
  timeSeries: 30_000,     // 30 seconds
  events: 15_000,         // 15 seconds
};

// Query cache times (milliseconds)
const CACHE_TIMES = {
  campaigns: 5 * 60_000,   // 5 minutes
  statistics: 10 * 60_000, // 10 minutes
  comparison: 5 * 60_000,  // 5 minutes
};
```

### Chart Configuration

Default chart dimensions and styling:

```typescript
const CHART_DEFAULTS = {
  height: 400,
  margin: { top: 20, right: 30, left: 60, bottom: 60 },
  colors: {
    success: '#22c55e',
    failure: '#ef4444',
    partial: '#f59e0b',
    primary: '#3b82f6',
    secondary: '#8b5cf6',
  },
  animation: {
    duration: 300,
    easing: 'ease-in-out'
  }
};
```

---

## Troubleshooting

### Common Issues

#### Charts Not Rendering

**Symptoms:** Chart area is blank or shows loading indefinitely.

**Solutions:**
1. Ensure Recharts is installed: `npm install recharts`
2. Check that campaign has telemetry events
3. Verify TanStack Query Provider is wrapped around the app
4. Check browser console for errors

#### Export Button Disabled

**Symptoms:** Export button is grayed out or shows "No data".

**Solutions:**
1. Ensure chart has rendered before exporting
2. Pass valid `chartRef` for chart exports
3. Pass non-empty `data` array for CSV exports
4. Check that campaign has at least one telemetry event

#### Comparison Shows No Delta

**Symptoms:** Comparison table shows metrics but no delta values.

**Solutions:**
1. Ensure at least 2 campaigns are selected
2. Verify campaigns have overlapping metrics
3. Check that `normalize_metrics: true` is set in comparison request

#### Slow Performance with Large Campaigns

**Symptoms:** Dashboard takes long to load or becomes unresponsive.

**Solutions:**
1. Use pagination for telemetry events (`page_size: 50`)
2. Use time range filters to limit data
3. Enable server-side caching (check cache stats endpoint)
4. Use compact chart variants for dashboards

### Error States

All components include proper error handling with retry capabilities:

```tsx
// Error state example
<TelemetryTableError
  error={new Error('Failed to fetch events')}
  onRetry={() => refetch()}
/>
```

### Debug Mode

Enable debug logging for TanStack Query:

```typescript
// In your app provider
const queryClient = new QueryClient({
  logger: {
    log: console.log,
    warn: console.warn,
    error: console.error,
  }
});
```

---

## Related Documentation

- [API Documentation](./API_DOCUMENTATION.md) - Full API reference
- [Frontend Architecture](./FRONTEND_ARCHITECTURE.md) - Component patterns
- [Data Pipeline Architecture](./DATA_PIPELINE_ARCHITECTURE.md) - ETL and analytics
- [Developer Guide](./DEVELOPER_GUIDE.md) - Development setup

---

## Changelog

### v1.0.0 (2026-01-11)

**Initial Release**

- Backend: SQLAlchemy models, Pydantic schemas, REST API endpoints
- Frontend: 80+ React components with TypeScript
- Charts: 5 Recharts-based visualizations
- Comparison: 2-4 campaign side-by-side comparison
- Export: PNG/SVG charts, CSV data, ZIP archives
- Filtering: Technique, provider, date range, status filters
- Caching: TTL-based LRU cache for expensive computations
- Testing: 100+ unit and integration tests
