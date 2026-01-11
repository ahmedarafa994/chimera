/**
 * Campaign Analytics TanStack Query Hooks
 * Type-safe queries for campaign telemetry analytics
 */

import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { queryKeys, STALE_TIMES } from "./query-client";
import { apiClient } from "../core/client";
import type {
  Campaign,
  CampaignSummary,
  CampaignDetail,
  CampaignStatistics,
  CampaignListResponse,
  CampaignFilterParams,
  CampaignComparison,
  CampaignComparisonRequest,
  TelemetryTimeSeries,
  TelemetryListResponse,
  TelemetryEventDetail,
  TelemetryFilterParams,
  TechniqueBreakdown,
  ProviderBreakdown,
  PotencyBreakdown,
  ExportResponse,
  ExportChartOptions,
  TimeGranularity,
  CampaignCacheStats,
} from "../../../types/campaign-analytics";

// ============================================================================
// Query Key Factory Extensions
// ============================================================================

/**
 * Campaign-specific query keys for cache management
 */
export const campaignQueryKeys = {
  all: ["campaigns"] as const,
  lists: () => [...campaignQueryKeys.all, "list"] as const,
  list: (filters?: CampaignFilterParams, page?: number, pageSize?: number) =>
    [...campaignQueryKeys.lists(), { filters, page, pageSize }] as const,
  details: () => [...campaignQueryKeys.all, "detail"] as const,
  detail: (id: string) => [...campaignQueryKeys.details(), id] as const,
  summary: (id: string) => [...campaignQueryKeys.all, "summary", id] as const,
  statistics: (id: string) => [...campaignQueryKeys.all, "statistics", id] as const,
  timeSeries: (id: string, metric?: string, granularity?: TimeGranularity) =>
    [...campaignQueryKeys.all, "time-series", id, { metric, granularity }] as const,
  comparison: (ids: string[]) =>
    [...campaignQueryKeys.all, "comparison", ...ids.sort()] as const,
  breakdowns: (id: string) => [...campaignQueryKeys.all, "breakdown", id] as const,
  techniqueBreakdown: (id: string) =>
    [...campaignQueryKeys.breakdowns(id), "techniques"] as const,
  providerBreakdown: (id: string) =>
    [...campaignQueryKeys.breakdowns(id), "providers"] as const,
  potencyBreakdown: (id: string) =>
    [...campaignQueryKeys.breakdowns(id), "potency"] as const,
  events: (id: string, page?: number, pageSize?: number, filters?: TelemetryFilterParams) =>
    [...campaignQueryKeys.all, "events", id, { page, pageSize, filters }] as const,
  eventDetail: (campaignId: string, eventId: string) =>
    [...campaignQueryKeys.all, "events", campaignId, "detail", eventId] as const,
  cacheStats: () => [...campaignQueryKeys.all, "cache", "stats"] as const,
} as const;

// ============================================================================
// Types
// ============================================================================

export interface CampaignListParams {
  page?: number;
  pageSize?: number;
  sortBy?: string;
  sortOrder?: "asc" | "desc";
  filters?: CampaignFilterParams | null;
}

export interface TimeSeriesParams {
  metric?: string;
  granularity?: TimeGranularity;
  startTime?: string | null;
  endTime?: string | null;
  techniqueSuite?: string[] | null;
  provider?: string[] | null;
}

export interface TelemetryEventListParams {
  page?: number;
  pageSize?: number;
  filters?: TelemetryFilterParams | null;
}

export interface ExportCSVParams {
  includePrompts?: boolean;
  includeResponses?: boolean;
}

// ============================================================================
// API Functions
// ============================================================================

async function fetchCampaigns(params: CampaignListParams = {}): Promise<CampaignListResponse> {
  const {
    page = 1,
    pageSize = 20,
    sortBy = "created_at",
    sortOrder = "desc",
    filters,
  } = params;

  const searchParams = new URLSearchParams({
    page: String(page),
    page_size: String(pageSize),
    sort_by: sortBy,
    sort_order: sortOrder,
  });

  // Add filter parameters
  if (filters) {
    if (filters.status) {
      filters.status.forEach((s) => searchParams.append("status", s));
    }
    if (filters.provider) {
      filters.provider.forEach((p) => searchParams.append("provider", p));
    }
    if (filters.technique_suite) {
      filters.technique_suite.forEach((t) => searchParams.append("technique_suite", t));
    }
    if (filters.tags) {
      filters.tags.forEach((t) => searchParams.append("tags", t));
    }
    if (filters.start_date) {
      searchParams.set("start_date", filters.start_date);
    }
    if (filters.end_date) {
      searchParams.set("end_date", filters.end_date);
    }
    if (filters.search) {
      searchParams.set("search", filters.search);
    }
    if (filters.min_attempts !== null && filters.min_attempts !== undefined) {
      searchParams.set("min_attempts", String(filters.min_attempts));
    }
    if (filters.min_success_rate !== null && filters.min_success_rate !== undefined) {
      searchParams.set("min_success_rate", String(filters.min_success_rate));
    }
  }

  return apiClient.get<CampaignListResponse>(`/campaigns?${searchParams.toString()}`);
}

async function fetchCampaign(campaignId: string): Promise<CampaignDetail> {
  return apiClient.get<CampaignDetail>(`/campaigns/${encodeURIComponent(campaignId)}`);
}

async function fetchCampaignSummary(campaignId: string): Promise<CampaignSummary> {
  return apiClient.get<CampaignSummary>(
    `/campaigns/${encodeURIComponent(campaignId)}/summary`
  );
}

async function fetchCampaignStatistics(campaignId: string): Promise<CampaignStatistics> {
  return apiClient.get<CampaignStatistics>(
    `/campaigns/${encodeURIComponent(campaignId)}/statistics`
  );
}

async function fetchTechniqueBreakdown(campaignId: string): Promise<TechniqueBreakdown> {
  return apiClient.get<TechniqueBreakdown>(
    `/campaigns/${encodeURIComponent(campaignId)}/breakdown/techniques`
  );
}

async function fetchProviderBreakdown(campaignId: string): Promise<ProviderBreakdown> {
  return apiClient.get<ProviderBreakdown>(
    `/campaigns/${encodeURIComponent(campaignId)}/breakdown/providers`
  );
}

async function fetchPotencyBreakdown(campaignId: string): Promise<PotencyBreakdown> {
  return apiClient.get<PotencyBreakdown>(
    `/campaigns/${encodeURIComponent(campaignId)}/breakdown/potency`
  );
}

async function fetchTimeSeries(
  campaignId: string,
  params: TimeSeriesParams = {}
): Promise<TelemetryTimeSeries> {
  const {
    metric = "success_rate",
    granularity = "hour",
    startTime,
    endTime,
    techniqueSuite,
    provider,
  } = params;

  const searchParams = new URLSearchParams({
    metric,
    granularity,
  });

  if (startTime) {
    searchParams.set("start_time", startTime);
  }
  if (endTime) {
    searchParams.set("end_time", endTime);
  }
  if (techniqueSuite) {
    techniqueSuite.forEach((t) => searchParams.append("technique_suite", t));
  }
  if (provider) {
    provider.forEach((p) => searchParams.append("provider", p));
  }

  return apiClient.get<TelemetryTimeSeries>(
    `/campaigns/${encodeURIComponent(campaignId)}/time-series?${searchParams.toString()}`
  );
}

async function compareCampaigns(
  request: CampaignComparisonRequest
): Promise<CampaignComparison> {
  return apiClient.post<CampaignComparison>("/campaigns/compare", request);
}

async function fetchTelemetryEvents(
  campaignId: string,
  params: TelemetryEventListParams = {}
): Promise<TelemetryListResponse> {
  const { page = 1, pageSize = 50, filters } = params;

  const searchParams = new URLSearchParams({
    page: String(page),
    page_size: String(pageSize),
  });

  if (filters) {
    if (filters.status) {
      filters.status.forEach((s) => searchParams.append("status", s));
    }
    if (filters.technique_suite) {
      filters.technique_suite.forEach((t) => searchParams.append("technique_suite", t));
    }
    if (filters.provider) {
      filters.provider.forEach((p) => searchParams.append("provider", p));
    }
    if (filters.model) {
      filters.model.forEach((m) => searchParams.append("model", m));
    }
    if (filters.success_only !== null && filters.success_only !== undefined) {
      searchParams.set("success_only", String(filters.success_only));
    }
    if (filters.start_time) {
      searchParams.set("start_time", filters.start_time);
    }
    if (filters.end_time) {
      searchParams.set("end_time", filters.end_time);
    }
    if (filters.min_potency !== null && filters.min_potency !== undefined) {
      searchParams.set("min_potency", String(filters.min_potency));
    }
    if (filters.max_potency !== null && filters.max_potency !== undefined) {
      searchParams.set("max_potency", String(filters.max_potency));
    }
  }

  return apiClient.get<TelemetryListResponse>(
    `/campaigns/${encodeURIComponent(campaignId)}/events?${searchParams.toString()}`
  );
}

async function fetchTelemetryEventDetail(
  campaignId: string,
  eventId: string
): Promise<TelemetryEventDetail> {
  return apiClient.get<TelemetryEventDetail>(
    `/campaigns/${encodeURIComponent(campaignId)}/events/${encodeURIComponent(eventId)}`
  );
}

async function exportChartRequest(
  campaignId: string,
  options: ExportChartOptions
): Promise<ExportResponse> {
  return apiClient.post<ExportResponse>(
    `/campaigns/${encodeURIComponent(campaignId)}/export/chart`,
    options
  );
}

async function invalidateCampaignCache(campaignId: string): Promise<{ success: boolean }> {
  return apiClient.delete<{ success: boolean }>(
    `/campaigns/${encodeURIComponent(campaignId)}/cache`
  );
}

async function fetchCacheStats(): Promise<CampaignCacheStats> {
  return apiClient.get<CampaignCacheStats>("/campaigns/cache/stats");
}

// ============================================================================
// Query Hooks
// ============================================================================

/**
 * Fetch paginated list of campaigns with optional filtering
 */
export function useCampaigns(params: CampaignListParams = {}, enabled = true) {
  const { page = 1, pageSize = 20, filters } = params;

  return useQuery({
    queryKey: campaignQueryKeys.list(filters ?? undefined, page, pageSize),
    queryFn: () => fetchCampaigns(params),
    staleTime: STALE_TIMES.SEMI_DYNAMIC,
    enabled,
  });
}

/**
 * Fetch a single campaign by ID
 */
export function useCampaign(campaignId: string | null, enabled = true) {
  return useQuery({
    queryKey: campaignQueryKeys.detail(campaignId || ""),
    queryFn: () => fetchCampaign(campaignId!),
    staleTime: STALE_TIMES.SEMI_DYNAMIC,
    enabled: enabled && !!campaignId,
  });
}

/**
 * Fetch campaign summary (lighter than full detail)
 */
export function useCampaignSummary(campaignId: string | null, enabled = true) {
  return useQuery({
    queryKey: campaignQueryKeys.summary(campaignId || ""),
    queryFn: () => fetchCampaignSummary(campaignId!),
    staleTime: STALE_TIMES.SEMI_DYNAMIC,
    enabled: enabled && !!campaignId,
  });
}

/**
 * Fetch comprehensive campaign statistics
 */
export function useCampaignStatistics(campaignId: string | null, enabled = true) {
  return useQuery({
    queryKey: campaignQueryKeys.statistics(campaignId || ""),
    queryFn: () => fetchCampaignStatistics(campaignId!),
    staleTime: STALE_TIMES.SEMI_DYNAMIC,
    enabled: enabled && !!campaignId,
  });
}

/**
 * Fetch technique breakdown for a campaign
 */
export function useTechniqueBreakdown(campaignId: string | null, enabled = true) {
  return useQuery({
    queryKey: campaignQueryKeys.techniqueBreakdown(campaignId || ""),
    queryFn: () => fetchTechniqueBreakdown(campaignId!),
    staleTime: STALE_TIMES.SEMI_DYNAMIC,
    enabled: enabled && !!campaignId,
  });
}

/**
 * Fetch provider breakdown for a campaign
 */
export function useProviderBreakdown(campaignId: string | null, enabled = true) {
  return useQuery({
    queryKey: campaignQueryKeys.providerBreakdown(campaignId || ""),
    queryFn: () => fetchProviderBreakdown(campaignId!),
    staleTime: STALE_TIMES.SEMI_DYNAMIC,
    enabled: enabled && !!campaignId,
  });
}

/**
 * Fetch potency breakdown for a campaign
 */
export function usePotencyBreakdown(campaignId: string | null, enabled = true) {
  return useQuery({
    queryKey: campaignQueryKeys.potencyBreakdown(campaignId || ""),
    queryFn: () => fetchPotencyBreakdown(campaignId!),
    staleTime: STALE_TIMES.SEMI_DYNAMIC,
    enabled: enabled && !!campaignId,
  });
}

/**
 * Fetch time series data for campaign visualization
 */
export function useCampaignTimeSeries(
  campaignId: string | null,
  options: TimeSeriesParams = {},
  enabled = true
) {
  const { metric = "success_rate", granularity = "hour" } = options;

  return useQuery({
    queryKey: campaignQueryKeys.timeSeries(
      campaignId || "",
      metric,
      granularity as TimeGranularity
    ),
    queryFn: () => fetchTimeSeries(campaignId!, options),
    staleTime: STALE_TIMES.SEMI_DYNAMIC,
    enabled: enabled && !!campaignId,
  });
}

/**
 * Compare multiple campaigns (2-4)
 */
export function useCampaignComparison(
  campaignIds: string[],
  options: Omit<CampaignComparisonRequest, "campaign_ids"> = {},
  enabled = true
) {
  const isValidComparison = campaignIds.length >= 2 && campaignIds.length <= 4;

  return useQuery({
    queryKey: campaignQueryKeys.comparison(campaignIds),
    queryFn: () =>
      compareCampaigns({
        campaign_ids: campaignIds,
        include_time_series: options.include_time_series ?? false,
        normalize_metrics: options.normalize_metrics ?? true,
        metrics: options.metrics,
      }),
    staleTime: STALE_TIMES.SEMI_DYNAMIC,
    enabled: enabled && isValidComparison,
  });
}

/**
 * Fetch paginated telemetry events for a campaign
 */
export function useTelemetryEvents(
  campaignId: string | null,
  params: TelemetryEventListParams = {},
  enabled = true
) {
  const { page = 1, pageSize = 50, filters } = params;

  return useQuery({
    queryKey: campaignQueryKeys.events(campaignId || "", page, pageSize, filters ?? undefined),
    queryFn: () => fetchTelemetryEvents(campaignId!, params),
    staleTime: STALE_TIMES.SEMI_DYNAMIC,
    enabled: enabled && !!campaignId,
  });
}

/**
 * Fetch detailed telemetry event
 */
export function useTelemetryEventDetail(
  campaignId: string | null,
  eventId: string | null,
  enabled = true
) {
  return useQuery({
    queryKey: campaignQueryKeys.eventDetail(campaignId || "", eventId || ""),
    queryFn: () => fetchTelemetryEventDetail(campaignId!, eventId!),
    staleTime: STALE_TIMES.SEMI_DYNAMIC,
    enabled: enabled && !!campaignId && !!eventId,
  });
}

/**
 * Fetch cache statistics
 */
export function useCampaignCacheStats(enabled = true) {
  return useQuery({
    queryKey: campaignQueryKeys.cacheStats(),
    queryFn: fetchCacheStats,
    staleTime: STALE_TIMES.DYNAMIC,
    enabled,
  });
}

// ============================================================================
// Mutation Hooks
// ============================================================================

/**
 * Export campaign chart (PNG/SVG)
 */
export function useCampaignExport(campaignId: string) {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (options: ExportChartOptions) => exportChartRequest(campaignId, options),
    onSuccess: () => {
      // Optionally invalidate any export-related caches
      queryClient.invalidateQueries({ queryKey: campaignQueryKeys.cacheStats() });
    },
  });
}

/**
 * Generate CSV export URL for campaign
 *
 * Note: CSV export is a direct download, not a mutation.
 * This utility generates the download URL.
 */
export function getCampaignCSVExportUrl(
  campaignId: string,
  params: ExportCSVParams = {}
): string {
  const { includePrompts = false, includeResponses = false } = params;

  const searchParams = new URLSearchParams({
    include_prompts: String(includePrompts),
    include_responses: String(includeResponses),
  });

  // Build URL relative to API base
  return `/campaigns/${encodeURIComponent(campaignId)}/export/csv?${searchParams.toString()}`;
}

/**
 * Invalidate campaign cache
 */
export function useInvalidateCampaignCache() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: invalidateCampaignCache,
    onSuccess: (_, campaignId) => {
      // Invalidate all queries related to this campaign
      queryClient.invalidateQueries({ queryKey: campaignQueryKeys.detail(campaignId) });
      queryClient.invalidateQueries({ queryKey: campaignQueryKeys.statistics(campaignId) });
      queryClient.invalidateQueries({ queryKey: campaignQueryKeys.breakdowns(campaignId) });
      queryClient.invalidateQueries({ queryKey: campaignQueryKeys.cacheStats() });
    },
  });
}

// ============================================================================
// Prefetch Utilities
// ============================================================================

/**
 * Prefetch campaign data for faster navigation
 */
export function usePrefetchCampaign() {
  const queryClient = useQueryClient();

  return async (campaignId: string) => {
    await Promise.all([
      queryClient.prefetchQuery({
        queryKey: campaignQueryKeys.detail(campaignId),
        queryFn: () => fetchCampaign(campaignId),
        staleTime: STALE_TIMES.SEMI_DYNAMIC,
      }),
      queryClient.prefetchQuery({
        queryKey: campaignQueryKeys.statistics(campaignId),
        queryFn: () => fetchCampaignStatistics(campaignId),
        staleTime: STALE_TIMES.SEMI_DYNAMIC,
      }),
    ]);
  };
}

/**
 * Prefetch campaign list for faster initial load
 */
export function usePrefetchCampaignList() {
  const queryClient = useQueryClient();

  return async (params: CampaignListParams = {}) => {
    await queryClient.prefetchQuery({
      queryKey: campaignQueryKeys.list(
        params.filters ?? undefined,
        params.page ?? 1,
        params.pageSize ?? 20
      ),
      queryFn: () => fetchCampaigns(params),
      staleTime: STALE_TIMES.SEMI_DYNAMIC,
    });
  };
}

// ============================================================================
// Cache Invalidation Utilities
// ============================================================================

/**
 * Hook to invalidate all campaign-related caches
 */
export function useInvalidateAllCampaigns() {
  const queryClient = useQueryClient();

  return () => {
    queryClient.invalidateQueries({ queryKey: campaignQueryKeys.all });
  };
}

/**
 * Hook to invalidate a specific campaign's caches
 */
export function useInvalidateCampaign() {
  const queryClient = useQueryClient();

  return (campaignId: string) => {
    queryClient.invalidateQueries({ queryKey: campaignQueryKeys.detail(campaignId) });
    queryClient.invalidateQueries({ queryKey: campaignQueryKeys.summary(campaignId) });
    queryClient.invalidateQueries({ queryKey: campaignQueryKeys.statistics(campaignId) });
    queryClient.invalidateQueries({ queryKey: campaignQueryKeys.breakdowns(campaignId) });
    queryClient.invalidateQueries({ queryKey: [...campaignQueryKeys.all, "time-series", campaignId] });
    queryClient.invalidateQueries({ queryKey: [...campaignQueryKeys.all, "events", campaignId] });
  };
}
