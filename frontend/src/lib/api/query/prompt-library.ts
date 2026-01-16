/**
 * Prompt Library TanStack Query Hooks
 * Type-safe queries for Prompt Library template management operations
 */

import { useQuery, useMutation, useQueryClient, useInfiniteQuery } from "@tanstack/react-query";
import { queryKeys, STALE_TIMES } from "./query-client";
import { apiClient } from "../core/client";
import type {
  // Core types
  PromptTemplate,
  TemplateVersion,
  TemplateRating,
  RatingStatistics,
  // Request types
  CreateTemplateRequest,
  UpdateTemplateRequest,
  SearchTemplatesRequest,
  RateTemplateRequest,
  UpdateRatingRequest,
  CreateVersionRequest,
  SaveFromCampaignRequest,
  // Response types
  TemplateResponse,
  TemplateListResponse,
  TemplateVersionResponse,
  TemplateVersionListResponse,
  RatingResponse,
  RatingListResponse,
  RatingStatisticsResponse,
  TopRatedTemplatesResponse,
  TemplateDeleteResponse,
  TemplateStatsResponse,
  VersionComparisonResponse,
} from "../../../types/prompt-library-types";

// ============================================================================
// Prompt Library Query Keys
// ============================================================================

/**
 * Query keys for prompt library operations
 * Added to the queryKeys factory in query-client.ts
 */
export const promptLibraryKeys = {
  all: ["prompt-library"] as const,
  templates: () => [...promptLibraryKeys.all, "templates"] as const,
  template: (id: string) => [...promptLibraryKeys.templates(), id] as const,
  templateVersions: (id: string) => [...promptLibraryKeys.template(id), "versions"] as const,
  templateVersion: (id: string, version: number) =>
    [...promptLibraryKeys.templateVersions(id), version] as const,
  templateRatings: (id: string) => [...promptLibraryKeys.template(id), "ratings"] as const,
  templateRatingStats: (id: string) =>
    [...promptLibraryKeys.template(id), "rating-stats"] as const,
  search: (params: SearchTemplatesRequest) =>
    [...promptLibraryKeys.templates(), "search", params] as const,
  topRated: (limit: number, minRatings: number) =>
    [...promptLibraryKeys.templates(), "top-rated", limit, minRatings] as const,
  stats: () => [...promptLibraryKeys.all, "stats"] as const,
};

// ============================================================================
// API Functions
// ============================================================================

const API_BASE = "/prompt-library";

// Template CRUD
async function fetchTemplates(
  page = 1,
  pageSize = 20,
  sharingLevel?: string,
  status?: string,
  createdBy?: string
): Promise<TemplateListResponse> {
  const params = new URLSearchParams();
  params.set("page", String(page));
  params.set("page_size", String(pageSize));
  if (sharingLevel) params.set("sharing_level", sharingLevel);
  if (status) params.set("status", status);
  if (createdBy) params.set("created_by", createdBy);

  return apiClient.get<TemplateListResponse>(
    `${API_BASE}/templates?${params.toString()}`
  );
}

async function fetchTemplate(templateId: string): Promise<TemplateResponse> {
  return apiClient.get<TemplateResponse>(
    `${API_BASE}/templates/${encodeURIComponent(templateId)}`
  );
}

async function createTemplate(
  request: CreateTemplateRequest
): Promise<TemplateResponse> {
  return apiClient.post<TemplateResponse>(`${API_BASE}/templates`, request);
}

async function updateTemplate(
  templateId: string,
  request: UpdateTemplateRequest
): Promise<TemplateResponse> {
  return apiClient.put<TemplateResponse>(
    `${API_BASE}/templates/${encodeURIComponent(templateId)}`,
    request
  );
}

async function deleteTemplate(templateId: string): Promise<TemplateDeleteResponse> {
  return apiClient.delete<TemplateDeleteResponse>(
    `${API_BASE}/templates/${encodeURIComponent(templateId)}`
  );
}

// Search
async function searchTemplates(
  request: SearchTemplatesRequest
): Promise<TemplateListResponse> {
  return apiClient.post<TemplateListResponse>(
    `${API_BASE}/templates/search`,
    request
  );
}

// Campaign integration
async function saveFromCampaign(
  request: SaveFromCampaignRequest
): Promise<TemplateResponse> {
  return apiClient.post<TemplateResponse>(
    `${API_BASE}/templates/save-from-campaign`,
    request
  );
}

// Top rated
async function fetchTopRatedTemplates(
  limit = 10,
  minRatings = 1
): Promise<TopRatedTemplatesResponse> {
  const params = new URLSearchParams();
  params.set("limit", String(limit));
  params.set("min_ratings", String(minRatings));

  return apiClient.get<TopRatedTemplatesResponse>(
    `${API_BASE}/templates/top-rated?${params.toString()}`
  );
}

// Statistics
async function fetchLibraryStats(): Promise<TemplateStatsResponse> {
  return apiClient.get<TemplateStatsResponse>(`${API_BASE}/templates/stats`);
}

// Ratings
async function rateTemplate(
  templateId: string,
  request: RateTemplateRequest
): Promise<RatingResponse> {
  return apiClient.post<RatingResponse>(
    `${API_BASE}/templates/${encodeURIComponent(templateId)}/rate`,
    request
  );
}

async function fetchTemplateRatings(
  templateId: string,
  page = 1,
  pageSize = 20
): Promise<RatingListResponse> {
  const params = new URLSearchParams();
  params.set("page", String(page));
  params.set("page_size", String(pageSize));

  return apiClient.get<RatingListResponse>(
    `${API_BASE}/templates/${encodeURIComponent(templateId)}/ratings?${params.toString()}`
  );
}

async function fetchRatingStatistics(
  templateId: string
): Promise<RatingStatisticsResponse> {
  return apiClient.get<RatingStatisticsResponse>(
    `${API_BASE}/templates/${encodeURIComponent(templateId)}/ratings/statistics`
  );
}

async function updateRating(
  templateId: string,
  request: UpdateRatingRequest
): Promise<RatingResponse> {
  return apiClient.put<RatingResponse>(
    `${API_BASE}/templates/${encodeURIComponent(templateId)}/ratings/my-rating`,
    request
  );
}

async function deleteRating(templateId: string): Promise<void> {
  return apiClient.delete<void>(
    `${API_BASE}/templates/${encodeURIComponent(templateId)}/ratings/my-rating`
  );
}

// Versions
async function fetchTemplateVersions(
  templateId: string,
  limit = 100
): Promise<TemplateVersionListResponse> {
  const params = new URLSearchParams();
  params.set("limit", String(limit));

  return apiClient.get<TemplateVersionListResponse>(
    `${API_BASE}/templates/${encodeURIComponent(templateId)}/versions?${params.toString()}`
  );
}

async function fetchTemplateVersion(
  templateId: string,
  versionNumber: number
): Promise<TemplateVersionResponse> {
  return apiClient.get<TemplateVersionResponse>(
    `${API_BASE}/templates/${encodeURIComponent(templateId)}/versions/${versionNumber}`
  );
}

async function createVersion(
  templateId: string,
  request: CreateVersionRequest
): Promise<TemplateVersionResponse> {
  return apiClient.post<TemplateVersionResponse>(
    `${API_BASE}/templates/${encodeURIComponent(templateId)}/versions`,
    request
  );
}

async function restoreVersion(
  templateId: string,
  versionNumber: number
): Promise<TemplateResponse> {
  return apiClient.post<TemplateResponse>(
    `${API_BASE}/templates/${encodeURIComponent(templateId)}/versions/${versionNumber}/restore`
  );
}

// ============================================================================
// Query Hooks
// ============================================================================

/**
 * Fetch paginated list of prompt templates
 */
export function usePromptTemplates(
  page = 1,
  pageSize = 20,
  options?: {
    sharingLevel?: string;
    status?: string;
    createdBy?: string;
    enabled?: boolean;
  }
) {
  return useQuery({
    queryKey: [
      ...promptLibraryKeys.templates(),
      { page, pageSize, ...options },
    ],
    queryFn: () =>
      fetchTemplates(
        page,
        pageSize,
        options?.sharingLevel,
        options?.status,
        options?.createdBy
      ),
    staleTime: STALE_TIMES.SEMI_DYNAMIC,
    enabled: options?.enabled ?? true,
  });
}

/**
 * Fetch a single prompt template by ID
 */
export function usePromptTemplate(templateId: string, enabled = true) {
  return useQuery({
    queryKey: promptLibraryKeys.template(templateId),
    queryFn: () => fetchTemplate(templateId),
    staleTime: STALE_TIMES.SEMI_DYNAMIC,
    enabled: enabled && !!templateId,
  });
}

/**
 * Search prompt templates with advanced filtering
 */
export function useSearchTemplates(
  request: SearchTemplatesRequest,
  enabled = true
) {
  return useQuery({
    queryKey: promptLibraryKeys.search(request),
    queryFn: () => searchTemplates(request),
    staleTime: STALE_TIMES.SEMI_DYNAMIC,
    enabled,
  });
}

/**
 * Infinite query for paginated search results
 */
export function useInfiniteSearchTemplates(
  baseRequest: Omit<SearchTemplatesRequest, "page">,
  pageSize = 20,
  enabled = true
) {
  return useInfiniteQuery({
    queryKey: [...promptLibraryKeys.templates(), "infinite", baseRequest],
    queryFn: async ({ pageParam = 1 }) => {
      return searchTemplates({
        ...baseRequest,
        page: pageParam,
        page_size: pageSize,
      });
    },
    getNextPageParam: (lastPage) => {
      if (lastPage.page < lastPage.total_pages) {
        return lastPage.page + 1;
      }
      return undefined;
    },
    initialPageParam: 1,
    staleTime: STALE_TIMES.SEMI_DYNAMIC,
    enabled,
  });
}

/**
 * Fetch top-rated templates
 */
export function useTopRatedTemplates(limit = 10, minRatings = 1) {
  return useQuery({
    queryKey: promptLibraryKeys.topRated(limit, minRatings),
    queryFn: () => fetchTopRatedTemplates(limit, minRatings),
    staleTime: STALE_TIMES.SEMI_DYNAMIC,
  });
}

/**
 * Fetch library statistics
 */
export function useLibraryStatistics() {
  return useQuery({
    queryKey: promptLibraryKeys.stats(),
    queryFn: fetchLibraryStats,
    staleTime: STALE_TIMES.SEMI_DYNAMIC,
  });
}

/**
 * Fetch versions for a template
 */
export function useTemplateVersions(templateId: string, limit = 100, enabled = true) {
  return useQuery({
    queryKey: promptLibraryKeys.templateVersions(templateId),
    queryFn: () => fetchTemplateVersions(templateId, limit),
    staleTime: STALE_TIMES.SEMI_DYNAMIC,
    enabled: enabled && !!templateId,
  });
}

/**
 * Fetch a specific version of a template
 */
export function useTemplateVersion(
  templateId: string,
  versionNumber: number,
  enabled = true
) {
  return useQuery({
    queryKey: promptLibraryKeys.templateVersion(templateId, versionNumber),
    queryFn: () => fetchTemplateVersion(templateId, versionNumber),
    staleTime: STALE_TIMES.SEMI_DYNAMIC,
    enabled: enabled && !!templateId && versionNumber > 0,
  });
}

/**
 * Fetch ratings for a template
 */
export function useTemplateRatings(
  templateId: string,
  page = 1,
  pageSize = 20,
  enabled = true
) {
  return useQuery({
    queryKey: [...promptLibraryKeys.templateRatings(templateId), { page, pageSize }],
    queryFn: () => fetchTemplateRatings(templateId, page, pageSize),
    staleTime: STALE_TIMES.SEMI_DYNAMIC,
    enabled: enabled && !!templateId,
  });
}

/**
 * Fetch rating statistics for a template
 */
export function useRatingStatistics(templateId: string, enabled = true) {
  return useQuery({
    queryKey: promptLibraryKeys.templateRatingStats(templateId),
    queryFn: () => fetchRatingStatistics(templateId),
    staleTime: STALE_TIMES.SEMI_DYNAMIC,
    enabled: enabled && !!templateId,
  });
}

// ============================================================================
// Mutation Hooks
// ============================================================================

/**
 * Create a new prompt template
 */
export function useCreateTemplate() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: createTemplate,
    onSuccess: (data) => {
      // Invalidate template list queries
      queryClient.invalidateQueries({ queryKey: promptLibraryKeys.templates() });
      // Invalidate stats
      queryClient.invalidateQueries({ queryKey: promptLibraryKeys.stats() });
      // Optionally cache the newly created template
      if (data.template) {
        queryClient.setQueryData(
          promptLibraryKeys.template(data.template.id),
          data
        );
      }
    },
  });
}

/**
 * Update an existing prompt template
 */
export function useUpdateTemplate() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({
      templateId,
      request,
    }: {
      templateId: string;
      request: UpdateTemplateRequest;
    }) => updateTemplate(templateId, request),
    onSuccess: (data, variables) => {
      // Invalidate template list queries
      queryClient.invalidateQueries({ queryKey: promptLibraryKeys.templates() });
      // Update the specific template cache
      queryClient.setQueryData(
        promptLibraryKeys.template(variables.templateId),
        data
      );
      // If content was updated and version created, invalidate versions
      if (variables.request.create_version) {
        queryClient.invalidateQueries({
          queryKey: promptLibraryKeys.templateVersions(variables.templateId),
        });
      }
    },
  });
}

/**
 * Delete a prompt template
 */
export function useDeleteTemplate() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: deleteTemplate,
    onSuccess: (data, templateId) => {
      // Invalidate template list queries
      queryClient.invalidateQueries({ queryKey: promptLibraryKeys.templates() });
      // Remove the specific template from cache
      queryClient.removeQueries({
        queryKey: promptLibraryKeys.template(templateId),
      });
      // Invalidate stats
      queryClient.invalidateQueries({ queryKey: promptLibraryKeys.stats() });
    },
  });
}

/**
 * Rate a template
 */
export function useRateTemplate() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({
      templateId,
      request,
    }: {
      templateId: string;
      request: RateTemplateRequest;
    }) => rateTemplate(templateId, request),
    onSuccess: (data, variables) => {
      // Invalidate template ratings
      queryClient.invalidateQueries({
        queryKey: promptLibraryKeys.templateRatings(variables.templateId),
      });
      // Invalidate rating statistics
      queryClient.invalidateQueries({
        queryKey: promptLibraryKeys.templateRatingStats(variables.templateId),
      });
      // Invalidate template to update rating_stats
      queryClient.invalidateQueries({
        queryKey: promptLibraryKeys.template(variables.templateId),
      });
      // Invalidate top rated
      queryClient.invalidateQueries({
        queryKey: [...promptLibraryKeys.templates(), "top-rated"],
      });
    },
  });
}

/**
 * Update user's rating for a template
 */
export function useUpdateRating() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({
      templateId,
      request,
    }: {
      templateId: string;
      request: UpdateRatingRequest;
    }) => updateRating(templateId, request),
    onSuccess: (data, variables) => {
      // Invalidate template ratings
      queryClient.invalidateQueries({
        queryKey: promptLibraryKeys.templateRatings(variables.templateId),
      });
      // Invalidate rating statistics
      queryClient.invalidateQueries({
        queryKey: promptLibraryKeys.templateRatingStats(variables.templateId),
      });
      // Invalidate template to update rating_stats
      queryClient.invalidateQueries({
        queryKey: promptLibraryKeys.template(variables.templateId),
      });
    },
  });
}

/**
 * Delete user's rating for a template
 */
export function useDeleteRating() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: deleteRating,
    onSuccess: (_, templateId) => {
      // Invalidate template ratings
      queryClient.invalidateQueries({
        queryKey: promptLibraryKeys.templateRatings(templateId),
      });
      // Invalidate rating statistics
      queryClient.invalidateQueries({
        queryKey: promptLibraryKeys.templateRatingStats(templateId),
      });
      // Invalidate template to update rating_stats
      queryClient.invalidateQueries({
        queryKey: promptLibraryKeys.template(templateId),
      });
    },
  });
}

/**
 * Create a new version of a template
 */
export function useCreateVersion() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({
      templateId,
      request,
    }: {
      templateId: string;
      request: CreateVersionRequest;
    }) => createVersion(templateId, request),
    onSuccess: (data, variables) => {
      // Invalidate versions list
      queryClient.invalidateQueries({
        queryKey: promptLibraryKeys.templateVersions(variables.templateId),
      });
      // Invalidate template to update current_version
      queryClient.invalidateQueries({
        queryKey: promptLibraryKeys.template(variables.templateId),
      });
    },
  });
}

/**
 * Restore a template to a previous version
 */
export function useRestoreVersion() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({
      templateId,
      versionNumber,
    }: {
      templateId: string;
      versionNumber: number;
    }) => restoreVersion(templateId, versionNumber),
    onSuccess: (data, variables) => {
      // Invalidate template
      queryClient.invalidateQueries({
        queryKey: promptLibraryKeys.template(variables.templateId),
      });
      // Invalidate versions list
      queryClient.invalidateQueries({
        queryKey: promptLibraryKeys.templateVersions(variables.templateId),
      });
      // Invalidate template list queries
      queryClient.invalidateQueries({ queryKey: promptLibraryKeys.templates() });
    },
  });
}

/**
 * Save a prompt from campaign execution to the library
 */
export function useSaveFromCampaign() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: saveFromCampaign,
    onSuccess: (data) => {
      // Invalidate template list queries
      queryClient.invalidateQueries({ queryKey: promptLibraryKeys.templates() });
      // Invalidate stats
      queryClient.invalidateQueries({ queryKey: promptLibraryKeys.stats() });
      // Cache the newly created template
      if (data.template) {
        queryClient.setQueryData(
          promptLibraryKeys.template(data.template.id),
          data
        );
      }
    },
  });
}

// ============================================================================
// Utility Hooks
// ============================================================================

/**
 * Prefetch template data
 */
export function usePrefetchTemplate() {
  const queryClient = useQueryClient();

  return (templateId: string) => {
    queryClient.prefetchQuery({
      queryKey: promptLibraryKeys.template(templateId),
      queryFn: () => fetchTemplate(templateId),
      staleTime: STALE_TIMES.SEMI_DYNAMIC,
    });
  };
}

/**
 * Prefetch template versions
 */
export function usePrefetchTemplateVersions() {
  const queryClient = useQueryClient();

  return (templateId: string) => {
    queryClient.prefetchQuery({
      queryKey: promptLibraryKeys.templateVersions(templateId),
      queryFn: () => fetchTemplateVersions(templateId),
      staleTime: STALE_TIMES.SEMI_DYNAMIC,
    });
  };
}

/**
 * Invalidate all prompt library queries
 */
export function useInvalidatePromptLibrary() {
  const queryClient = useQueryClient();

  return () => {
    queryClient.invalidateQueries({ queryKey: promptLibraryKeys.all });
  };
}
