/**
 * AutoDAN TanStack Query Hooks
 * Type-safe queries for AutoDAN-Turbo operations
 */

import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { queryKeys, STALE_TIMES } from "./query-client";
import { apiClient } from "../core/client";
import type {
  StrategyListResponse,
  StrategySearchRequest,
  StrategySearchResponse,
  StrategyCreateRequest,
  JailbreakStrategy,
  LibraryStatsResponse,
  ProgressResponse,
  HealthResponse,
  ExportedLibrary,
  ImportLibraryResponse,
  DeleteStrategyResponse,
  AttackRequest,
  AttackResponse,
  WarmupRequest,
  WarmupResponse,
  LifelongLoopRequest,
  LifelongLoopResponse,
} from "../../types/autodan-turbo-types";

// ============================================================================
// API Functions
// ============================================================================

async function fetchStrategies(offset = 0, limit = 100): Promise<StrategyListResponse> {
  return apiClient.get<StrategyListResponse>(
    `/autodan-turbo/strategies?offset=${offset}&limit=${limit}`
  );
}

async function searchStrategies(request: StrategySearchRequest): Promise<StrategySearchResponse> {
  return apiClient.post<StrategySearchResponse>("/autodan-turbo/strategies/search", request);
}

async function createStrategy(request: StrategyCreateRequest): Promise<JailbreakStrategy> {
  return apiClient.post<JailbreakStrategy>("/autodan-turbo/strategies", request);
}

async function deleteStrategy(strategyId: string): Promise<DeleteStrategyResponse> {
  return apiClient.delete<DeleteStrategyResponse>(
    `/autodan-turbo/strategies/${encodeURIComponent(strategyId)}`
  );
}

async function fetchLibraryStats(): Promise<LibraryStatsResponse> {
  return apiClient.get<LibraryStatsResponse>("/autodan-turbo/library/stats");
}

async function exportLibrary(): Promise<ExportedLibrary> {
  return apiClient.get<ExportedLibrary>("/autodan-turbo/library/export");
}

async function importLibrary(library: ExportedLibrary, merge = true): Promise<ImportLibraryResponse> {
  return apiClient.post<ImportLibraryResponse>("/autodan-turbo/library/import", {
    data: library,
    merge,
  });
}

async function fetchProgress(): Promise<ProgressResponse> {
  return apiClient.get<ProgressResponse>("/autodan-turbo/progress");
}

async function fetchHealth(): Promise<HealthResponse> {
  return apiClient.get<HealthResponse>("/autodan-turbo/health");
}

async function executeAttack(request: AttackRequest): Promise<AttackResponse> {
  return apiClient.post<AttackResponse>("/autodan-turbo/attack", request);
}

async function runWarmup(request: WarmupRequest): Promise<WarmupResponse> {
  return apiClient.post<WarmupResponse>("/autodan-turbo/warmup", request, {
    timeout: 600000, // 10 minutes
  });
}

async function runLifelong(request: LifelongLoopRequest): Promise<LifelongLoopResponse> {
  return apiClient.post<LifelongLoopResponse>("/autodan-turbo/lifelong", request, {
    timeout: 600000, // 10 minutes
  });
}

async function resetEngine(): Promise<{ success: boolean; message: string }> {
  return apiClient.post<{ success: boolean; message: string }>("/autodan-turbo/reset");
}

async function saveLibrary(): Promise<{ success: boolean; path: string }> {
  return apiClient.post<{ success: boolean; path: string }>("/autodan-turbo/library/save");
}

async function clearLibrary(): Promise<{ success: boolean; message: string }> {
  return apiClient.post<{ success: boolean; message: string }>("/autodan-turbo/library/clear");
}

// ============================================================================
// Query Hooks
// ============================================================================

/**
 * Fetch strategy list with pagination
 */
export function useStrategies(offset = 0, limit = 100) {
  return useQuery({
    queryKey: [...queryKeys.autodan.strategies(), { offset, limit }],
    queryFn: () => fetchStrategies(offset, limit),
    staleTime: STALE_TIMES.SEMI_DYNAMIC,
  });
}

/**
 * Search strategies
 */
export function useStrategySearch(query: string, enabled = true) {
  return useQuery({
    queryKey: [...queryKeys.autodan.strategies(), "search", query],
    queryFn: () => searchStrategies({ query }),
    staleTime: STALE_TIMES.SEMI_DYNAMIC,
    enabled: enabled && query.length > 0,
  });
}

/**
 * Fetch library statistics
 */
export function useLibraryStats() {
  return useQuery({
    queryKey: queryKeys.autodan.libraryStats(),
    queryFn: fetchLibraryStats,
    staleTime: STALE_TIMES.SEMI_DYNAMIC,
  });
}

/**
 * Fetch current progress
 */
export function useProgress(enabled = true) {
  return useQuery({
    queryKey: queryKeys.autodan.progress(),
    queryFn: fetchProgress,
    staleTime: STALE_TIMES.REALTIME,
    refetchInterval: enabled ? 2000 : false, // Poll every 2 seconds when enabled
    enabled,
  });
}

/**
 * Fetch engine health status
 */
export function useAutodanHealth() {
  return useQuery({
    queryKey: queryKeys.autodan.health(),
    queryFn: fetchHealth,
    staleTime: STALE_TIMES.DYNAMIC,
    refetchInterval: 30000, // Refetch every 30 seconds
  });
}

// ============================================================================
// Mutation Hooks
// ============================================================================

/**
 * Create a new strategy
 */
export function useCreateStrategy() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: createStrategy,
    onSuccess: () => {
      // Invalidate strategies list
      queryClient.invalidateQueries({ queryKey: queryKeys.autodan.strategies() });
      queryClient.invalidateQueries({ queryKey: queryKeys.autodan.libraryStats() });
    },
  });
}

/**
 * Delete a strategy
 */
export function useDeleteStrategy() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: deleteStrategy,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: queryKeys.autodan.strategies() });
      queryClient.invalidateQueries({ queryKey: queryKeys.autodan.libraryStats() });
    },
  });
}

/**
 * Execute a single attack
 */
export function useAttack() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: executeAttack,
    onSuccess: () => {
      // Refresh progress after attack
      queryClient.invalidateQueries({ queryKey: queryKeys.autodan.progress() });
    },
  });
}

/**
 * Run warmup exploration
 */
export function useWarmup() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: runWarmup,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: queryKeys.autodan.strategies() });
      queryClient.invalidateQueries({ queryKey: queryKeys.autodan.libraryStats() });
      queryClient.invalidateQueries({ queryKey: queryKeys.autodan.progress() });
    },
  });
}

/**
 * Run lifelong learning loop
 */
export function useLifelong() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: runLifelong,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: queryKeys.autodan.strategies() });
      queryClient.invalidateQueries({ queryKey: queryKeys.autodan.libraryStats() });
      queryClient.invalidateQueries({ queryKey: queryKeys.autodan.progress() });
    },
  });
}

/**
 * Export library
 */
export function useExportLibrary() {
  return useMutation({
    mutationFn: exportLibrary,
  });
}

/**
 * Import library
 */
export function useImportLibrary() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({ library, merge }: { library: ExportedLibrary; merge?: boolean }) =>
      importLibrary(library, merge),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: queryKeys.autodan.strategies() });
      queryClient.invalidateQueries({ queryKey: queryKeys.autodan.libraryStats() });
    },
  });
}

/**
 * Save library to disk
 */
export function useSaveLibrary() {
  return useMutation({
    mutationFn: saveLibrary,
  });
}

/**
 * Clear library
 */
export function useClearLibrary() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: clearLibrary,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: queryKeys.autodan.strategies() });
      queryClient.invalidateQueries({ queryKey: queryKeys.autodan.libraryStats() });
    },
  });
}

/**
 * Reset engine
 */
export function useResetEngine() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: resetEngine,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: queryKeys.autodan.all });
    },
  });
}