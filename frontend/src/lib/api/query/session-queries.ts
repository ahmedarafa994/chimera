/**
 * Session TanStack Query Hooks
 * Type-safe queries for session management
 */

import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { queryKeys, STALE_TIMES } from "./query-client";
import { apiClient } from "../core/client";
import type {
  SessionInfoResponse,
  SessionStatsResponse,
  CreateSessionResponse,
} from "../../api-enhanced";

// ============================================================================
// Types
// ============================================================================

export interface CreateSessionRequest {
  provider?: string;
  model?: string;
}

export interface UpdateSessionModelRequest {
  provider?: string;
  model: string;
}

export interface UpdateSessionModelResponse {
  success: boolean;
  message?: string;
  reverted_to_default?: boolean;
  provider?: string;
  model?: string;
}

// ============================================================================
// API Functions
// ============================================================================

async function fetchSession(sessionId: string): Promise<SessionInfoResponse | null> {
  try {
    return await apiClient.get<SessionInfoResponse>(`/session/${sessionId}`);
  } catch {
    return null;
  }
}

async function createSession(options?: CreateSessionRequest): Promise<CreateSessionResponse> {
  return apiClient.post<CreateSessionResponse>("/session", options || {});
}

async function updateSessionModel(
  sessionId: string,
  request: UpdateSessionModelRequest
): Promise<UpdateSessionModelResponse> {
  return apiClient.put<UpdateSessionModelResponse>("/session/model", request, {
    headers: {
      "X-Session-ID": sessionId,
    },
  });
}

async function fetchSessionStats(): Promise<SessionStatsResponse> {
  return apiClient.get<SessionStatsResponse>("/session/stats");
}

async function deleteSession(sessionId: string): Promise<{ success: boolean }> {
  return apiClient.delete<{ success: boolean }>(`/session/${sessionId}`);
}

// ============================================================================
// Query Hooks
// ============================================================================

/**
 * Fetch session by ID
 */
export function useSession(sessionId: string | null, enabled = true) {
  return useQuery({
    queryKey: queryKeys.sessions.detail(sessionId || ""),
    queryFn: () => fetchSession(sessionId!),
    staleTime: STALE_TIMES.SEMI_DYNAMIC,
    enabled: enabled && !!sessionId,
  });
}

/**
 * Fetch session statistics (admin)
 */
export function useSessionStats(enabled = true) {
  return useQuery({
    queryKey: queryKeys.sessions.stats(),
    queryFn: fetchSessionStats,
    staleTime: STALE_TIMES.DYNAMIC,
    enabled,
  });
}

// ============================================================================
// Mutation Hooks
// ============================================================================

/**
 * Create a new session
 */
export function useCreateSession() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: createSession,
    onSuccess: (data) => {
      // Cache the new session
      if (data.session_id) {
        queryClient.setQueryData(queryKeys.sessions.detail(data.session_id), {
          session_id: data.session_id,
          provider: data.provider,
          model: data.model,
          created_at: new Date().toISOString(),
          last_activity: new Date().toISOString(),
          request_count: 0,
        });
        // Store as current session
        queryClient.setQueryData(queryKeys.sessions.current(), data.session_id);
      }
      // Invalidate stats
      queryClient.invalidateQueries({ queryKey: queryKeys.sessions.stats() });
    },
  });
}

/**
 * Update session model
 */
export function useUpdateSessionModel(sessionId: string) {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (request: UpdateSessionModelRequest) => updateSessionModel(sessionId, request),
    onSuccess: (data) => {
      // Update session cache
      queryClient.setQueryData(
        queryKeys.sessions.detail(sessionId),
        (old: SessionInfoResponse | undefined) => {
          if (!old) return old;
          return {
            ...old,
            provider: data.provider || old.provider,
            model: data.model || old.model,
          };
        }
      );
      // Also invalidate providers since active model may have changed
      queryClient.invalidateQueries({ queryKey: queryKeys.providers.active() });
    },
  });
}

/**
 * Delete session
 */
export function useDeleteSession() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: deleteSession,
    onSuccess: (_, sessionId) => {
      // Remove from cache
      queryClient.removeQueries({ queryKey: queryKeys.sessions.detail(sessionId) });
      // Invalidate stats
      queryClient.invalidateQueries({ queryKey: queryKeys.sessions.stats() });
    },
  });
}

// ============================================================================
// Session Management Utilities
// ============================================================================

const SESSION_STORAGE_KEY = "chimera_session_id";

/**
 * Get stored session ID from localStorage
 */
export function getStoredSessionId(): string | null {
  if (typeof window === "undefined") return null;
  return localStorage.getItem(SESSION_STORAGE_KEY);
}

/**
 * Store session ID in localStorage
 */
export function storeSessionId(sessionId: string): void {
  if (typeof window === "undefined") return;
  localStorage.setItem(SESSION_STORAGE_KEY, sessionId);
}

/**
 * Clear stored session ID
 */
export function clearStoredSessionId(): void {
  if (typeof window === "undefined") return;
  localStorage.removeItem(SESSION_STORAGE_KEY);
}

/**
 * Hook to manage current session with persistence
 */
export function useCurrentSession() {
  const queryClient = useQueryClient();
  const storedId = getStoredSessionId();
  
  const sessionQuery = useSession(storedId);
  const createSessionMutation = useCreateSession();

  const initializeSession = async (options?: CreateSessionRequest) => {
    // Check if we have a valid stored session
    if (storedId && sessionQuery.data) {
      return sessionQuery.data;
    }

    // Create new session
    const result = await createSessionMutation.mutateAsync(options);
    if (result.session_id) {
      storeSessionId(result.session_id);
    }
    return result;
  };

  const clearSession = () => {
    if (storedId) {
      queryClient.removeQueries({ queryKey: queryKeys.sessions.detail(storedId) });
    }
    clearStoredSessionId();
  };

  return {
    sessionId: storedId,
    session: sessionQuery.data,
    isLoading: sessionQuery.isLoading || createSessionMutation.isPending,
    error: sessionQuery.error || createSessionMutation.error,
    initializeSession,
    clearSession,
    refetch: sessionQuery.refetch,
  };
}