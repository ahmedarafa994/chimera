/**
 * @deprecated This file is maintained for backwards compatibility only.
 * New code should use api-enhanced.ts and api-config.ts instead.
 * 
 * This module re-exports from api-enhanced.ts to provide a migration path
 * for existing code that imports from api.ts.
 */

import { apiClient } from "./api-enhanced";
import { getActiveApiUrl, getCurrentApiUrl } from "./api-config";
import { FuzzRequest, FuzzResponse, HealthCheckResponse } from "@/types/schemas";

// Re-export the enhanced API client for backwards compatibility
export { apiClient };

// Get the active API URL from centralized config instead of hardcoded value
const API_BASE_URL = getActiveApiUrl();

// Strictly typed API methods using enhanced client
export const api = {
  health: {
    check: () => apiClient.get<HealthCheckResponse>("/health"),
  },
  gptfuzz: {
    run: (data: FuzzRequest) => apiClient.post<FuzzResponse>("/gptfuzz/run", data),
    status: (sessionId: string) => apiClient.get<any>(`/gptfuzz/status/${sessionId}`),
  },
};

/**
 * @deprecated Use getActiveApiUrl() from api-config.ts instead.
 * This function is provided for backwards compatibility.
 */
export function getLegacyApiBaseUrl(): string {
  return getCurrentApiUrl();
}