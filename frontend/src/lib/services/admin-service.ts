/**
 * Admin Service for Project Chimera Frontend
 * 
 * Provides API methods for admin functionality including:
 * - Feature flag management
 * - Tenant management
 * - Usage analytics
 */

import { enhancedApi } from "../api-enhanced";
import {
  TechniqueConfig,
  TechniqueListResponse,
  TechniqueToggleRequest,
  TechniqueToggleResponse,
  TechniqueUpdateRequest,
  CreateTenantRequest,
  TenantResponse,
  TenantDetailResponse,
  TenantListResponse,
  UpdateTenantRequest,
  GlobalUsageResponse,
  TenantUsageResponse,
  UsageQueryParams,
  AdminAuthConfig,
  getAdminAuthHeaders,
} from "../types/admin-types";

// =============================================================================
// Configuration
// =============================================================================

const ADMIN_BASE_PATH = "/api/v1/admin";

// =============================================================================
// Feature Flag Management
// =============================================================================

/**
 * List all technique feature flags
 */
export async function listTechniqueFlags(
  authConfig: AdminAuthConfig
): Promise<TechniqueListResponse> {
  const response = await enhancedApi.get<TechniqueListResponse>(
    `${ADMIN_BASE_PATH}/feature-flags`,
    {
      headers: getAdminAuthHeaders(authConfig),
    }
  );
  return response;
}

/**
 * Get a specific technique configuration
 */
export async function getTechniqueFlag(
  techniqueName: string,
  authConfig: AdminAuthConfig
): Promise<TechniqueConfig> {
  const response = await enhancedApi.get<TechniqueConfig>(
    `${ADMIN_BASE_PATH}/feature-flags/${encodeURIComponent(techniqueName)}`,
    {
      headers: getAdminAuthHeaders(authConfig),
    }
  );
  return response;
}

/**
 * Toggle a technique on/off
 */
export async function toggleTechnique(
  request: TechniqueToggleRequest,
  authConfig: AdminAuthConfig
): Promise<TechniqueToggleResponse> {
  const response = await enhancedApi.post<TechniqueToggleResponse>(
    `${ADMIN_BASE_PATH}/feature-flags/toggle`,
    request,
    {
      headers: getAdminAuthHeaders(authConfig),
    }
  );
  return response;
}

/**
 * Update technique configuration
 * Note: Backend doesn't have PUT endpoint for individual techniques,
 * use toggleTechnique for enabling/disabling
 */
export async function updateTechniqueConfig(
  techniqueName: string,
  request: TechniqueUpdateRequest,
  authConfig: AdminAuthConfig
): Promise<TechniqueConfig> {
  // Use toggle endpoint for enable/disable changes
  if ('enabled' in request) {
    await toggleTechnique(
      { technique_name: techniqueName, enabled: request.enabled ?? false },
      authConfig
    );
  }
  // Return the updated config
  return getTechniqueFlag(techniqueName, authConfig);
}

/**
 * Reset technique to default configuration (reload from disk)
 */
export async function resetTechniqueConfig(
  authConfig: AdminAuthConfig
): Promise<{ success: boolean; message: string }> {
  const response = await enhancedApi.post<{ success: boolean; message: string }>(
    `${ADMIN_BASE_PATH}/feature-flags/reload`,
    {},
    {
      headers: getAdminAuthHeaders(authConfig),
    }
  );
  return response;
}

// =============================================================================
// Tenant Management
// =============================================================================

/**
 * List all tenants
 */
export async function listTenants(
  authConfig: AdminAuthConfig,
  params?: { skip?: number; limit?: number; active_only?: boolean }
): Promise<TenantListResponse> {
  const queryParams = new URLSearchParams();
  if (params?.skip !== undefined) queryParams.set("skip", params.skip.toString());
  if (params?.limit !== undefined) queryParams.set("limit", params.limit.toString());
  if (params?.active_only !== undefined) queryParams.set("active_only", params.active_only.toString());
  
  const queryString = queryParams.toString();
  const url = `${ADMIN_BASE_PATH}/tenants${queryString ? `?${queryString}` : ""}`;
  
  const response = await enhancedApi.get<TenantListResponse>(url, {
    headers: getAdminAuthHeaders(authConfig),
  });
  return response;
}

/**
 * Get a specific tenant by ID
 */
export async function getTenant(
  tenantId: string,
  authConfig: AdminAuthConfig
): Promise<TenantDetailResponse> {
  const response = await enhancedApi.get<TenantDetailResponse>(
    `${ADMIN_BASE_PATH}/tenants/${encodeURIComponent(tenantId)}`,
    {
      headers: getAdminAuthHeaders(authConfig),
    }
  );
  return response;
}

/**
 * Create a new tenant
 */
export async function createTenant(
  request: CreateTenantRequest,
  authConfig: AdminAuthConfig
): Promise<TenantResponse> {
  const response = await enhancedApi.post<TenantResponse>(
    `${ADMIN_BASE_PATH}/tenants`,
    request,
    {
      headers: getAdminAuthHeaders(authConfig),
    }
  );
  return response;
}

/**
 * Update an existing tenant
 */
export async function updateTenant(
  tenantId: string,
  request: UpdateTenantRequest,
  authConfig: AdminAuthConfig
): Promise<TenantResponse> {
  const response = await enhancedApi.put<TenantResponse>(
    `${ADMIN_BASE_PATH}/tenants/${encodeURIComponent(tenantId)}`,
    request,
    {
      headers: getAdminAuthHeaders(authConfig),
    }
  );
  return response;
}

/**
 * Delete a tenant
 */
export async function deleteTenant(
  tenantId: string,
  authConfig: AdminAuthConfig
): Promise<void> {
  await enhancedApi.delete(
    `${ADMIN_BASE_PATH}/tenants/${encodeURIComponent(tenantId)}`,
    {
      headers: getAdminAuthHeaders(authConfig),
    }
  );
}

/**
 * Activate a tenant
 */
export async function activateTenant(
  tenantId: string,
  authConfig: AdminAuthConfig
): Promise<TenantResponse> {
  const response = await enhancedApi.post<TenantResponse>(
    `${ADMIN_BASE_PATH}/tenants/${encodeURIComponent(tenantId)}/activate`,
    {},
    {
      headers: getAdminAuthHeaders(authConfig),
    }
  );
  return response;
}

/**
 * Deactivate a tenant
 */
export async function deactivateTenant(
  tenantId: string,
  authConfig: AdminAuthConfig
): Promise<TenantResponse> {
  const response = await enhancedApi.post<TenantResponse>(
    `${ADMIN_BASE_PATH}/tenants/${encodeURIComponent(tenantId)}/deactivate`,
    {},
    {
      headers: getAdminAuthHeaders(authConfig),
    }
  );
  return response;
}

// =============================================================================
// Usage Analytics
// =============================================================================

/**
 * Get global usage statistics
 */
export async function getGlobalUsage(
  authConfig: AdminAuthConfig,
  params?: UsageQueryParams
): Promise<GlobalUsageResponse> {
  const queryParams = new URLSearchParams();
  if (params?.start_date) queryParams.set("start_date", params.start_date);
  if (params?.end_date) queryParams.set("end_date", params.end_date);
  if (params?.granularity) queryParams.set("granularity", params.granularity);
  if (params?.include_breakdown !== undefined) {
    queryParams.set("include_breakdown", params.include_breakdown.toString());
  }
  
  const queryString = queryParams.toString();
  const url = `${ADMIN_BASE_PATH}/usage/global${queryString ? `?${queryString}` : ""}`;
  
  const response = await enhancedApi.get<GlobalUsageResponse>(url, {
    headers: getAdminAuthHeaders(authConfig),
  });
  return response;
}

/**
 * Get usage statistics for a specific tenant
 */
export async function getTenantUsage(
  tenantId: string,
  authConfig: AdminAuthConfig,
  params?: UsageQueryParams
): Promise<TenantUsageResponse> {
  const queryParams = new URLSearchParams();
  if (params?.start_date) queryParams.set("start_date", params.start_date);
  if (params?.end_date) queryParams.set("end_date", params.end_date);
  if (params?.granularity) queryParams.set("granularity", params.granularity);
  if (params?.include_breakdown !== undefined) {
    queryParams.set("include_breakdown", params.include_breakdown.toString());
  }
  
  const queryString = queryParams.toString();
  const url = `${ADMIN_BASE_PATH}/usage/tenant/${encodeURIComponent(tenantId)}${queryString ? `?${queryString}` : ""}`;
  
  const response = await enhancedApi.get<TenantUsageResponse>(url, {
    headers: getAdminAuthHeaders(authConfig),
  });
  return response;
}

// =============================================================================
// Admin Service Object (for convenience)
// =============================================================================

export const adminService = {
  // Feature flags
  listTechniqueFlags,
  getTechniqueFlag,
  toggleTechnique,
  updateTechniqueConfig,
  resetTechniqueConfig,
  
  // Tenants
  listTenants,
  getTenant,
  createTenant,
  updateTenant,
  deleteTenant,
  activateTenant,
  deactivateTenant,
  
  // Usage
  getGlobalUsage,
  getTenantUsage,
};

export default adminService;