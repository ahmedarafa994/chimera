/**
 * Admin API Types for Project Chimera Frontend
 * 
 * Types for admin endpoints including:
 * - Feature flag management
 * - Tenant management
 * - Usage analytics
 */

// =============================================================================
// Feature Flag Types
// =============================================================================

export interface TechniqueConfig {
  name: string;
  enabled: boolean;
  risk_level: "low" | "medium" | "high" | "critical" | "unknown";
  requires_approval: boolean;
  description: string;
}

export interface TechniqueListResponse {
  techniques: TechniqueConfig[];
  total_count: number;
}

export interface FeatureFlagStatsResponse {
  total_techniques: number;
  enabled_count: number;
  disabled_count: number;
  risk_distribution: Record<string, number>;
  approval_required_count: number;
  plugin_enabled: boolean;
}

export interface TechniqueToggleRequest {
  technique_name: string;
  enabled: boolean;
}

export interface TechniqueToggleResponse {
  success: boolean;
  technique_name: string;
  enabled: boolean;
  message: string;
}

export interface FeatureFlagReloadResponse {
  success: boolean;
  message: string;
  enabled_techniques: number;
}

export interface TechniqueDetailResponse {
  name: string;
  enabled: boolean;
  risk_level: string;
  requires_approval: boolean;
  description: string;
  [key: string]: unknown; // Additional config fields
}

// =============================================================================
// Tenant Types
// =============================================================================

export type TenantTier = "free" | "basic" | "professional" | "enterprise";

export interface CreateTenantRequest {
  tenant_id: string;
  name: string;
  tier?: TenantTier;
  api_key?: string;
}

export interface UpdateTenantRequest {
  name?: string;
  tier?: TenantTier;
  is_active?: boolean;
  rate_limit_per_minute?: number;
  monthly_quota?: number;
}

export interface TenantResponse {
  tenant_id: string;
  name: string;
  tier: TenantTier;
  rate_limit_per_minute: number;
  monthly_quota: number;
  is_active: boolean;
}

export interface TenantDetailResponse extends TenantResponse {
  allowed_techniques: string[];
  blocked_techniques: string[];
  custom_settings: Record<string, unknown>;
  created_at: string;
}

export interface TenantListResponse {
  tenants: TenantResponse[];
  total_count: number;
}

export interface TenantStatsResponse {
  total_tenants: number;
  active_tenants: number;
  tenants_by_tier: Record<TenantTier, number>;
  total_monthly_requests: number;
}

export interface DeleteTenantResponse {
  success: boolean;
  message: string;
}

// =============================================================================
// Usage Analytics Types
// =============================================================================

export interface GlobalUsageResponse {
  total_requests: number;
  total_tokens: number;
  total_errors: number;
  requests_by_endpoint: Record<string, number>;
  requests_by_technique: Record<string, number>;
  cache_hit_rate: number;
  avg_duration_ms: number;
  period_start: string;
  period_end: string;
}

export interface TenantUsageResponse {
  tenant_id: string;
  period_start: string;
  period_end: string;
  total_requests: number;
  requests_by_endpoint: Record<string, number>;
  requests_by_technique: Record<string, number>;
  total_tokens: number;
  total_errors: number;
  cache_hit_rate: string;
  avg_duration_ms: string;
}

export interface TopTechniquesResponse {
  techniques: Array<{
    name: string;
    count: number;
  }>;
  tenant_filter: string | null;
}

export interface QuotaStatusResponse {
  tenant_id: string;
  monthly_quota: number;
  current_usage: number;
  remaining: number | "unlimited";
  within_quota: boolean;
  quota_percentage: number;
}

// =============================================================================
// Admin Authentication
// =============================================================================

export interface AdminAuthConfig {
  apiKey: string;
}

export function createAdminAuthHeaders(config: AdminAuthConfig): Record<string, string> {
  return {
    Authorization: `Bearer ${config.apiKey}`,
  };
}

/**
 * Alias for createAdminAuthHeaders for backward compatibility
 */
export const getAdminAuthHeaders = createAdminAuthHeaders;

// =============================================================================
// Additional Types for Admin Operations
// =============================================================================

/**
 * Request to update technique configuration
 */
export interface TechniqueUpdateRequest {
  enabled?: boolean;
  risk_level?: "low" | "medium" | "high" | "critical" | "unknown";
  requires_approval?: boolean;
  description?: string;
  [key: string]: unknown;
}

/**
 * Query parameters for usage analytics
 */
export interface UsageQueryParams {
  start_date?: string;
  end_date?: string;
  granularity?: "hour" | "day" | "week" | "month";
  include_breakdown?: boolean;
}