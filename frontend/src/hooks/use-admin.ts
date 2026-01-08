"use client";

/**
 * Admin Hooks for Project Chimera Frontend
 * 
 * React hooks for admin functionality including:
 * - Feature flag management
 * - Tenant management
 * - Usage analytics
 */

import { useState, useCallback, useEffect } from "react";
import { adminService } from "@/lib/services/admin-service";
import {
  TechniqueConfig,
  TechniqueListResponse,
  TenantResponse,
  TenantListResponse,
  TenantDetailResponse,
  GlobalUsageResponse,
  TenantUsageResponse,
  AdminAuthConfig,
  CreateTenantRequest,
  UpdateTenantRequest,
  TechniqueToggleRequest,
  TechniqueUpdateRequest,
  UsageQueryParams,
} from "@/lib/types/admin-types";

// =============================================================================
// Auth Hook
// =============================================================================

export interface UseAdminAuthReturn {
  authConfig: AdminAuthConfig | null;
  isAuthenticated: boolean;
  isLoading: boolean;
  error: string | null;
  setApiKey: (apiKey: string) => void;
  clearAuth: () => void;
  login: (apiKey: string) => Promise<void>;
}

export function useAdminAuth(): UseAdminAuthReturn {
  const [authConfig, setAuthConfig] = useState<AdminAuthConfig | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const setApiKey = useCallback((apiKey: string) => {
    setAuthConfig({ apiKey });
    // Optionally store in sessionStorage for persistence
    if (typeof window !== "undefined") {
      sessionStorage.setItem("chimera_admin_key", apiKey);
    }
  }, []);

  const clearAuth = useCallback(() => {
    setAuthConfig(null);
    setError(null);
    if (typeof window !== "undefined") {
      sessionStorage.removeItem("chimera_admin_key");
    }
  }, []);

  const login = useCallback(async (apiKey: string) => {
    setIsLoading(true);
    setError(null);
    
    try {
      // Validate the API key by attempting to use it
      // For now, we just set it - actual validation would happen on first API call
      if (!apiKey || apiKey.trim().length === 0) {
        throw new Error("API key is required");
      }
      
      setAuthConfig({ apiKey: apiKey.trim() });
      if (typeof window !== "undefined") {
        sessionStorage.setItem("chimera_admin_key", apiKey.trim());
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Login failed");
      throw err;
    } finally {
      setIsLoading(false);
    }
  }, []);

  // Load from sessionStorage on mount
  useEffect(() => {
    if (typeof window !== "undefined") {
      const storedKey = sessionStorage.getItem("chimera_admin_key");
      if (storedKey) {
        setAuthConfig({ apiKey: storedKey });
      }
      setIsLoading(false);
    } else {
      setIsLoading(false);
    }
  }, []);

  return {
    authConfig,
    isAuthenticated: !!authConfig?.apiKey,
    isLoading,
    error,
    setApiKey,
    clearAuth,
    login,
  };
}

// =============================================================================
// Feature Flags Hook
// =============================================================================

export interface UseFeatureFlagsReturn {
  techniques: TechniqueConfig[];
  isLoading: boolean;
  error: string | null;
  refresh: () => Promise<void>;
  toggleTechnique: (name: string, enabled: boolean) => Promise<void>;
  updateTechnique: (name: string, updates: TechniqueUpdateRequest) => Promise<void>;
  resetTechnique: (name: string) => Promise<void>;
}

export function useFeatureFlags(authConfig: AdminAuthConfig | null): UseFeatureFlagsReturn {
  const [techniques, setTechniques] = useState<TechniqueConfig[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    if (!authConfig) {
      setError("Not authenticated");
      return;
    }

    setIsLoading(true);
    setError(null);

    try {
      const response = await adminService.listTechniqueFlags(authConfig);
      setTechniques(response.techniques);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load techniques");
    } finally {
      setIsLoading(false);
    }
  }, [authConfig]);

  const toggleTechnique = useCallback(async (name: string, enabled: boolean) => {
    if (!authConfig) {
      setError("Not authenticated");
      return;
    }

    try {
      await adminService.toggleTechnique({ technique_name: name, enabled }, authConfig);
      await refresh();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to toggle technique");
      throw err;
    }
  }, [authConfig, refresh]);

  const updateTechnique = useCallback(async (name: string, updates: TechniqueUpdateRequest) => {
    if (!authConfig) {
      setError("Not authenticated");
      return;
    }

    try {
      await adminService.updateTechniqueConfig(name, updates, authConfig);
      await refresh();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to update technique");
      throw err;
    }
  }, [authConfig, refresh]);

  const resetTechnique = useCallback(async () => {
    if (!authConfig) {
      setError("Not authenticated");
      return;
    }

    try {
      await adminService.resetTechniqueConfig(authConfig);
      await refresh();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to reset technique");
      throw err;
    }
  }, [authConfig, refresh]);

  // Load on mount when authenticated
  useEffect(() => {
    if (authConfig) {
      refresh();
    }
  }, [authConfig, refresh]);

  return {
    techniques,
    isLoading,
    error,
    refresh,
    toggleTechnique,
    updateTechnique,
    resetTechnique,
  };
}

// =============================================================================
// Tenants Hook
// =============================================================================

export interface UseTenantsReturn {
  tenants: TenantResponse[];
  totalCount: number;
  isLoading: boolean;
  error: string | null;
  refresh: () => Promise<void>;
  createTenant: (request: CreateTenantRequest) => Promise<TenantResponse>;
  updateTenant: (id: string, request: UpdateTenantRequest) => Promise<TenantResponse>;
  deleteTenant: (id: string) => Promise<void>;
  activateTenant: (id: string) => Promise<void>;
  deactivateTenant: (id: string) => Promise<void>;
  getTenantDetails: (id: string) => Promise<TenantDetailResponse>;
}

export function useTenants(authConfig: AdminAuthConfig | null): UseTenantsReturn {
  const [tenants, setTenants] = useState<TenantResponse[]>([]);
  const [totalCount, setTotalCount] = useState(0);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    if (!authConfig) {
      setError("Not authenticated");
      return;
    }

    setIsLoading(true);
    setError(null);

    try {
      const response = await adminService.listTenants(authConfig);
      setTenants(response.tenants);
      setTotalCount(response.total_count);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load tenants");
    } finally {
      setIsLoading(false);
    }
  }, [authConfig]);

  const createTenant = useCallback(async (request: CreateTenantRequest): Promise<TenantResponse> => {
    if (!authConfig) {
      throw new Error("Not authenticated");
    }

    const tenant = await adminService.createTenant(request, authConfig);
    await refresh();
    return tenant;
  }, [authConfig, refresh]);

  const updateTenant = useCallback(async (id: string, request: UpdateTenantRequest): Promise<TenantResponse> => {
    if (!authConfig) {
      throw new Error("Not authenticated");
    }

    const tenant = await adminService.updateTenant(id, request, authConfig);
    await refresh();
    return tenant;
  }, [authConfig, refresh]);

  const deleteTenant = useCallback(async (id: string): Promise<void> => {
    if (!authConfig) {
      throw new Error("Not authenticated");
    }

    await adminService.deleteTenant(id, authConfig);
    await refresh();
  }, [authConfig, refresh]);

  const activateTenant = useCallback(async (id: string): Promise<void> => {
    if (!authConfig) {
      throw new Error("Not authenticated");
    }

    await adminService.activateTenant(id, authConfig);
    await refresh();
  }, [authConfig, refresh]);

  const deactivateTenant = useCallback(async (id: string): Promise<void> => {
    if (!authConfig) {
      throw new Error("Not authenticated");
    }

    await adminService.deactivateTenant(id, authConfig);
    await refresh();
  }, [authConfig, refresh]);

  const getTenantDetails = useCallback(async (id: string): Promise<TenantDetailResponse> => {
    if (!authConfig) {
      throw new Error("Not authenticated");
    }

    return adminService.getTenant(id, authConfig);
  }, [authConfig]);

  // Load on mount when authenticated
  useEffect(() => {
    if (authConfig) {
      refresh();
    }
  }, [authConfig, refresh]);

  return {
    tenants,
    totalCount,
    isLoading,
    error,
    refresh,
    createTenant,
    updateTenant,
    deleteTenant,
    activateTenant,
    deactivateTenant,
    getTenantDetails,
  };
}

// =============================================================================
// Usage Analytics Hook
// =============================================================================

export interface UseUsageAnalyticsReturn {
  globalUsage: GlobalUsageResponse | null;
  tenantUsage: TenantUsageResponse | null;
  isLoading: boolean;
  error: string | null;
  fetchGlobalUsage: (params?: UsageQueryParams) => Promise<void>;
  fetchTenantUsage: (tenantId: string, params?: UsageQueryParams) => Promise<void>;
}

export function useUsageAnalytics(authConfig: AdminAuthConfig | null): UseUsageAnalyticsReturn {
  const [globalUsage, setGlobalUsage] = useState<GlobalUsageResponse | null>(null);
  const [tenantUsage, setTenantUsage] = useState<TenantUsageResponse | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchGlobalUsage = useCallback(async (params?: UsageQueryParams) => {
    if (!authConfig) {
      setError("Not authenticated");
      return;
    }

    setIsLoading(true);
    setError(null);

    try {
      const response = await adminService.getGlobalUsage(authConfig, params);
      setGlobalUsage(response);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load global usage");
    } finally {
      setIsLoading(false);
    }
  }, [authConfig]);

  const fetchTenantUsage = useCallback(async (tenantId: string, params?: UsageQueryParams) => {
    if (!authConfig) {
      setError("Not authenticated");
      return;
    }

    setIsLoading(true);
    setError(null);

    try {
      const response = await adminService.getTenantUsage(tenantId, authConfig, params);
      setTenantUsage(response);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load tenant usage");
    } finally {
      setIsLoading(false);
    }
  }, [authConfig]);

  return {
    globalUsage,
    tenantUsage,
    isLoading,
    error,
    fetchGlobalUsage,
    fetchTenantUsage,
  };
}