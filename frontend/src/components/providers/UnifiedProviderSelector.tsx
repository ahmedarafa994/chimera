/**
 * UnifiedProviderSelector Component
 *
 * A comprehensive provider and model selection component with:
 * - Database-backed session preferences
 * - Real-time WebSocket synchronization
 * - Three-tier selection hierarchy (Request → Session → Global)
 * - Health status monitoring
 * - Optimistic UI updates
 *
 * @module UnifiedProviderSelector
 */

"use client";

import React, { useState, useEffect, useMemo } from "react";
import { cn } from "@/lib/utils";
import { Search, Loader2, WifiOff, Wifi, AlertCircle, CheckCircle2 } from "lucide-react";
import {
  useUnifiedProviders,
  useUnifiedModels,
  useCurrentSelection,
  useSaveSelection,
  useModelSelectionSync,
  type UnifiedProvider,
  type UnifiedModel,
  type SelectionScope,
} from "@/hooks";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Alert, AlertDescription } from "@/components/ui/alert";

// ============================================================================
// Type Definitions
// ============================================================================

export interface UnifiedProviderSelectorProps {
  /**
   * Optional session ID for session-scoped selection
   */
  sessionId?: string;

  /**
   * Callback when selection changes
   */
  onSelectionChange?: (provider: string, model: string, scope: SelectionScope) => void;

  /**
   * Additional CSS classes
   */
  className?: string;

  /**
   * Whether to show session information and scope badges
   * @default true
   */
  showSessionInfo?: boolean;

  /**
   * Whether to show WebSocket connection status
   * @default true
   */
  showConnectionStatus?: boolean;
}

// ============================================================================
// Helper Components
// ============================================================================

/**
 * Badge component for selection scope
 */
function ScopeBadge({ scope }: { scope: SelectionScope | string | undefined }) {
  const variants: Record<string, { variant: "default" | "secondary" | "outline"; label: string }> = {
    REQUEST: { variant: "default", label: "Request Override" },
    SESSION: { variant: "secondary", label: "Session Preference" },
    GLOBAL: { variant: "outline", label: "Global Default" },
    // Handle lowercase variants from backend
    request: { variant: "default", label: "Request Override" },
    session: { variant: "secondary", label: "Session Preference" },
    global: { variant: "outline", label: "Global Default" },
  };

  // Handle undefined or unknown scope
  const normalizedScope = scope?.toUpperCase?.() || 'GLOBAL';
  const config = variants[normalizedScope] || variants[scope as string] || { variant: "outline" as const, label: scope || "Unknown" };

  return (
    <Badge variant={config.variant} className="text-xs">
      {config.label}
    </Badge>
  );
}

/**
 * Badge component for provider health status
 */
function HealthBadge({ isAvailable, healthScore }: { isAvailable: boolean; healthScore?: number }) {
  if (!isAvailable) {
    return (
      <Badge variant="destructive" className="text-xs">
        <AlertCircle className="mr-1 h-3 w-3" />
        Unavailable
      </Badge>
    );
  }

  if (healthScore !== undefined) {
    if (healthScore >= 0.8) {
      return (
        <Badge variant="default" className="bg-green-600 text-xs">
          <CheckCircle2 className="mr-1 h-3 w-3" />
          Healthy
        </Badge>
      );
    }
    if (healthScore >= 0.5) {
      return (
        <Badge variant="secondary" className="bg-yellow-600 text-xs">
          <AlertCircle className="mr-1 h-3 w-3" />
          Degraded
        </Badge>
      );
    }
  }

  return (
    <Badge variant="outline" className="text-xs">
      Unknown
    </Badge>
  );
}

/**
 * Connection status indicator
 */
function ConnectionStatus({ isConnected, reconnect }: { isConnected: boolean; reconnect: () => void }) {
  if (isConnected) {
    return (
      <div className="flex items-center gap-2 text-xs text-green-600 dark:text-green-400">
        <Wifi className="h-4 w-4" />
        <span>Live sync active</span>
      </div>
    );
  }

  return (
    <div className="flex items-center gap-2">
      <div className="flex items-center gap-1 text-xs text-red-600 dark:text-red-400">
        <WifiOff className="h-4 w-4" />
        <span>Disconnected</span>
      </div>
      <Button variant="ghost" size="sm" onClick={reconnect} className="h-6 px-2 text-xs">
        Reconnect
      </Button>
    </div>
  );
}

// ============================================================================
// Main Component
// ============================================================================

export function UnifiedProviderSelector({
  sessionId,
  onSelectionChange,
  className,
  showSessionInfo = true,
  showConnectionStatus = true,
}: UnifiedProviderSelectorProps) {
  // ============================================================================
  // State
  // ============================================================================

  const [selectedProviderId, setSelectedProviderId] = useState<string | undefined>();
  const [providerSearchQuery, setProviderSearchQuery] = useState("");
  const [modelSearchQuery, setModelSearchQuery] = useState("");

  // ============================================================================
  // Data Fetching
  // ============================================================================

  const {
    data: providersData,
    isLoading: providersLoading,
    error: providersError,
  } = useUnifiedProviders();

  const {
    data: modelsData,
    isLoading: modelsLoading,
    error: modelsError,
  } = useUnifiedModels(selectedProviderId, {
    enabled: !!selectedProviderId,
  });

  const {
    data: currentSelection,
    isLoading: selectionLoading,
  } = useCurrentSelection(sessionId);

  // ============================================================================
  // Mutations
  // ============================================================================

  const saveMutation = useSaveSelection();

  // ============================================================================
  // WebSocket Synchronization
  // ============================================================================

  const { isConnected, reconnect } = useModelSelectionSync({
    sessionId,
    enabled: true,
    onSelectionChanged: (selection) => {
      // Handle real-time selection changes from other tabs/devices
      setSelectedProviderId(selection.provider_id);
      onSelectionChange?.(selection.provider_id, selection.model_id, selection.scope);
    },
  });

  // ============================================================================
  // Effects
  // ============================================================================

  // Initialize selected provider from current selection
  useEffect(() => {
    if (currentSelection && !selectedProviderId) {
      setSelectedProviderId(currentSelection.provider_id);
    }
  }, [currentSelection, selectedProviderId]);

  // ============================================================================
  // Computed Values
  // ============================================================================

  const providers = useMemo(() => providersData?.providers ?? [], [providersData]);
  const models = useMemo(() => modelsData?.models ?? [], [modelsData]);

  // Filter providers based on search query
  const filteredProviders = useMemo(() => {
    if (!providerSearchQuery.trim()) return providers;

    const query = providerSearchQuery.toLowerCase();
    return providers.filter(
      (provider) => {
        const name = provider.display_name || provider.name || '';
        const id = provider.provider_id || provider.id || '';
        return (
          name.toLowerCase().includes(query) ||
          id.toLowerCase().includes(query) ||
          provider.description?.toLowerCase().includes(query)
        );
      }
    );
  }, [providers, providerSearchQuery]);

  // Filter models based on search query
  const filteredModels = useMemo(() => {
    if (!modelSearchQuery.trim()) return models;

    const query = modelSearchQuery.toLowerCase();
    return models.filter(
      (model) =>
        model.name.toLowerCase().includes(query) ||
        model.id.toLowerCase().includes(query) ||
        model.description?.toLowerCase().includes(query)
    );
  }, [models, modelSearchQuery]);

  // ============================================================================
  // Event Handlers
  // ============================================================================

  /**
   * Handle provider selection
   */
  const handleProviderSelect = async (provider: UnifiedProvider) => {
    // Use provider_id (backend field name) or fallback to id
    const providerId = provider.provider_id?.trim() || provider.id?.trim();
    if (!providerId) {
      console.error("Provider has no valid identifier:", provider);
      return;
    }
    setSelectedProviderId(providerId);

    // If provider has a default model, select it
    if (provider.default_model) {
      await handleModelSelect(providerId, provider.default_model);
    }
  };

  /**
   * Handle model selection
   */
  const handleModelSelect = async (providerId: string, modelId: string) => {
    // Add validation
    if (!providerId?.trim() || !modelId?.trim()) {
      console.error("Invalid selection: provider or model ID is empty", { providerId, modelId });
      return;
    }

    try {
      const result = await saveMutation.mutateAsync({
        provider_id: providerId,
        model_id: modelId,
        session_id: sessionId,
      });

      // Call user callback with updated selection
      onSelectionChange?.(result.provider_id, result.model_id, result.scope);
    } catch (error) {
      console.error("Failed to save selection:", error);
    }
  };

  // ============================================================================
  // Render Helpers
  // ============================================================================

  /**
   * Render provider card
   */
  const renderProviderCard = (provider: UnifiedProvider, index: number) => {
    // Use provider_id (backend field name) or fallback to id
    const providerId = provider.provider_id?.trim() || provider.id?.trim() || '';
    const displayName = provider.display_name || provider.name || providerId || 'Unknown Provider';
    const isSelected = selectedProviderId === providerId;
    const isCurrent = currentSelection?.provider_id === providerId;

    return (
      <Card
        key={`provider-card-${index}`}
        className={cn(
          "cursor-pointer transition-all hover:shadow-md",
          isSelected && "ring-2 ring-primary",
          isCurrent && !isSelected && "border-primary/50"
        )}
        onClick={() => handleProviderSelect(provider)}
      >
        <CardHeader className="pb-3">
          <div className="flex items-start justify-between">
            <CardTitle className="text-base">{displayName}</CardTitle>
            <HealthBadge isAvailable={provider.is_available} healthScore={provider.health_score} />
          </div>
          {provider.description && (
            <CardDescription className="text-xs">{provider.description}</CardDescription>
          )}
        </CardHeader>
        <CardContent className="pt-0">
          <div className="text-xs text-muted-foreground">
            {provider.models_count ?? 0} model(s) available
          </div>
        </CardContent>
      </Card>
    );
  };

  /**
   * Render model card
   */
  const renderModelCard = (model: UnifiedModel, index: number) => {
    const modelId = model.id?.trim() || '';
    const isCurrent = currentSelection?.model_id === modelId;

    return (
      <Card
        key={`model-card-${index}`}
        className={cn(
          "cursor-pointer transition-all hover:shadow-md",
          isCurrent && "ring-2 ring-primary",
          !model.is_available && "opacity-50 cursor-not-allowed"
        )}
        onClick={() => {
          if (model.is_available && selectedProviderId) {
            handleModelSelect(selectedProviderId, model.id);
          }
        }}
      >
        <CardHeader className="pb-3">
          <div className="flex items-start justify-between">
            <CardTitle className="text-sm">{model.name}</CardTitle>
            {!model.is_available && (
              <Badge variant="destructive" className="text-xs">
                Unavailable
              </Badge>
            )}
          </div>
          {model.description && (
            <CardDescription className="text-xs line-clamp-2">{model.description}</CardDescription>
          )}
        </CardHeader>
        {(model.context_window || model.max_tokens || model.supports_streaming !== undefined) && (
          <CardContent className="pt-0">
            <div className="flex flex-wrap gap-2 text-xs text-muted-foreground">
              {model.context_window && (
                <span>Context: {model.context_window.toLocaleString()}</span>
              )}
              {model.max_tokens && (
                <span>Max tokens: {model.max_tokens.toLocaleString()}</span>
              )}
              {model.supports_streaming && (
                <Badge variant="outline" className="text-xs">
                  Streaming
                </Badge>
              )}
            </div>
          </CardContent>
        )}
      </Card>
    );
  };

  // ============================================================================
  // Render
  // ============================================================================

  return (
    <div className={cn("flex flex-col gap-6", className)}>
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex flex-col gap-1">
          <h3 className="text-lg font-semibold">Select AI Provider & Model</h3>
          {showSessionInfo && currentSelection && (
            <div className="flex items-center gap-2">
              <span className="text-sm text-muted-foreground">
                Current: {currentSelection.provider_id} / {currentSelection.model_id}
              </span>
              <ScopeBadge scope={currentSelection.scope} />
            </div>
          )}
        </div>
        {showConnectionStatus && (
          <ConnectionStatus isConnected={isConnected} reconnect={reconnect} />
        )}
      </div>

      {/* Error Display */}
      {providersError && (
        <Alert variant="destructive">
          <AlertCircle className="h-4 w-4" />
          <AlertDescription>
            Failed to load providers: {providersError.message}
          </AlertDescription>
        </Alert>
      )}

      {modelsError && (
        <Alert variant="destructive">
          <AlertCircle className="h-4 w-4" />
          <AlertDescription>
            Failed to load models: {modelsError.message}
          </AlertDescription>
        </Alert>
      )}

      {/* Two-Panel Layout */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Providers Panel */}
        <div className="flex flex-col gap-4">
          <div className="flex items-center justify-between">
            <h4 className="text-sm font-medium">Providers</h4>
            {providersLoading && <Loader2 className="h-4 w-4 animate-spin" />}
          </div>

          {/* Provider Search */}
          <div className="relative">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
            <Input
              placeholder="Search providers..."
              value={providerSearchQuery}
              onChange={(e) => setProviderSearchQuery(e.target.value)}
              className="pl-9"
            />
          </div>

          {/* Provider List */}
          <div className="flex flex-col gap-3 max-h-[600px] overflow-y-auto">
            {filteredProviders.length === 0 ? (
              <div className="text-sm text-muted-foreground text-center py-8">
                {providerSearchQuery ? "No providers found" : "No providers available"}
              </div>
            ) : (
              filteredProviders.map((provider, index) => renderProviderCard(provider, index))
            )}
          </div>
        </div>

        {/* Models Panel */}
        <div className="flex flex-col gap-4">
          <div className="flex items-center justify-between">
            <h4 className="text-sm font-medium">
              Models {selectedProviderId && `(${selectedProviderId})`}
            </h4>
            {modelsLoading && <Loader2 className="h-4 w-4 animate-spin" />}
          </div>

          {/* Model Search */}
          <div className="relative">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
            <Input
              placeholder="Search models..."
              value={modelSearchQuery}
              onChange={(e) => setModelSearchQuery(e.target.value)}
              className="pl-9"
              disabled={!selectedProviderId}
            />
          </div>

          {/* Model List */}
          <div className="flex flex-col gap-3 max-h-[600px] overflow-y-auto">
            {!selectedProviderId ? (
              <div className="text-sm text-muted-foreground text-center py-8">
                Select a provider to view models
              </div>
            ) : filteredModels.length === 0 ? (
              <div className="text-sm text-muted-foreground text-center py-8">
                {modelSearchQuery ? "No models found" : "No models available for this provider"}
              </div>
            ) : (
              filteredModels.map((model, index) => renderModelCard(model, index))
            )}
          </div>
        </div>
      </div>

      {/* Loading Overlay */}
      {saveMutation.isPending && (
        <div className="fixed inset-0 bg-black/20 flex items-center justify-center z-50">
          <Card className="p-6">
            <div className="flex items-center gap-3">
              <Loader2 className="h-5 w-5 animate-spin" />
              <span className="text-sm">Saving selection...</span>
            </div>
          </Card>
        </div>
      )}
    </div>
  );
}
