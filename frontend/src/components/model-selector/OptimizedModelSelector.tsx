"use client";

/**
 * Optimized Model Selector Component
 * 
 * Performance optimizations:
 * - Uses selective subscriptions to prevent unnecessary re-renders
 * - Memoized callbacks and computed values
 * - Optimistic UI updates for instant feedback
 * - Separated loading states (initial vs syncing)
 */

import React, { memo, useCallback, useMemo, useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Skeleton } from "@/components/ui/skeleton";
import { Loader2, RefreshCw, Check, AlertCircle, Server, Cpu, Zap, Wifi, WifiOff } from "lucide-react";
import { cn } from "@/lib/utils";
import {
  useOptimizedModelSelection,
  useModelSelectionValue,
  useProvidersList,
  useModelsList,
  useModelSelectionLoading,
  useModelSelectionActions,
  type ProviderInfo,
} from "@/lib/stores/optimized-model-selection-store";

// ============================================================================
// Sub-components with selective subscriptions
// ============================================================================

// Provider selector - only re-renders when providers or selection changes
const ProviderSelect = memo(function ProviderSelect({
  selectedProvider,
  providers,
  onSelect,
  disabled,
  compact,
}: {
  selectedProvider: string | null;
  providers: ProviderInfo[];
  onSelect: (provider: string) => void;
  disabled: boolean;
  compact: boolean;
}) {
  return (
    <Select
      value={selectedProvider || ""}
      onValueChange={onSelect}
      disabled={disabled}
    >
      <SelectTrigger className={compact ? "w-[140px]" : undefined}>
        <SelectValue placeholder="Select provider" />
      </SelectTrigger>
      <SelectContent>
        {providers.map((provider) => (
          <SelectItem key={provider.provider} value={provider.provider}>
            <div className="flex items-center gap-2">
              <Server className="h-4 w-4" />
              <span>{provider.displayName || provider.provider}</span>
              {!compact && (
                <Badge
                  variant={provider.isHealthy ? "default" : "secondary"}
                  className="ml-2"
                >
                  {provider.isHealthy ? "active" : "inactive"}
                </Badge>
              )}
            </div>
          </SelectItem>
        ))}
      </SelectContent>
    </Select>
  );
});

// Model selector - only re-renders when models or selection changes
const ModelSelect = memo(function ModelSelect({
  selectedModel,
  models,
  onSelect,
  disabled,
  compact,
  defaultModel,
}: {
  selectedModel: string | null;
  models: string[];
  onSelect: (model: string) => void;
  disabled: boolean;
  compact: boolean;
  defaultModel?: string | null;
}) {
  return (
    <Select
      value={selectedModel || ""}
      onValueChange={onSelect}
      disabled={disabled}
    >
      <SelectTrigger className={compact ? "w-[180px]" : undefined}>
        <SelectValue placeholder="Select model" />
      </SelectTrigger>
      <SelectContent>
        {models.map((model) => (
          <SelectItem key={model} value={model}>
            <div className="flex items-center gap-2">
              <Cpu className="h-4 w-4" />
              <span>{model}</span>
              {model === defaultModel && !compact && (
                <Badge variant="outline" className="ml-2">
                  Default
                </Badge>
              )}
            </div>
          </SelectItem>
        ))}
      </SelectContent>
    </Select>
  );
});

// Sync indicator - only re-renders when sync state changes
const SyncIndicator = memo(function SyncIndicator({
  isSyncing,
  wsConnected,
}: {
  isSyncing: boolean;
  wsConnected: boolean;
}) {
  if (isSyncing) {
    return (
      <div className="flex items-center gap-1 text-xs text-muted-foreground">
        <Loader2 className="h-3 w-3 animate-spin" />
        <span>Syncing...</span>
      </div>
    );
  }

  return (
    <div className="flex items-center gap-1 text-xs">
      {wsConnected ? (
        <>
          <Wifi className="h-3 w-3 text-green-500" />
          <span className="text-green-600">Live</span>
        </>
      ) : (
        <>
          <WifiOff className="h-3 w-3 text-muted-foreground" />
          <span className="text-muted-foreground">Offline</span>
        </>
      )}
    </div>
  );
});

// ============================================================================
// Main Component Props
// ============================================================================

export interface OptimizedModelSelectorProps {
  onModelChange?: (provider: string, model: string) => void;
  showSessionInfo?: boolean;
  compact?: boolean;
  className?: string;
}

// ============================================================================
// Compact Variant
// ============================================================================

const CompactSelector = memo(function CompactSelector({
  onModelChange,
}: {
  onModelChange?: (provider: string, model: string) => void;
}) {
  const selection = useModelSelectionValue();
  const providers = useProvidersList();
  const { isLoading, isSyncing } = useModelSelectionLoading();
  const { selectProvider, selectModel } = useModelSelectionActions();

  // Get models for selected provider
  const selectedProviderInfo = useMemo(
    () => providers.find((p) => p.provider === selection.provider),
    [providers, selection.provider]
  );
  const models = selectedProviderInfo?.models || [];

  const handleProviderChange = useCallback(
    async (provider: string) => {
      const success = await selectProvider(provider);
      if (success) {
        const providerInfo = providers.find((p) => p.provider === provider);
        if (providerInfo?.defaultModel && onModelChange) {
          onModelChange(provider, providerInfo.defaultModel);
        }
      }
    },
    [selectProvider, providers, onModelChange]
  );

  const handleModelChange = useCallback(
    async (model: string) => {
      const success = await selectModel(model);
      if (success && selection.provider && onModelChange) {
        onModelChange(selection.provider, model);
      }
    },
    [selectModel, selection.provider, onModelChange]
  );

  if (isLoading && providers.length === 0) {
    return (
      <div className="flex items-center gap-4">
        <Skeleton className="h-9 w-[140px]" />
        <Skeleton className="h-9 w-[180px]" />
      </div>
    );
  }

  return (
    <div className="flex items-center gap-4">
      <div className="flex items-center gap-2">
        <Server className="h-4 w-4 text-muted-foreground" />
        <ProviderSelect
          selectedProvider={selection.provider}
          providers={providers}
          onSelect={handleProviderChange}
          disabled={isLoading}
          compact
        />
      </div>

      <div className="flex items-center gap-2">
        <Cpu className="h-4 w-4 text-muted-foreground" />
        <ModelSelect
          selectedModel={selection.model}
          models={models}
          onSelect={handleModelChange}
          disabled={isLoading || !selection.provider}
          compact
          defaultModel={selectedProviderInfo?.defaultModel}
        />
      </div>

      {isSyncing && <Loader2 className="h-4 w-4 animate-spin text-muted-foreground" />}
    </div>
  );
});

// ============================================================================
// Full Variant
// ============================================================================

const FullSelector = memo(function FullSelector({
  onModelChange,
  showSessionInfo,
}: {
  onModelChange?: (provider: string, model: string) => void;
  showSessionInfo: boolean;
}) {
  const {
    selectedProvider,
    selectedModel,
    providers,
    sessionId,
    isLoading,
    isSyncing,
    error,
    wsConnected,
    selectProvider,
    selectModel,
    refresh,
  } = useOptimizedModelSelection();

  // Get models for selected provider
  const selectedProviderInfo = useMemo(
    () => providers.find((p) => p.provider === selectedProvider),
    [providers, selectedProvider]
  );
  const models = selectedProviderInfo?.models || [];

  const handleProviderChange = useCallback(
    async (provider: string) => {
      const success = await selectProvider(provider);
      if (success) {
        const providerInfo = providers.find((p) => p.provider === provider);
        if (providerInfo?.defaultModel && onModelChange) {
          onModelChange(provider, providerInfo.defaultModel);
        }
      }
    },
    [selectProvider, providers, onModelChange]
  );

  const handleModelChange = useCallback(
    async (model: string) => {
      const success = await selectModel(model);
      if (success && selectedProvider && onModelChange) {
        onModelChange(selectedProvider, model);
      }
    },
    [selectModel, selectedProvider, onModelChange]
  );

  const handleRefresh = useCallback(async () => {
    await refresh();
  }, [refresh]);

  if (isLoading && providers.length === 0) {
    return (
      <Card>
        <CardContent className="flex items-center justify-center py-8">
          <Loader2 className="h-6 w-6 animate-spin mr-2" />
          <span>Loading models...</span>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <div>
            <CardTitle className="flex items-center gap-2">
              <Cpu className="h-5 w-5" />
              Model Selection
            </CardTitle>
            <CardDescription>
              Choose the AI provider and model for your requests
            </CardDescription>
          </div>
          <div className="flex items-center gap-2">
            <SyncIndicator isSyncing={isSyncing} wsConnected={wsConnected} />
            <Button variant="ghost" size="icon" onClick={handleRefresh} disabled={isLoading}>
              <RefreshCw className={cn("h-4 w-4", isLoading && "animate-spin")} />
            </Button>
          </div>
        </div>
      </CardHeader>
      <CardContent className="space-y-6">
        {/* Error Alert */}
        {error && (
          <Alert variant="destructive">
            <AlertCircle className="h-4 w-4" />
            <AlertTitle>Error</AlertTitle>
            <AlertDescription>{error}</AlertDescription>
          </Alert>
        )}

        {/* Provider Selection */}
        <div className="space-y-2">
          <Label>Provider</Label>
          <ProviderSelect
            selectedProvider={selectedProvider}
            providers={providers}
            onSelect={handleProviderChange}
            disabled={isLoading}
            compact={false}
          />
          {selectedProvider && (
            <p className="text-xs text-muted-foreground">
              {models.length} models available
            </p>
          )}
        </div>

        {/* Model Selection */}
        <div className="space-y-2">
          <Label>Model</Label>
          <ModelSelect
            selectedModel={selectedModel}
            models={models}
            onSelect={handleModelChange}
            disabled={isLoading || !selectedProvider}
            compact={false}
            defaultModel={selectedProviderInfo?.defaultModel}
          />
        </div>

        {/* Current Selection Summary */}
        <div className="rounded-lg border bg-muted/50 p-3">
          <div className="flex items-center gap-2">
            <Zap className="h-4 w-4 text-primary" />
            <span className="font-medium">Current Selection</span>
            {isSyncing && (
              <Badge variant="outline" className="ml-auto">
                <Loader2 className="h-3 w-3 animate-spin mr-1" />
                Syncing
              </Badge>
            )}
          </div>
          <p className="text-sm text-muted-foreground mt-1">
            {selectedProvider && selectedModel
              ? `${selectedModel} via ${selectedProvider}`
              : "No model selected"}
          </p>
        </div>

        {/* Session Info */}
        {showSessionInfo && sessionId && (
          <>
            <Separator />
            <div className="space-y-2">
              <Label className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
                Session Info
              </Label>
              <div className="text-sm">
                <span className="text-muted-foreground">Session ID:</span>
                <p className="font-mono text-xs truncate">{sessionId}</p>
              </div>
            </div>
          </>
        )}
      </CardContent>
    </Card>
  );
});

// ============================================================================
// Main Export
// ============================================================================

export const OptimizedModelSelector = memo(function OptimizedModelSelector({
  onModelChange,
  showSessionInfo = true,
  compact = false,
  className,
}: OptimizedModelSelectorProps) {
  if (compact) {
    return (
      <div className={className}>
        <CompactSelector onModelChange={onModelChange} />
      </div>
    );
  }

  return (
    <div className={className}>
      <FullSelector onModelChange={onModelChange} showSessionInfo={showSessionInfo} />
    </div>
  );
});

export default OptimizedModelSelector;