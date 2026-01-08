"use client";

/**
 * Enhanced Model Selector Component
 *
 * Two-panel layout for selecting AI provider and model:
 * - Left panel: Provider selection with health status
 * - Right panel: Model list filtered by selected provider
 */

import React, { useState, useEffect } from "react";
import { cn } from "@/lib/utils";
import {
  useModelSelection,
  type ProviderInfo,
  type ModelInfo,
} from "@/lib/stores/model-selection-store";
import { ProviderCard, ProviderStatusBadge } from "@/components/ui/provider-status-badge";

export interface EnhancedModelSelectorProps {
  onSelectionChange?: (provider: string, model: string) => void;
  className?: string;
  showSessionInfo?: boolean;
}

export function EnhancedModelSelector({
  onSelectionChange,
  className,
  showSessionInfo = true,
}: EnhancedModelSelectorProps) {
  const {
    providers,
    models,
    selectedProvider,
    selectedModel,
    sessionId,
    isDefault,
    isLoading,
    error,
    wsConnected,
    selectProvider,
    selectModel,
  } = useModelSelection();

  const [searchQuery, setSearchQuery] = useState("");

  // Filter models based on search
  const filteredModels = models.filter(
    (model) =>
      model.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
      model.id.toLowerCase().includes(searchQuery.toLowerCase())
  );

  // Handle provider selection
  const handleProviderSelect = async (provider: string) => {
    const success = await selectProvider(provider);
    if (success && onSelectionChange) {
      const providerInfo = providers.find((p) => p.provider === provider);
      if (providerInfo?.defaultModel) {
        onSelectionChange(provider, providerInfo.defaultModel);
      }
    }
  };

  // Handle model selection
  const handleModelSelect = async (model: string) => {
    const success = await selectModel(model);
    if (success && selectedProvider && onSelectionChange) {
      onSelectionChange(selectedProvider, model);
    }
  };

  // Get status for provider
  const getProviderStatus = (provider: ProviderInfo): "healthy" | "degraded" | "unavailable" | "unknown" => {
    if (!provider.isHealthy) return "unavailable";
    if (provider.status === "active") return "healthy";
    if (provider.status === "degraded") return "degraded";
    return "unknown";
  };

  // Get tier badge color
  const getTierColor = (tier: string) => {
    switch (tier) {
      case "premium":
        return "bg-purple-100 text-purple-800 dark:bg-purple-900/30 dark:text-purple-300";
      case "experimental":
        return "bg-orange-100 text-orange-800 dark:bg-orange-900/30 dark:text-orange-300";
      default:
        return "bg-gray-100 text-gray-800 dark:bg-gray-800 dark:text-gray-300";
    }
  };

  return (
    <div className={cn("flex flex-col gap-4", className)}>
      {/* Header with connection status */}
      <div className="flex items-center justify-between">
        <h3 className="text-lg font-semibold">Select AI Model</h3>
        <div className="flex items-center gap-2">
          {wsConnected && (
            <span className="flex items-center gap-1 text-xs text-green-600 dark:text-green-400">
              <span className="relative flex h-2 w-2">
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-green-400 opacity-75"></span>
                <span className="relative inline-flex rounded-full h-2 w-2 bg-green-500"></span>
              </span>
              Live sync
            </span>
          )}
          {isLoading && (
            <span className="text-xs text-muted-foreground">Loading...</span>
          )}
        </div>
      </div>

      {/* Error display */}
      {error && (
        <div className="p-3 rounded-lg bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800">
          <p className="text-sm text-red-600 dark:text-red-400">{error}</p>
        </div>
      )}

      {/* Two-panel layout */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* Provider Selection Panel */}
        <div className="space-y-3">
          <h4 className="text-sm font-medium text-muted-foreground">Provider</h4>
          <div className="grid grid-cols-1 gap-2">
            {providers.map((provider) => (
              <ProviderCard
                key={provider.provider}
                provider={provider.provider}
                displayName={provider.displayName}
                status={getProviderStatus(provider)}
                isSelected={selectedProvider === provider.provider}
                latencyMs={provider.latencyMs}
                modelCount={provider.models.length}
                onClick={() => handleProviderSelect(provider.provider)}
                disabled={isLoading}
              />
            ))}
          </div>
        </div>

        {/* Model Selection Panel */}
        <div className="space-y-3">
          <div className="flex items-center justify-between">
            <h4 className="text-sm font-medium text-muted-foreground">Model</h4>
            {selectedProvider && (
              <span className="text-xs text-muted-foreground">
                {filteredModels.length} model{filteredModels.length !== 1 ? "s" : ""}
              </span>
            )}
          </div>

          {/* Search input */}
          {models.length > 5 && (
            <input
              type="text"
              placeholder="Search models..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="w-full px-3 py-2 text-sm border rounded-lg bg-background focus:outline-none focus:ring-2 focus:ring-primary/50"
            />
          )}

          {/* Model list */}
          <div className="space-y-2 max-h-[300px] overflow-y-auto">
            {!selectedProvider ? (
              <p className="text-sm text-muted-foreground text-center py-4">
                Select a provider to see available models
              </p>
            ) : filteredModels.length === 0 ? (
              <p className="text-sm text-muted-foreground text-center py-4">
                No models found
              </p>
            ) : (
              filteredModels.map((model) => (
                <button
                  key={model.id}
                  onClick={() => handleModelSelect(model.id)}
                  disabled={isLoading}
                  className={cn(
                    "w-full flex items-center justify-between p-3 rounded-lg border transition-all",
                    "hover:bg-accent focus:outline-none focus:ring-2 focus:ring-primary/50",
                    selectedModel === model.id
                      ? "border-primary bg-primary/5"
                      : "border-border",
                    isLoading && "opacity-50 cursor-not-allowed"
                  )}
                >
                  <div className="flex flex-col items-start gap-1">
                    <span className="font-medium text-sm">{model.name}</span>
                    <span className="text-xs text-muted-foreground">{model.id}</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <span
                      className={cn(
                        "px-2 py-0.5 rounded text-xs font-medium",
                        getTierColor(model.tier)
                      )}
                    >
                      {model.tier}
                    </span>
                    {model.isDefault && (
                      <span className="px-2 py-0.5 rounded text-xs font-medium bg-blue-100 text-blue-800 dark:bg-blue-900/30 dark:text-blue-300">
                        default
                      </span>
                    )}
                    {selectedModel === model.id && (
                      <svg
                        className="w-4 h-4 text-primary"
                        fill="currentColor"
                        viewBox="0 0 20 20"
                      >
                        <path
                          fillRule="evenodd"
                          d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z"
                          clipRule="evenodd"
                        />
                      </svg>
                    )}
                  </div>
                </button>
              ))
            )}
          </div>
        </div>
      </div>

      {/* Session info footer */}
      {showSessionInfo && (
        <div className="flex items-center justify-between pt-3 border-t text-xs text-muted-foreground">
          <div className="flex items-center gap-4">
            {selectedProvider && selectedModel && (
              <span>
                Current: <strong>{selectedProvider}</strong> / <strong>{selectedModel}</strong>
                {isDefault && " (default)"}
              </span>
            )}
          </div>
          {sessionId && (
            <span className="font-mono text-[10px]">
              Session: {sessionId.slice(0, 8)}...
            </span>
          )}
        </div>
      )}
    </div>
  );
}

export default EnhancedModelSelector;
