"use client";

/**
 * Enhanced Provider Selector Component
 * 
 * Features:
 * - Real-time provider status indicators
 * - Rate limit warnings
 * - Fallback suggestions
 * - WebSocket-based updates
 */

import { useState, useEffect, useCallback } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Progress } from "@/components/ui/progress";
import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectLabel,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import {
  useProviders,
  useModels,
  useModelSelection,
  useProviderHealth,
  useProviderSync,
  useModelUpdates,
} from "@/hooks";
import { ProviderStatus, ModelInfo } from "@/lib/types/provider-management-types";
import {
  CheckCircle,
  XCircle,
  AlertTriangle,
  Loader2,
  RefreshCw,
  Zap,
  Clock,
  Shield,
  Wifi,
  WifiOff,
} from "lucide-react";

// =============================================================================
// Status Badge Component
// =============================================================================

function ProviderStatusBadge({ status }: { status: ProviderStatus | null }) {
  if (!status) {
    return (
      <Badge variant="secondary" className="flex items-center gap-1">
        <Loader2 className="h-3 w-3 animate-spin" />
        Loading
      </Badge>
    );
  }

  const statusConfig: Record<ProviderStatus, { variant: "default" | "secondary" | "destructive" | "outline"; icon: React.ReactNode; label: string }> = {
    [ProviderStatus.AVAILABLE]: {
      variant: "default",
      icon: <CheckCircle className="h-3 w-3" />,
      label: "Available",
    },
    [ProviderStatus.DEGRADED]: {
      variant: "outline",
      icon: <AlertTriangle className="h-3 w-3 text-yellow-500" />,
      label: "Degraded",
    },
    [ProviderStatus.UNAVAILABLE]: {
      variant: "destructive",
      icon: <XCircle className="h-3 w-3" />,
      label: "Unavailable",
    },
    [ProviderStatus.RATE_LIMITED]: {
      variant: "secondary",
      icon: <Clock className="h-3 w-3" />,
      label: "Rate Limited",
    },
    [ProviderStatus.UNKNOWN]: {
      variant: "secondary",
      icon: <Loader2 className="h-3 w-3 animate-spin" />,
      label: "Unknown",
    },
  };

  const config = statusConfig[status];

  return (
    <Badge variant={config.variant} className="flex items-center gap-1">
      {config.icon}
      {config.label}
    </Badge>
  );
}

// =============================================================================
// Rate Limit Warning Component
// =============================================================================

interface RateLimitWarningProps {
  remaining: number;
  limit: number;
  resetTime?: Date;
  fallbackSuggestions?: string[];
}

function RateLimitWarning({ remaining, limit, resetTime, fallbackSuggestions }: RateLimitWarningProps) {
  const percentage = (remaining / limit) * 100;
  const isLow = percentage < 20;
  const isCritical = percentage < 5;

  if (percentage > 50) return null;

  return (
    <Alert variant={isCritical ? "destructive" : "default"} className="mt-4">
      <AlertTriangle className="h-4 w-4" />
      <AlertTitle>
        {isCritical ? "Rate Limit Critical" : isLow ? "Rate Limit Warning" : "Rate Limit Notice"}
      </AlertTitle>
      <AlertDescription className="space-y-2">
        <div className="flex items-center gap-2">
          <span>{remaining} / {limit} requests remaining</span>
          <Progress value={percentage} className="w-24 h-2" />
        </div>
        {resetTime && (
          <p className="text-sm text-muted-foreground">
            Resets at {resetTime.toLocaleTimeString()}
          </p>
        )}
        {fallbackSuggestions && fallbackSuggestions.length > 0 && (
          <div className="mt-2">
            <p className="text-sm font-medium">Suggested alternatives:</p>
            <div className="flex flex-wrap gap-1 mt-1">
              {fallbackSuggestions.map((suggestion) => (
                <Badge key={suggestion} variant="outline" className="text-xs">
                  {suggestion}
                </Badge>
              ))}
            </div>
          </div>
        )}
      </AlertDescription>
    </Alert>
  );
}

// =============================================================================
// Model Card Component
// =============================================================================

interface ModelCardProps {
  model: ModelInfo;
  isSelected: boolean;
  onSelect: () => void;
  disabled?: boolean;
}

function ModelCard({ model, isSelected, onSelect, disabled }: ModelCardProps) {
  return (
    <div
      className={`p-3 rounded-lg border cursor-pointer transition-all ${
        isSelected
          ? "border-primary bg-primary/5"
          : "border-border hover:border-primary/50"
      } ${disabled ? "opacity-50 cursor-not-allowed" : ""}`}
      onClick={disabled ? undefined : onSelect}
    >
      <div className="flex items-start justify-between">
        <div>
          <p className="font-medium text-sm">{model.name || model.model_id}</p>
          <p className="text-xs text-muted-foreground font-mono">{model.model_id}</p>
        </div>
        {isSelected && <CheckCircle className="h-4 w-4 text-primary" />}
      </div>
      <div className="flex flex-wrap gap-1 mt-2">
        {model.capabilities?.map((cap) => (
          <Badge key={cap} variant="secondary" className="text-xs">
            {cap}
          </Badge>
        ))}
      </div>
      {model.context_window && (
        <p className="text-xs text-muted-foreground mt-1">
          Context: {model.context_window.toLocaleString()} tokens
        </p>
      )}
    </div>
  );
}

// =============================================================================
// Main Enhanced Provider Selector
// =============================================================================

export interface EnhancedProviderSelectorProps {
  onSelectionChange?: (providerId: string, modelId: string) => void;
  showRateLimitWarnings?: boolean;
  showFallbackSuggestions?: boolean;
  enableWebSocketSync?: boolean;
}

export function EnhancedProviderSelector({
  onSelectionChange,
  showRateLimitWarnings = true,
  showFallbackSuggestions = true,
  enableWebSocketSync = true,
}: EnhancedProviderSelectorProps) {
  const { providers, isLoading: providersLoading, refresh: refreshProviders } = useProviders();
  const { modelsByProvider, loadModelsForProvider, getModels } = useModels();
  const { currentSelection, selectModel, clearSelection, isLoading: selectionLoading } = useModelSelection();
  const { healthStatus, getProviderHealth, isHealthy, refresh: refreshHealth } = useProviderHealth();

  const [selectedProviderId, setSelectedProviderId] = useState<string>("");
  const [selectedModelId, setSelectedModelId] = useState<string>("");
  const [rateLimitInfo, setRateLimitInfo] = useState<{
    remaining: number;
    limit: number;
    resetTime?: Date;
  } | null>(null);

  // WebSocket sync for cross-tab updates
  const { isConnected: syncConnected, connect: connectSync, broadcastSelection } = useProviderSync(
    (providerId, modelId) => {
      // Handle selection from another tab
      setSelectedProviderId(providerId);
      setSelectedModelId(modelId);
    },
    () => {
      // Handle selection cleared
      setSelectedProviderId("");
      setSelectedModelId("");
    }
  );

  // WebSocket for model updates
  const { isConnected: updatesConnected, connect: connectUpdates } = useModelUpdates(
    (providerId, model) => {
      // Model added - refresh models for provider
      loadModelsForProvider(providerId);
    },
    (providerId, modelId) => {
      // Model removed - refresh models for provider
      loadModelsForProvider(providerId);
    },
    (providerId, status) => {
      // Provider status changed - refresh health
      refreshHealth();
    }
  );

  // Connect WebSockets on mount
  useEffect(() => {
    if (enableWebSocketSync) {
      connectSync();
      connectUpdates();
    }
  }, [enableWebSocketSync, connectSync, connectUpdates]);

  // Sync with current selection
  useEffect(() => {
    if (currentSelection) {
      setSelectedProviderId(currentSelection.provider_id || "");
      setSelectedModelId(currentSelection.model_id || "");
    }
  }, [currentSelection]);

  // Load models when provider changes
  useEffect(() => {
    if (selectedProviderId) {
      loadModelsForProvider(selectedProviderId);
    }
  }, [selectedProviderId, loadModelsForProvider]);

  // Handle provider selection
  const handleProviderChange = useCallback((providerId: string) => {
    setSelectedProviderId(providerId);
    setSelectedModelId("");
  }, []);

  // Handle model selection
  const handleModelSelect = useCallback(async (modelId: string) => {
    if (!selectedProviderId) return;

    setSelectedModelId(modelId);

    try {
      await selectModel(selectedProviderId, modelId);
      
      // Broadcast to other tabs
      if (enableWebSocketSync) {
        broadcastSelection(selectedProviderId, modelId);
      }

      onSelectionChange?.(selectedProviderId, modelId);
    } catch (error) {
      console.error("Failed to select model:", error);
    }
  }, [selectedProviderId, selectModel, enableWebSocketSync, broadcastSelection, onSelectionChange]);

  // Handle clear selection
  const handleClearSelection = useCallback(async () => {
    try {
      await clearSelection();
      setSelectedProviderId("");
      setSelectedModelId("");
    } catch (error) {
      console.error("Failed to clear selection:", error);
    }
  }, [clearSelection]);

  // Get fallback suggestions based on current provider
  const getFallbackSuggestions = useCallback((): string[] => {
    if (!selectedProviderId) return [];

    return providers
      .filter((p) => p.provider_id !== selectedProviderId && isHealthy(p.provider_id))
      .slice(0, 3)
      .map((p) => p.name || p.provider_id);
  }, [selectedProviderId, providers, isHealthy]);

  const models = selectedProviderId ? getModels(selectedProviderId) : [];
  const providerStatus = selectedProviderId ? getProviderHealth(selectedProviderId) : null;

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <div>
            <CardTitle className="flex items-center gap-2">
              <Zap className="h-5 w-5" />
              Model Selection
            </CardTitle>
            <CardDescription>
              Select a provider and model for AI operations
            </CardDescription>
          </div>
          <div className="flex items-center gap-2">
            {/* WebSocket Status */}
            <TooltipProvider>
              <Tooltip>
                <TooltipTrigger>
                  {syncConnected ? (
                    <Wifi className="h-4 w-4 text-green-500" />
                  ) : (
                    <WifiOff className="h-4 w-4 text-muted-foreground" />
                  )}
                </TooltipTrigger>
                <TooltipContent>
                  {syncConnected ? "Real-time sync active" : "Real-time sync disconnected"}
                </TooltipContent>
              </Tooltip>
            </TooltipProvider>

            <Button variant="ghost" size="sm" onClick={() => { refreshProviders(); refreshHealth(); }}>
              <RefreshCw className="h-4 w-4" />
            </Button>
          </div>
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Provider Selection */}
        <div className="space-y-2">
          <label className="text-sm font-medium">Provider</label>
          <Select value={selectedProviderId} onValueChange={handleProviderChange}>
            <SelectTrigger>
              <SelectValue placeholder="Select a provider" />
            </SelectTrigger>
            <SelectContent>
              <SelectGroup>
                <SelectLabel>Available Providers</SelectLabel>
                {providers.map((provider) => {
                  const status = getProviderHealth(provider.provider_id);
                  return (
                    <SelectItem
                      key={provider.provider_id}
                      value={provider.provider_id}
                      disabled={status === ProviderStatus.UNAVAILABLE}
                    >
                      <div className="flex items-center gap-2">
                        <span>{provider.name || provider.provider_id}</span>
                        <ProviderStatusBadge status={status} />
                      </div>
                    </SelectItem>
                  );
                })}
              </SelectGroup>
            </SelectContent>
          </Select>
        </div>

        {/* Provider Status */}
        {selectedProviderId && (
          <div className="flex items-center gap-2">
            <span className="text-sm text-muted-foreground">Status:</span>
            <ProviderStatusBadge status={providerStatus} />
          </div>
        )}

        {/* Model Selection */}
        {selectedProviderId && models.length > 0 && (
          <div className="space-y-2">
            <label className="text-sm font-medium">Model</label>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-2 max-h-64 overflow-y-auto">
              {models.map((model) => (
                <ModelCard
                  key={model.model_id}
                  model={model}
                  isSelected={selectedModelId === model.model_id}
                  onSelect={() => handleModelSelect(model.model_id)}
                  disabled={selectionLoading}
                />
              ))}
            </div>
          </div>
        )}

        {/* Loading State */}
        {providersLoading && (
          <div className="flex items-center justify-center p-4">
            <Loader2 className="h-6 w-6 animate-spin" />
          </div>
        )}

        {/* Rate Limit Warning */}
        {showRateLimitWarnings && rateLimitInfo && (
          <RateLimitWarning
            remaining={rateLimitInfo.remaining}
            limit={rateLimitInfo.limit}
            resetTime={rateLimitInfo.resetTime}
            fallbackSuggestions={showFallbackSuggestions ? getFallbackSuggestions() : undefined}
          />
        )}

        {/* Current Selection Summary */}
        {currentSelection?.provider_id && currentSelection?.model_id && (
          <div className="flex items-center justify-between p-3 bg-muted rounded-lg">
            <div>
              <p className="text-sm font-medium">Current Selection</p>
              <p className="text-xs text-muted-foreground">
                {currentSelection.provider_id} / {currentSelection.model_id}
              </p>
            </div>
            <Button variant="ghost" size="sm" onClick={handleClearSelection}>
              Clear
            </Button>
          </div>
        )}
      </CardContent>
    </Card>
  );
}

export default EnhancedProviderSelector;