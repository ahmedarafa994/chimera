
"use client";

import * as React from "react";
import { useState, useCallback, useMemo, useEffect } from "react";
import { ProviderSelector, type Provider, type ProviderStatus as ProviderSelectorStatus } from "./ProviderSelector";
import { type ProviderType } from "@/lib/api/validation";
import {
  ProviderConfigForm,
  ProviderList,
  type ProviderConfigData,
} from "./ProviderConfigForm";
import {
  SyncStatusIndicator,
  ModelDeprecationWarning,
  ProviderUnavailableWarning,
} from "./SyncStatusIndicator";
import { useProviderSystem } from "@/hooks/useProviderConfig";
import { useProviderSync } from "@/contexts/ProviderSyncContext";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Skeleton } from "@/components/ui/skeleton";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Separator } from "@/components/ui/separator";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { cn } from "@/lib/utils";
import {
  Wifi,
  WifiOff,
  Settings,
  Zap,
  Activity,
  AlertTriangle,
  CheckCircle2,
  RefreshCw,
  Server,
  Cpu,
  TrendingUp,
  AlertCircle,
  Sparkles,
  Eye,
  MessageSquare,
  Code,
  DollarSign,
} from "lucide-react";
import {
  ModelSpecification,
  ProviderSyncInfo,
  ProviderStatus,
  ModelDeprecationStatus,
  SyncStatus,
} from "@/types/provider-sync";

// =============================================================================
// Model Selector Component
// =============================================================================

interface ModelSelectorProps {
  models: ModelSpecification[];
  selectedModelId?: string;
  onModelSelect: (modelId: string) => void;
  disabled?: boolean;
  showCapabilities?: boolean;
  className?: string;
}

function ModelSelector({
  models,
  selectedModelId,
  onModelSelect,
  disabled = false,
  showCapabilities = true,
  className,
}: ModelSelectorProps) {
  const selectedModel = models.find((m) => m.id === selectedModelId);

  const getModelIcon = (model: ModelSpecification) => {
    if (model.supports_vision) return Eye;
    if (model.supports_function_calling) return Code;
    return MessageSquare;
  };

  const formatContextWindow = (tokens: number) => {
    if (tokens >= 1000000) return `${(tokens / 1000000).toFixed(1)}M`;
    if (tokens >= 1000) return `${(tokens / 1000).toFixed(0)}K`;
    return tokens.toString();
  };

  return (
    <div className={cn("space-y-3", className)}>
      <Select
        value={selectedModelId}
        onValueChange={onModelSelect}
        disabled={disabled}
      >
        <SelectTrigger className="w-full">
          <SelectValue placeholder="Select a model">
            {selectedModel && (
              <div className="flex items-center gap-2">
                <span>{selectedModel.name}</span>
                {selectedModel.deprecation_status !== "active" && (
                  <Badge variant="destructive" className="text-[10px] h-4">
                    Deprecated
                  </Badge>
                )}
              </div>
            )}
          </SelectValue>
        </SelectTrigger>
        <SelectContent>
          <ScrollArea className="h-[300px]">
            {models.map((model) => {
              const Icon = getModelIcon(model);
              const isDeprecated = model.deprecation_status !== "active";

              return (
                <SelectItem
                  key={model.id}
                  value={model.id}
                  className={cn(
                    "py-3",
                    isDeprecated && "opacity-70"
                  )}
                >
                  <div className="flex items-start gap-3 w-full">
                    <Icon className="h-4 w-4 mt-0.5 text-muted-foreground" />
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2">
                        <span className="font-medium">{model.name}</span>
                        {model.is_default && (
                          <Badge variant="secondary" className="text-[10px] h-4">
                            Default
                          </Badge>
                        )}
                        {isDeprecated && (
                          <Badge variant="destructive" className="text-[10px] h-4">
                            Deprecated
                          </Badge>
                        )}
                      </div>
                      {model.description && (
                        <p className="text-xs text-muted-foreground truncate mt-0.5">
                          {model.description}
                        </p>
                      )}
                      <div className="flex items-center gap-2 mt-1 text-[10px] text-muted-foreground">
                        <span>{formatContextWindow(model.context_window)} ctx</span>
                        {model.supports_vision && <span>• Vision</span>}
                        {model.supports_function_calling && <span>• Functions</span>}
                        {model.supports_streaming && <span>• Streaming</span>}
                      </div>
                    </div>
                  </div>
                </SelectItem>
              );
            })}
          </ScrollArea>
        </SelectContent>
      </Select>

      {/* Selected model details */}
      {selectedModel && showCapabilities && (
        <div className="p-3 rounded-lg bg-muted/50 space-y-2">
          <div className="flex items-center justify-between">
            <span className="text-sm font-medium">{selectedModel.name}</span>
            <Badge variant="outline" className="text-[10px]">
              {selectedModel.tier}
            </Badge>
          </div>

          <div className="grid grid-cols-2 gap-2 text-xs">
            <div className="flex items-center gap-1.5 text-muted-foreground">
              <Cpu className="h-3 w-3" />
              <span>{formatContextWindow(selectedModel.context_window)} context</span>
            </div>
            <div className="flex items-center gap-1.5 text-muted-foreground">
              <TrendingUp className="h-3 w-3" />
              <span>{formatContextWindow(selectedModel.max_output_tokens)} max output</span>
            </div>
          </div>

          <div className="flex flex-wrap gap-1.5">
            {selectedModel.supports_streaming && (
              <Badge variant="secondary" className="text-[10px]">Streaming</Badge>
            )}
            {selectedModel.supports_vision && (
              <Badge variant="secondary" className="text-[10px]">Vision</Badge>
            )}
            {selectedModel.supports_function_calling && (
              <Badge variant="secondary" className="text-[10px]">Functions</Badge>
            )}
            {selectedModel.supports_json_mode && (
              <Badge variant="secondary" className="text-[10px]">JSON Mode</Badge>
            )}
          </div>

          {selectedModel.pricing && (
            <div className="flex items-center gap-1.5 text-xs text-muted-foreground pt-1 border-t border-border/50">
              <DollarSign className="h-3 w-3" />
              <span>
                ${selectedModel.pricing.input_per_1k_tokens}/1K in •
                ${selectedModel.pricing.output_per_1k_tokens}/1K out
              </span>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

// =============================================================================
// Provider Health Dashboard
// =============================================================================

interface ProviderHealthDashboardProps {
  providers: ProviderSyncInfo[];
  className?: string;
}

function ProviderHealthDashboard({ providers, className }: ProviderHealthDashboardProps) {
  const healthyCount = providers.filter(
    (p) => p.health?.status === ProviderStatus.AVAILABLE
  ).length;
  const degradedCount = providers.filter(
    (p) => p.health?.status === ProviderStatus.DEGRADED
  ).length;
  const unavailableCount = providers.filter(
    (p) => p.health?.status === ProviderStatus.UNAVAILABLE
  ).length;

  const getStatusColor = (status?: ProviderStatus) => {
    switch (status) {
      case ProviderStatus.AVAILABLE:
        return "text-emerald-500";
      case ProviderStatus.DEGRADED:
        return "text-amber-500";
      case ProviderStatus.UNAVAILABLE:
        return "text-red-500";
      case ProviderStatus.RATE_LIMITED:
        return "text-orange-500";
      case ProviderStatus.MAINTENANCE:
        return "text-blue-500";
      default:
        return "text-zinc-500";
    }
  };

  const getStatusBg = (status?: ProviderStatus) => {
    switch (status) {
      case ProviderStatus.AVAILABLE:
        return "bg-emerald-500/10";
      case ProviderStatus.DEGRADED:
        return "bg-amber-500/10";
      case ProviderStatus.UNAVAILABLE:
        return "bg-red-500/10";
      case ProviderStatus.RATE_LIMITED:
        return "bg-orange-500/10";
      case ProviderStatus.MAINTENANCE:
        return "bg-blue-500/10";
      default:
        return "bg-zinc-500/10";
    }
  };

  return (
    <Card className={className}>
      <CardHeader className="pb-3">
        <CardTitle className="text-base flex items-center gap-2">
          <Activity className="h-4 w-4" />
          Provider Health
        </CardTitle>
        <CardDescription>Real-time status of all configured providers</CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Summary stats */}
        <div className="grid grid-cols-3 gap-3">
          <div className="flex items-center gap-2 p-2 rounded-lg bg-emerald-500/10">
            <CheckCircle2 className="h-4 w-4 text-emerald-500" />
            <div>
              <p className="text-lg font-semibold text-emerald-500">{healthyCount}</p>
              <p className="text-[10px] text-muted-foreground">Healthy</p>
            </div>
          </div>
          <div className="flex items-center gap-2 p-2 rounded-lg bg-amber-500/10">
            <AlertTriangle className="h-4 w-4 text-amber-500" />
            <div>
              <p className="text-lg font-semibold text-amber-500">{degradedCount}</p>
              <p className="text-[10px] text-muted-foreground">Degraded</p>
            </div>
          </div>
          <div className="flex items-center gap-2 p-2 rounded-lg bg-red-500/10">
            <AlertCircle className="h-4 w-4 text-red-500" />
            <div>
              <p className="text-lg font-semibold text-red-500">{unavailableCount}</p>
              <p className="text-[10px] text-muted-foreground">Down</p>
            </div>
          </div>
        </div>

        <Separator />

        {/* Provider list */}
        <ScrollArea className="h-[200px]">
          <div className="space-y-2">
            {providers.map((provider) => (
              <div
                key={provider.id}
                className={cn(
                  "flex items-center justify-between p-2 rounded-lg",
                  getStatusBg(provider.health?.status)
                )}
              >
                <div className="flex items-center gap-2">
                  <Server className={cn("h-4 w-4", getStatusColor(provider.health?.status))} />
                  <div>
                    <p className="text-sm font-medium">{provider.display_name}</p>
                    <p className="text-[10px] text-muted-foreground">
                      {provider.model_count} models
                    </p>
                  </div>
                </div>
                <div className="flex items-center gap-2">
                  {provider.health?.latency_ms && (
                    <TooltipProvider>
                      <Tooltip>
                        <TooltipTrigger>
                          <Badge variant="outline" className="text-[10px]">
                            {provider.health.latency_ms}ms
                          </Badge>
                        </TooltipTrigger>
                        <TooltipContent>Response latency</TooltipContent>
                      </Tooltip>
                    </TooltipProvider>
                  )}
                  <Badge
                    variant="outline"
                    className={cn("text-[10px]", getStatusColor(provider.health?.status))}
                  >
                    {provider.health?.status || "unknown"}
                  </Badge>
                </div>
              </div>
            ))}
          </div>
        </ScrollArea>
      </CardContent>
    </Card>
  );
}

// =============================================================================
// Deprecation Alerts Panel
// =============================================================================

interface DeprecationAlertsPanelProps {
  models: ModelSpecification[];
  onSelectModel: (modelId: string) => void;
  className?: string;
}

function DeprecationAlertsPanel({
  models,
  onSelectModel,
  className,
}: DeprecationAlertsPanelProps) {
  const deprecatedModels = models.filter(
    (m) => m.deprecation_status !== ModelDeprecationStatus.ACTIVE
  );

  if (deprecatedModels.length === 0) {
    return null;
  }

  return (
    <Card className={cn("border-amber-500/30", className)}>
      <CardHeader className="pb-3">
        <CardTitle className="text-base flex items-center gap-2 text-amber-500">
          <AlertTriangle className="h-4 w-4" />
          Deprecation Alerts
        </CardTitle>
        <CardDescription>
          {deprecatedModels.length} model{deprecatedModels.length > 1 ? "s" : ""} require attention
        </CardDescription>
      </CardHeader>
      <CardContent>
        <ScrollArea className="h-[200px]">
          <div className="space-y-3">
            {deprecatedModels.map((model) => (
              <ModelDeprecationWarning
                key={model.id}
                modelId={model.id}
                modelName={model.name}
                deprecationDate={model.deprecation_date}
                sunsetDate={model.sunset_date}
                replacementModelId={model.replacement_model_id}
                onSelectReplacement={onSelectModel}
              />
            ))}
          </div>
        </ScrollArea>
      </CardContent>
    </Card>
  );
}

// =============================================================================
// Helper function to convert sync status to selector status
// =============================================================================

function mapProviderStatusToSelectorStatus(
  syncStatus: ProviderStatus | undefined
): ProviderSelectorStatus {
  switch (syncStatus) {
    case ProviderStatus.AVAILABLE:
      return "available";
    case ProviderStatus.DEGRADED:
      return "degraded";
    case ProviderStatus.UNAVAILABLE:
    case ProviderStatus.RATE_LIMITED:
    case ProviderStatus.MAINTENANCE:
      return "unavailable";
    default:
      return "unknown";
  }
}

// =============================================================================
// Provider Settings Page Component (Enhanced)
// =============================================================================

export function ProviderSettingsPage() {
  // Legacy provider system hook (for CRUD operations)
  const {
    providers: legacyProviders,
    activeProviderId: legacyActiveProviderId,
    activeModelId: legacyActiveModelId,
    isLoading: isLegacyLoading,
    isRefreshing,
    isChangingProvider,
    isSaving,
    isDeleting,
    error: legacyError,
    setActiveProvider,
    createProvider,
    updateProvider,
    deleteProvider,
    testConnection,
    refreshAll,
    isWebSocketConnected: legacyWsConnected,
    webSocketError: legacyWsError,
  } = useProviderSystem({ enableWebSocket: true });

  // Use the provider sync context for real-time synchronization
  // This provides access to provider/model data with WebSocket updates
  const syncContext = useProviderSync();

  // Extract sync context values for easier access
  const syncProviders = syncContext?.providers ?? [];
  const syncModels = syncContext?.models ?? [];

  // Determine which data source to use - prefer sync system if available
  const useSyncSystem = syncContext !== null && syncProviders.length > 0;

  // Map sync providers to the Provider type expected by ProviderSelector
  const providers: Provider[] = useSyncSystem
    ? syncProviders.map((p) => ({
        id: p.id,
        name: p.display_name,
        // Cast provider-sync ProviderType to api/validation ProviderType (they have different enum definitions)
        type: p.type as unknown as ProviderType,
        status: mapProviderStatusToSelectorStatus(p.health?.status),
        isActive: p.id === syncContext?.activeProviderId,
        hasApiKey: p.is_configured,
        enabled: p.enabled,
        models: p.models.map((m) => ({
          id: m.id,
          name: m.name,
          contextWindow: m.context_window,
        })),
        responseTime: p.health?.latency_ms,
        baseUrl: p.base_url,
      }))
    : legacyProviders;

  const activeProviderId = useSyncSystem
    ? syncContext?.activeProviderId
    : legacyActiveProviderId;
  const activeModelId = useSyncSystem
    ? syncContext?.activeModelId
    : legacyActiveModelId;
  const isLoading = useSyncSystem
    ? syncContext?.isLoading ?? false
    : isLegacyLoading;
  const error = useSyncSystem
    ? syncContext?.error
    : legacyError;
  const isWebSocketConnected = useSyncSystem
    ? syncContext?.isConnected ?? false
    : legacyWsConnected;

  // Dialog state
  const [isConfigDialogOpen, setIsConfigDialogOpen] = useState(false);
  const [editingProviderId, setEditingProviderId] = useState<string | null>(null);
  const [isTesting, setIsTesting] = useState(false);
  const [selectedTab, setSelectedTab] = useState("selector");

  // Get editing provider data
  const editingProvider = editingProviderId
    ? providers.find((p) => p.id === editingProviderId)
    : null;

  // Get active provider and its models
  const activeProvider = providers.find((p) => p.id === activeProviderId);
  const activeProviderModels = useSyncSystem && activeProviderId
    ? syncModels.filter((m) => m.provider_id === activeProviderId)
    : [];

  // Handlers - use sync context methods when available for optimistic updates
  const handleProviderChange = useCallback(
    async (providerId: string) => {
      try {
        if (syncContext?.selectProvider) {
          // Use sync context for optimistic updates and cross-tab sync
          await syncContext.selectProvider(providerId);
        } else {
          // Fall back to legacy provider system
          await setActiveProvider(providerId);
        }
      } catch (err) {
        console.error("Failed to change provider:", err);
      }
    },
    [syncContext, setActiveProvider]
  );

  const handleModelChange = useCallback(
    async (modelId: string) => {
      try {
        if (syncContext?.selectModel) {
          // Use sync context for optimistic updates and cross-tab sync
          await syncContext.selectModel(modelId);
        } else if (activeProviderId) {
          // Fall back to legacy provider system
          await setActiveProvider(activeProviderId, modelId);
        }
      } catch (err) {
        console.error("Failed to change model:", err);
      }
    },
    [syncContext, activeProviderId, setActiveProvider]
  );

  const handleConfigureProvider = useCallback((providerId: string) => {
    setEditingProviderId(providerId);
    setIsConfigDialogOpen(true);
  }, []);

  const handleAddProvider = useCallback(() => {
    setEditingProviderId(null);
    setIsConfigDialogOpen(true);
  }, []);

  const handleSaveProvider = useCallback(
    async (config: ProviderConfigData) => {
      if (editingProviderId) {
        await updateProvider(editingProviderId, config);
      } else {
        await createProvider(config);
      }
      setIsConfigDialogOpen(false);
      setEditingProviderId(null);
    },
    [editingProviderId, createProvider, updateProvider]
  );

  const handleDeleteProvider = useCallback(
    async (providerId: string) => {
      await deleteProvider(providerId);
      setIsConfigDialogOpen(false);
      setEditingProviderId(null);
    },
    [deleteProvider]
  );

  const handleTestConnection = useCallback(
    async (config: ProviderConfigData) => {
      setIsTesting(true);
      try {
        return await testConnection(config);
      } finally {
        setIsTesting(false);
      }
    },
    [testConnection]
  );

  const handleForceSync = useCallback(async () => {
    if (syncContext) {
      await syncContext.forceSync();
    } else {
      await refreshAll();
    }
  }, [syncContext, refreshAll]);

  // Loading state
  if (isLoading) {
    return (
      <div className="space-y-6 p-6">
        <Skeleton className="h-8 w-48" />
        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
          <Skeleton className="h-32" />
          <Skeleton className="h-32" />
          <Skeleton className="h-32" />
        </div>
        <Skeleton className="h-64" />
      </div>
    );
  }

  // Error state
  if (error && !providers.length) {
    return (
      <div className="flex flex-col items-center justify-center p-12 text-center">
        <AlertTriangle className="h-12 w-12 text-destructive mb-4" />
        <h2 className="text-lg font-semibold mb-2">Failed to Load Providers</h2>
        <p className="text-muted-foreground mb-4">
          {error instanceof Error ? error.message : String(error)}
        </p>
        <Button onClick={handleForceSync}>
          <RefreshCw className="h-4 w-4 mr-2" />
          Retry
        </Button>
      </div>
    );
  }

  // Calculate stats
  const availableCount = providers.filter((p) => p.status === "available").length;
  const configuredCount = providers.filter((p) => p.hasApiKey).length;
  const deprecatedModelsCount = syncModels.filter(
    (m) => m.deprecation_status !== ModelDeprecationStatus.ACTIVE
  ).length;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight">AI Providers</h1>
          <p className="text-muted-foreground text-sm">
            Configure and manage your AI provider connections
          </p>
        </div>
        <div className="flex items-center gap-3">
          {/* Sync status indicator */}
          {syncContext && (
            <SyncStatusIndicator
              status={syncContext.status}
              isConnected={syncContext.isConnected}
              isLoading={syncContext.isLoading}
              lastSyncTime={syncContext.lastSyncTime}
              version={syncContext.version}
              error={syncContext.error}
              onSync={handleForceSync}
              compact
            />
          )}

          {/* Legacy WebSocket status (fallback) */}
          {!syncContext && (
            <Badge
              variant={isWebSocketConnected ? "default" : "secondary"}
              className="gap-1.5"
              title={legacyWsError || (isWebSocketConnected ? "Real-time updates active" : "Using polling for updates")}
            >
              {isWebSocketConnected ? (
                <>
                  <Wifi className="h-3 w-3" />
                  Live
                </>
              ) : (
                <>
                  <WifiOff className="h-3 w-3" />
                  Polling
                </>
              )}
            </Badge>
          )}

          {/* Refresh button */}
          <Button
            variant="outline"
            size="sm"
            onClick={handleForceSync}
            disabled={isRefreshing || syncContext?.isLoading}
          >
            <RefreshCw
              className={cn(
                "h-4 w-4 mr-2",
                (isRefreshing || syncContext?.isLoading) && "animate-spin"
              )}
            />
            Refresh
          </Button>
        </div>
      </div>

      {/* Quick Stats */}
      <div className="grid gap-4 md:grid-cols-4">
        <Card>
          <CardHeader className="pb-2">
            <CardDescription>Active Provider</CardDescription>
            <CardTitle className="text-lg flex items-center gap-2">
              {activeProvider ? (
                <>
                  <Zap className="h-4 w-4 text-primary" />
                  {activeProvider.name}
                </>
              ) : (
                <span className="text-muted-foreground">None selected</span>
              )}
            </CardTitle>
          </CardHeader>
          <CardContent>
            {activeProvider && (
              <div className="flex items-center gap-2 text-xs text-muted-foreground">
                <Badge
                  variant={
                    activeProvider.status === "available"
                      ? "default"
                      : activeProvider.status === "degraded"
                      ? "secondary"
                      : "destructive"
                  }
                  className="text-[10px]"
                >
                  {activeProvider.status}
                </Badge>
                {activeProvider.responseTime && (
                  <span>{activeProvider.responseTime}ms avg</span>
                )}
              </div>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <CardDescription>Available Providers</CardDescription>
            <CardTitle className="text-lg flex items-center gap-2">
              <Activity className="h-4 w-4 text-emerald-500" />
              {availableCount} / {providers.length}
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-xs text-muted-foreground">
              {providers.length - availableCount > 0 && (
                <span className="text-amber-500">
                  {providers.length - availableCount} unavailable
                </span>
              )}
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <CardDescription>Configured</CardDescription>
            <CardTitle className="text-lg flex items-center gap-2">
              <CheckCircle2 className="h-4 w-4 text-emerald-500" />
              {configuredCount} providers
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-xs text-muted-foreground">
              {providers.length - configuredCount > 0 && (
                <span>
                  {providers.length - configuredCount} need API keys
                </span>
              )}
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <CardDescription>Models</CardDescription>
            <CardTitle className="text-lg flex items-center gap-2">
              <Sparkles className="h-4 w-4 text-purple-500" />
              {syncModels.length} available
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-xs text-muted-foreground">
              {deprecatedModelsCount > 0 && (
                <span className="text-amber-500">
                  {deprecatedModelsCount} deprecated
                </span>
              )}
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Deprecation alerts (if any) */}
      {deprecatedModelsCount > 0 && (
        <DeprecationAlertsPanel
          models={syncModels}
          onSelectModel={handleModelChange}
        />
      )}

      {/* Provider unavailable warnings */}
      {useSyncSystem && syncProviders
        .filter((p) => p.health?.status === ProviderStatus.UNAVAILABLE && p.id === activeProviderId)
        .map((provider) => (
          <ProviderUnavailableWarning
            key={provider.id}
            providerId={provider.id}
            providerName={provider.display_name}
            status={provider.health?.status || "unknown"}
            errorMessage={provider.health?.error_message}
            fallbackProviderId={provider.is_fallback ? undefined : syncProviders.find((p) => p.is_fallback)?.id}
            fallbackProviderName={syncProviders.find((p) => p.is_fallback)?.display_name}
            onSelectFallback={handleProviderChange}
          />
        ))}

      {/* Main Content */}
      <Tabs value={selectedTab} onValueChange={setSelectedTab} className="space-y-4">
        <TabsList>
          <TabsTrigger value="selector" className="gap-2">
            <Zap className="h-3.5 w-3.5" />
            Quick Select
          </TabsTrigger>
          <TabsTrigger value="manage" className="gap-2">
            <Settings className="h-3.5 w-3.5" />
            Manage Providers
          </TabsTrigger>
        </TabsList>

        {/* Quick Select Tab */}
        <TabsContent value="selector" className="space-y-4">
          <div className="grid gap-4 md:grid-cols-2">
            {/* Provider Selector */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Server className="h-5 w-5" />
                  Select Provider
                </CardTitle>
                <CardDescription>
                  Choose your AI provider for generation tasks
                </CardDescription>
              </CardHeader>
              <CardContent>
                <ProviderSelector
                  providers={providers}
                  activeProviderId={activeProviderId || undefined}
                  onProviderChange={handleProviderChange}
                  isLoading={isLoading}
                />
              </CardContent>
            </Card>

            {/* Model Selector */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Cpu className="h-5 w-5" />
                  Select Model
                </CardTitle>
                <CardDescription>
                  Choose a model from the active provider
                </CardDescription>
              </CardHeader>
              <CardContent>
                {activeProvider ? (
                  <ModelSelector
                    models={activeProviderModels}
                    selectedModelId={activeModelId || undefined}
                    onModelSelect={handleModelChange}
                    disabled={isLoading}
                    showCapabilities={true}
                  />
                ) : (
                  <div className="text-sm text-muted-foreground">
                    Select a provider first to see available models
                  </div>
                )}
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        {/* Manage Providers Tab */}
        <TabsContent value="manage" className="space-y-4">
          <Card>
            <CardHeader className="flex flex-row items-center justify-between">
              <div>
                <CardTitle>Provider Configuration</CardTitle>
                <CardDescription>
                  Configure and manage your AI providers
                </CardDescription>
              </div>
              <Button onClick={handleAddProvider} className="gap-2">
                <Settings className="h-4 w-4" />
                Add Provider
              </Button>
            </CardHeader>
            <CardContent>
              <ProviderList
                providers={providers}
                onEdit={handleConfigureProvider}
                onAdd={() => {
                  setEditingProviderId(null);
                  setIsConfigDialogOpen(true);
                }}
                onRefresh={refreshAll}
                isRefreshing={isRefreshing}
              />
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>

      {/* Config Dialog */}
      {isConfigDialogOpen && (
        <ProviderConfigForm
          isOpen={isConfigDialogOpen}
          onClose={() => {
            setIsConfigDialogOpen(false);
            setEditingProviderId(null);
          }}
          onSave={handleSaveProvider}
          onDelete={editingProviderId ? () => handleDeleteProvider(editingProviderId) : undefined}
          initialData={editingProvider ? {
            id: editingProvider.id,
            name: editingProvider.name,
            type: editingProvider.type,
            baseUrl: editingProvider.baseUrl,
            enabled: editingProvider.enabled,
          } : undefined}
          isEditing={!!editingProviderId}
          isSaving={isSaving}
        />
      )}
    </div>
  );
}
