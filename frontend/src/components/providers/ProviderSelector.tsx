"use client";

import * as React from "react";
import { useCallback, useEffect, useState } from "react";
import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectLabel,
  SelectSeparator,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { ProviderType } from "@/lib/api/validation";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { cn } from "@/lib/utils";
import {
  Settings,
  RefreshCw,
  CheckCircle2,
  XCircle,
  AlertCircle,
  Loader2,
  Zap,
  Shield,
  Globe,
  Server,
  Cpu,
  Cloud,
} from "lucide-react";

// =============================================================================
// Types
// =============================================================================

export type ProviderStatus = "available" | "unavailable" | "degraded" | "unknown";

export interface ProviderModel {
  id: string;
  name: string;
  description?: string;
  contextWindow?: number;
  maxTokens?: number;
  capabilities?: string[];
}

export interface Provider {
  id: string;
  name: string;
  type: ProviderType;
  status: ProviderStatus;
  isActive: boolean;
  models: ProviderModel[];
  description?: string;
  baseUrl?: string;
  hasApiKey: boolean;
  enabled: boolean;
  lastHealthCheck?: string;
  responseTime?: number;
  errorRate?: number;
}

export interface ProviderSelectorProps {
  providers: Provider[];
  activeProviderId?: string;
  activeModelId?: string;
  isLoading?: boolean;
  isChangingProvider?: boolean;
  onProviderChange: (providerId: string) => void;
  onModelChange?: (modelId: string) => void;
  onConfigureProvider?: (providerId: string) => void;
  onRefreshProviders?: () => void;
  showModelSelector?: boolean;
  showStatusIndicator?: boolean;
  showConfigButton?: boolean;
  compact?: boolean;
  className?: string;
}

// =============================================================================
// Status Indicator Component
// =============================================================================

interface StatusIndicatorProps {
  status: ProviderStatus;
  size?: "sm" | "md" | "lg";
  showLabel?: boolean;
  className?: string;
}

function StatusIndicator({
  status,
  size = "md",
  showLabel = false,
  className,
}: StatusIndicatorProps) {
  const sizeClasses = {
    sm: "h-2 w-2",
    md: "h-2.5 w-2.5",
    lg: "h-3 w-3",
  };

  const statusConfig = {
    available: {
      color: "bg-emerald-500",
      pulse: "animate-pulse",
      label: "Available",
      icon: CheckCircle2,
    },
    unavailable: {
      color: "bg-red-500",
      pulse: "",
      label: "Unavailable",
      icon: XCircle,
    },
    degraded: {
      color: "bg-amber-500",
      pulse: "animate-pulse",
      label: "Degraded",
      icon: AlertCircle,
    },
    unknown: {
      color: "bg-gray-400",
      pulse: "",
      label: "Unknown",
      icon: AlertCircle,
    },
  };

  const config = statusConfig[status];
  const Icon = config.icon;

  return (
    <div className={cn("flex items-center gap-1.5", className)}>
      <span
        className={cn(
          "rounded-full",
          sizeClasses[size],
          config.color,
          config.pulse
        )}
        aria-label={config.label}
      />
      {showLabel && (
        <span className="text-xs text-muted-foreground">{config.label}</span>
      )}
    </div>
  );
}

// =============================================================================
// Provider Icon Component
// =============================================================================

interface ProviderIconProps {
  type: string;
  className?: string;
}

function ProviderIcon({ type, className }: ProviderIconProps) {
  const iconMap: Record<string, React.ElementType> = {
    openai: Zap,
    anthropic: Shield,
    google: Globe,
    gemini: Globe,
    deepseek: Cpu,
    qwen: Cloud,
    ollama: Server,
    local: Server,
    custom: Settings,
  };

  const Icon = iconMap[type.toLowerCase()] || Cloud;

  return <Icon className={cn("h-4 w-4", className)} />;
}

// =============================================================================
// Provider Card Component (for expanded view)
// =============================================================================

interface ProviderCardProps {
  provider: Provider;
  isSelected: boolean;
  onSelect: () => void;
  onConfigure?: () => void;
}

function ProviderCard({
  provider,
  isSelected,
  onSelect,
  onConfigure,
}: ProviderCardProps) {
  return (
    <Card
      className={cn(
        "cursor-pointer transition-all duration-200 hover:shadow-md",
        isSelected && "ring-2 ring-primary border-primary"
      )}
      onClick={onSelect}
    >
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <ProviderIcon type={provider.type} />
            <CardTitle className="text-sm font-medium">
              {provider.name}
            </CardTitle>
          </div>
          <StatusIndicator status={provider.status} size="sm" />
        </div>
        {provider.description && (
          <CardDescription className="text-xs">
            {provider.description}
          </CardDescription>
        )}
      </CardHeader>
      <CardContent className="pt-0">
        <div className="flex items-center justify-between">
          <div className="flex flex-wrap gap-1">
            <Badge variant="secondary" className="text-[10px]">
              {provider.models.length} models
            </Badge>
            {provider.hasApiKey && (
              <Badge variant="outline" className="text-[10px]">
                Configured
              </Badge>
            )}
          </div>
          {onConfigure && (
            <Button
              variant="ghost"
              size="sm"
              className="h-6 px-2"
              onClick={(e) => {
                e.stopPropagation();
                onConfigure();
              }}
            >
              <Settings className="h-3 w-3" />
            </Button>
          )}
        </div>
        {provider.responseTime !== undefined && (
          <div className="mt-2 text-[10px] text-muted-foreground">
            Avg response: {provider.responseTime}ms
          </div>
        )}
      </CardContent>
    </Card>
  );
}

// =============================================================================
// Main Provider Selector Component
// =============================================================================

export function ProviderSelector({
  providers,
  activeProviderId,
  activeModelId,
  isLoading = false,
  isChangingProvider = false,
  onProviderChange,
  onModelChange,
  onConfigureProvider,
  onRefreshProviders,
  showModelSelector = true,
  showStatusIndicator = true,
  showConfigButton = true,
  compact = false,
  className,
}: ProviderSelectorProps) {
  const [isDialogOpen, setIsDialogOpen] = useState(false);

  const activeProvider = providers.find((p) => p.id === activeProviderId);
  const activeModel = activeProvider?.models.find((m) => m.id === activeModelId);

  // Group providers by type
  const groupedProviders = React.useMemo(() => {
    const groups: Record<string, Provider[]> = {};
    providers.forEach((provider) => {
      const type = provider.type.toLowerCase();
      if (!groups[type]) {
        groups[type] = [];
      }
      groups[type].push(provider);
    });
    return groups;
  }, [providers]);

  const handleProviderSelect = useCallback(
    (providerId: string) => {
      onProviderChange(providerId);
      // Auto-select first model if model selector is enabled
      if (showModelSelector && onModelChange) {
        const provider = providers.find((p) => p.id === providerId);
        if (provider && provider.models.length > 0) {
          onModelChange(provider.models[0].id);
        }
      }
    },
    [onProviderChange, onModelChange, providers, showModelSelector]
  );

  // Loading state
  if (isLoading) {
    return (
      <div className={cn("flex items-center gap-2", className)}>
        <Skeleton className="h-9 w-[180px]" />
        {showModelSelector && <Skeleton className="h-9 w-[200px]" />}
      </div>
    );
  }

  // Compact mode - single select dropdown
  if (compact) {
    return (
      <div className={cn("flex items-center gap-2", className)}>
        <Select
          value={activeProviderId}
          onValueChange={handleProviderSelect}
          disabled={isChangingProvider}
        >
          <SelectTrigger className="w-[180px]">
            {isChangingProvider ? (
              <div className="flex items-center gap-2">
                <Loader2 className="h-4 w-4 animate-spin" />
                <span>Switching...</span>
              </div>
            ) : (
              <SelectValue placeholder="Select provider">
                {activeProvider && (
                  <div className="flex items-center gap-2">
                    {showStatusIndicator && (
                      <StatusIndicator status={activeProvider.status} size="sm" />
                    )}
                    <ProviderIcon type={activeProvider.type} className="h-3.5 w-3.5" />
                    <span className="truncate">{activeProvider.name}</span>
                  </div>
                )}
              </SelectValue>
            )}
          </SelectTrigger>
          <SelectContent>
            {Object.entries(groupedProviders).map(([type, typeProviders]) => (
              <SelectGroup key={type}>
                <SelectLabel className="capitalize">{type}</SelectLabel>
                {typeProviders.map((provider) => (
                  <SelectItem
                    key={provider.id}
                    value={provider.id}
                    disabled={provider.status === "unavailable"}
                  >
                    <div className="flex items-center gap-2">
                      <StatusIndicator status={provider.status} size="sm" />
                      <ProviderIcon type={provider.type} className="h-3.5 w-3.5" />
                      <span>{provider.name}</span>
                      {!provider.hasApiKey && (
                        <Badge variant="outline" className="ml-auto text-[10px]">
                          No key
                        </Badge>
                      )}
                    </div>
                  </SelectItem>
                ))}
                <SelectSeparator />
              </SelectGroup>
            ))}
          </SelectContent>
        </Select>

        {/* Model selector */}
        {showModelSelector && activeProvider && (
          <Select
            value={activeModelId}
            onValueChange={onModelChange}
            disabled={isChangingProvider || !activeProvider.models.length}
          >
            <SelectTrigger className="w-[200px]">
              <SelectValue placeholder="Select model">
                {activeModel && (
                  <span className="truncate">{activeModel.name}</span>
                )}
              </SelectValue>
            </SelectTrigger>
            <SelectContent>
              {activeProvider.models.map((model) => (
                <SelectItem key={model.id} value={model.id}>
                  <div className="flex flex-col">
                    <span>{model.name}</span>
                    {model.contextWindow && (
                      <span className="text-[10px] text-muted-foreground">
                        {(model.contextWindow / 1000).toFixed(0)}K context
                      </span>
                    )}
                  </div>
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        )}

        {/* Config button */}
        {showConfigButton && onConfigureProvider && activeProviderId && (
          <Button
            variant="ghost"
            size="icon"
            className="h-9 w-9"
            onClick={() => onConfigureProvider(activeProviderId)}
          >
            <Settings className="h-4 w-4" />
          </Button>
        )}

        {/* Refresh button */}
        {onRefreshProviders && (
          <Button
            variant="ghost"
            size="icon"
            className="h-9 w-9"
            onClick={onRefreshProviders}
            disabled={isChangingProvider}
          >
            <RefreshCw
              className={cn("h-4 w-4", isChangingProvider && "animate-spin")}
            />
          </Button>
        )}
      </div>
    );
  }

  // Full mode - with dialog for expanded view
  return (
    <div className={cn("flex items-center gap-2", className)}>
      {/* Provider selector */}
      <Select
        value={activeProviderId}
        onValueChange={handleProviderSelect}
        disabled={isChangingProvider}
      >
        <SelectTrigger className="w-[200px]">
          {isChangingProvider ? (
            <div className="flex items-center gap-2">
              <Loader2 className="h-4 w-4 animate-spin" />
              <span>Switching...</span>
            </div>
          ) : (
            <SelectValue placeholder="Select provider">
              {activeProvider && (
                <div className="flex items-center gap-2">
                  {showStatusIndicator && (
                    <StatusIndicator status={activeProvider.status} size="sm" />
                  )}
                  <ProviderIcon type={activeProvider.type} className="h-3.5 w-3.5" />
                  <span className="truncate">{activeProvider.name}</span>
                </div>
              )}
            </SelectValue>
          )}
        </SelectTrigger>
        <SelectContent>
          {Object.entries(groupedProviders).map(([type, typeProviders]) => (
            <SelectGroup key={type}>
              <SelectLabel className="capitalize">{type}</SelectLabel>
              {typeProviders.map((provider) => (
                <SelectItem
                  key={provider.id}
                  value={provider.id}
                  disabled={provider.status === "unavailable"}
                >
                  <div className="flex items-center gap-2 w-full">
                    <StatusIndicator status={provider.status} size="sm" />
                    <ProviderIcon type={provider.type} className="h-3.5 w-3.5" />
                    <span className="flex-1">{provider.name}</span>
                    {!provider.hasApiKey && (
                      <Badge variant="outline" className="text-[10px]">
                        No key
                      </Badge>
                    )}
                  </div>
                </SelectItem>
              ))}
              <SelectSeparator />
            </SelectGroup>
          ))}
        </SelectContent>
      </Select>

      {/* Model selector */}
      {showModelSelector && activeProvider && (
        <Select
          value={activeModelId}
          onValueChange={onModelChange}
          disabled={isChangingProvider || !activeProvider.models.length}
        >
          <SelectTrigger className="w-[220px]">
            <SelectValue placeholder="Select model">
              {activeModel && (
                <div className="flex items-center gap-2">
                  <span className="truncate">{activeModel.name}</span>
                  {activeModel.contextWindow && (
                    <Badge variant="secondary" className="text-[10px]">
                      {(activeModel.contextWindow / 1000).toFixed(0)}K
                    </Badge>
                  )}
                </div>
              )}
            </SelectValue>
          </SelectTrigger>
          <SelectContent>
            {activeProvider.models.map((model) => (
              <SelectItem key={model.id} value={model.id}>
                <div className="flex flex-col gap-0.5">
                  <div className="flex items-center gap-2">
                    <span>{model.name}</span>
                    {model.contextWindow && (
                      <Badge variant="secondary" className="text-[10px]">
                        {(model.contextWindow / 1000).toFixed(0)}K
                      </Badge>
                    )}
                  </div>
                  {model.description && (
                    <span className="text-[10px] text-muted-foreground">
                      {model.description}
                    </span>
                  )}
                </div>
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      )}

      {/* Expanded view dialog */}
      <Dialog open={isDialogOpen} onOpenChange={setIsDialogOpen}>
        <DialogTrigger asChild>
          <Button variant="outline" size="icon" className="h-9 w-9">
            <Settings className="h-4 w-4" />
          </Button>
        </DialogTrigger>
        <DialogContent className="max-w-2xl">
          <DialogHeader>
            <DialogTitle>AI Provider Settings</DialogTitle>
            <DialogDescription>
              Select and configure your AI provider. Changes take effect immediately.
            </DialogDescription>
          </DialogHeader>

          <div className="grid grid-cols-2 gap-3 py-4 max-h-[400px] overflow-y-auto">
            {providers.map((provider) => (
              <ProviderCard
                key={provider.id}
                provider={provider}
                isSelected={provider.id === activeProviderId}
                onSelect={() => {
                  handleProviderSelect(provider.id);
                  setIsDialogOpen(false);
                }}
                onConfigure={
                  onConfigureProvider
                    ? () => {
                        onConfigureProvider(provider.id);
                        setIsDialogOpen(false);
                      }
                    : undefined
                }
              />
            ))}
          </div>

          <DialogFooter>
            <Button variant="outline" onClick={() => setIsDialogOpen(false)}>
              Close
            </Button>
            {onRefreshProviders && (
              <Button
                variant="secondary"
                onClick={onRefreshProviders}
                disabled={isChangingProvider}
              >
                <RefreshCw
                  className={cn(
                    "h-4 w-4 mr-2",
                    isChangingProvider && "animate-spin"
                  )}
                />
                Refresh Status
              </Button>
            )}
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}

// =============================================================================
// Provider Status Badge Component (for use elsewhere)
// =============================================================================

export interface ProviderStatusBadgeProps {
  provider: Provider;
  showResponseTime?: boolean;
  className?: string;
}

export function ProviderStatusBadge({
  provider,
  showResponseTime = false,
  className,
}: ProviderStatusBadgeProps) {
  const statusVariant = {
    available: "default" as const,
    unavailable: "destructive" as const,
    degraded: "secondary" as const,
    unknown: "outline" as const,
  };

  return (
    <Badge
      variant={statusVariant[provider.status]}
      className={cn("gap-1.5", className)}
    >
      <StatusIndicator status={provider.status} size="sm" />
      <ProviderIcon type={provider.type} className="h-3 w-3" />
      <span>{provider.name}</span>
      {showResponseTime && provider.responseTime !== undefined && (
        <span className="text-[10px] opacity-70">
          {provider.responseTime}ms
        </span>
      )}
    </Badge>
  );
}

export default ProviderSelector;
