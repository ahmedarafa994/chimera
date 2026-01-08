/**
 * CascadingProviderModelSelector Component
 *
 * A two-dropdown cascading selector for provider and model selection.
 * Features:
 * - Provider dropdown that loads dynamically from the API
 * - Cascading model dropdown that updates when provider changes
 * - Loading states for both dropdowns
 * - Provider health status indicators
 * - Model capability badges
 * - Keyboard navigation support
 * - Mobile-responsive design
 *
 * @module components/providers/CascadingProviderModelSelector
 */

"use client";

import * as React from "react";
import { cn } from "@/lib/utils";
import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectLabel,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Skeleton } from "@/components/ui/skeleton";
import { Button } from "@/components/ui/button";
import { ProviderStatusBadge, ProviderStatusIndicator } from "./ProviderStatusBadge";
import { ModelCapabilityBadges, CompactCapabilityIcons } from "./ModelCapabilityBadges";
import { useUnifiedProviderSelection } from "@/hooks/useUnifiedProviderSelection";
import type {
  CascadingProviderModelSelectorProps,
  ProviderInfo,
  ModelInfo,
} from "@/types/unified-providers";

// =============================================================================
// Icons
// =============================================================================

function RefreshIcon({ className }: { className?: string }) {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      className={className}
    >
      <path d="M21 12a9 9 0 0 0-9-9 9.75 9.75 0 0 0-6.74 2.74L3 8" />
      <path d="M3 3v5h5" />
      <path d="M3 12a9 9 0 0 0 9 9 9.75 9.75 0 0 0 6.74-2.74L21 16" />
      <path d="M16 21h5v-5" />
    </svg>
  );
}

function ChevronDownIcon({ className }: { className?: string }) {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      className={className}
    >
      <path d="m6 9 6 6 6-6" />
    </svg>
  );
}

function ServerIcon({ className }: { className?: string }) {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      className={className}
    >
      <rect x="2" y="2" width="20" height="8" rx="2" ry="2" />
      <rect x="2" y="14" width="20" height="8" rx="2" ry="2" />
      <line x1="6" y1="6" x2="6.01" y2="6" />
      <line x1="6" y1="18" x2="6.01" y2="18" />
    </svg>
  );
}

function CpuIcon({ className }: { className?: string }) {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      className={className}
    >
      <rect x="4" y="4" width="16" height="16" rx="2" ry="2" />
      <rect x="9" y="9" width="6" height="6" />
      <line x1="9" y1="1" x2="9" y2="4" />
      <line x1="15" y1="1" x2="15" y2="4" />
      <line x1="9" y1="20" x2="9" y2="23" />
      <line x1="15" y1="20" x2="15" y2="23" />
      <line x1="20" y1="9" x2="23" y2="9" />
      <line x1="20" y1="14" x2="23" y2="14" />
      <line x1="1" y1="9" x2="4" y2="9" />
      <line x1="1" y1="14" x2="4" y2="14" />
    </svg>
  );
}

function AlertCircleIcon({ className }: { className?: string }) {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      className={className}
    >
      <circle cx="12" cy="12" r="10" />
      <line x1="12" y1="8" x2="12" y2="12" />
      <line x1="12" y1="16" x2="12.01" y2="16" />
    </svg>
  );
}

// =============================================================================
// Provider Item Component
// =============================================================================

interface ProviderItemProps {
  provider: ProviderInfo;
  showStatus?: boolean;
}

function ProviderItem({ provider, showStatus = true }: ProviderItemProps) {
  return (
    <div className="flex items-center justify-between w-full gap-2">
      <div className="flex items-center gap-2 min-w-0 flex-1">
        <ServerIcon className="h-4 w-4 flex-shrink-0 text-muted-foreground" />
        <span className="truncate">{provider.display_name}</span>
        {provider.model_count > 0 && (
          <span className="text-xs text-muted-foreground">
            ({provider.model_count})
          </span>
        )}
      </div>
      {showStatus && (
        <ProviderStatusIndicator
          status={provider.status}
          className="flex-shrink-0"
        />
      )}
    </div>
  );
}

// =============================================================================
// Model Item Component
// =============================================================================

interface ModelItemProps {
  model: ModelInfo;
  showCapabilities?: boolean;
}

function ModelItem({ model, showCapabilities = true }: ModelItemProps) {
  return (
    <div className="flex flex-col gap-1 w-full">
      <div className="flex items-center justify-between gap-2">
        <div className="flex items-center gap-2 min-w-0 flex-1">
          <CpuIcon className="h-4 w-4 flex-shrink-0 text-muted-foreground" />
          <span className="truncate font-medium">{model.name}</span>
          {model.is_default && (
            <span className="text-[10px] px-1.5 py-0.5 rounded-full bg-primary/10 text-primary font-medium">
              Default
            </span>
          )}
        </div>
        {model.context_length && (
          <span className="text-xs text-muted-foreground flex-shrink-0">
            {Math.round(model.context_length / 1000)}K ctx
          </span>
        )}
      </div>
      {showCapabilities && model.capabilities.length > 0 && (
        <CompactCapabilityIcons
          capabilities={model.capabilities}
          maxVisible={4}
          className="ml-6"
        />
      )}
    </div>
  );
}

// =============================================================================
// Loading Skeleton
// =============================================================================

function SelectSkeleton() {
  return (
    <div className="flex items-center gap-2 p-2">
      <Skeleton className="h-4 w-4 rounded" />
      <Skeleton className="h-4 flex-1" />
    </div>
  );
}

// =============================================================================
// Error State
// =============================================================================

interface ErrorStateProps {
  message: string;
  onRetry?: () => void;
}

function ErrorState({ message, onRetry }: ErrorStateProps) {
  return (
    <div className="flex items-center gap-2 p-3 text-destructive">
      <AlertCircleIcon className="h-4 w-4 flex-shrink-0" />
      <span className="text-sm flex-1">{message}</span>
      {onRetry && (
        <Button
          variant="ghost"
          size="sm"
          onClick={onRetry}
          className="h-7 px-2"
        >
          <RefreshIcon className="h-3 w-3 mr-1" />
          Retry
        </Button>
      )}
    </div>
  );
}

// =============================================================================
// Main Component
// =============================================================================

export function CascadingProviderModelSelector({
  onSelectionChange,
  defaultProvider,
  defaultModel,
  disabled = false,
  showStatus = true,
  showCapabilities = true,
  className,
  sessionId,
  providerPlaceholder = "Select a provider",
  modelPlaceholder = "Select a model",
}: CascadingProviderModelSelectorProps) {
  // Wrap the callback to handle null values
  const handleSelectionChange = React.useCallback(
    (provider: string | null, model: string | null) => {
      if (provider && model && onSelectionChange) {
        onSelectionChange(provider, model);
      }
    },
    [onSelectionChange]
  );

  // Use the unified provider selection hook
  const {
    providers,
    models,
    selectedProvider,
    selectedModel,
    isLoadingProviders,
    isLoadingModels,
    isSaving,
    providersError,
    modelsError,
    selectProvider,
    selectModel,
    refreshProviders,
    refreshModels,
    providerStatus,
    connectionStatus,
  } = useUnifiedProviderSelection({
    sessionId,
    enableWebSocket: true,
    autoRestoreSelection: !defaultProvider, // Don't auto-restore if defaults provided
    onSelectionChange: handleSelectionChange,
  });

  // Handle default values
  React.useEffect(() => {
    if (defaultProvider && !selectedProvider && providers.length > 0) {
      const providerExists = providers.some(
        (p) => p.provider_id === defaultProvider
      );
      if (providerExists) {
        selectProvider(defaultProvider);
      }
    }
  }, [defaultProvider, selectedProvider, providers, selectProvider]);

  React.useEffect(() => {
    if (defaultModel && !selectedModel && models.length > 0 && selectedProvider) {
      const modelExists = models.some((m) => m.model_id === defaultModel);
      if (modelExists) {
        selectModel(defaultModel);
      }
    }
  }, [defaultModel, selectedModel, models, selectedProvider, selectModel]);

  // Handle provider change
  const handleProviderChange = React.useCallback(
    (value: string) => {
      selectProvider(value);
    },
    [selectProvider]
  );

  // Handle model change
  const handleModelChange = React.useCallback(
    (value: string) => {
      selectModel(value);
    },
    [selectModel]
  );

  // Get current provider info for display
  const currentProvider = React.useMemo(
    () => providers.find((p) => p.provider_id === selectedProvider),
    [providers, selectedProvider]
  );

  // Get current model info for display
  const currentModel = React.useMemo(
    () => models.find((m) => m.model_id === selectedModel),
    [models, selectedModel]
  );

  // Connection status indicator
  const showConnectionWarning = connectionStatus === "disconnected";

  return (
    <div className={cn("flex flex-col gap-3", className)}>
      {/* Connection Warning */}
      {showConnectionWarning && (
        <div className="flex items-center gap-2 text-xs text-amber-600 dark:text-amber-400 bg-amber-50 dark:bg-amber-950/20 rounded-md px-3 py-2">
          <AlertCircleIcon className="h-3.5 w-3.5" />
          <span>Real-time sync disconnected. Changes may not sync immediately.</span>
        </div>
      )}

      {/* Provider Selector */}
      <div className="space-y-1.5">
        <div className="flex items-center justify-between">
          <label className="text-sm font-medium text-foreground">
            Provider
          </label>
          <Button
            variant="ghost"
            size="sm"
            onClick={() => refreshProviders()}
            disabled={isLoadingProviders || disabled}
            className="h-6 px-2 text-xs"
          >
            <RefreshIcon
              className={cn(
                "h-3 w-3 mr-1",
                isLoadingProviders && "animate-spin"
              )}
            />
            Refresh
          </Button>
        </div>

        {providersError ? (
          <ErrorState
            message="Failed to load providers"
            onRetry={refreshProviders}
          />
        ) : (
          <Select
            value={selectedProvider || ""}
            onValueChange={handleProviderChange}
            disabled={disabled || isLoadingProviders}
          >
            <SelectTrigger className="w-full">
              {isLoadingProviders ? (
                <SelectSkeleton />
              ) : selectedProvider && currentProvider ? (
                <div className="flex items-center gap-2">
                  <ProviderStatusIndicator
                    status={providerStatus[selectedProvider] || currentProvider.status}
                  />
                  <span>{currentProvider.display_name}</span>
                </div>
              ) : (
                <SelectValue placeholder={providerPlaceholder} />
              )}
            </SelectTrigger>
            <SelectContent>
              <SelectGroup>
                <SelectLabel>Available Providers</SelectLabel>
                {providers
                  .filter((p) => p.is_available)
                  .map((provider) => (
                    <SelectItem
                      key={provider.provider_id}
                      value={provider.provider_id}
                    >
                      <ProviderItem
                        provider={{
                          ...provider,
                          status: providerStatus[provider.provider_id] || provider.status,
                        }}
                        showStatus={showStatus}
                      />
                    </SelectItem>
                  ))}
              </SelectGroup>

              {providers.some((p) => !p.is_available) && (
                <SelectGroup>
                  <SelectLabel className="text-muted-foreground">
                    Unavailable
                  </SelectLabel>
                  {providers
                    .filter((p) => !p.is_available)
                    .map((provider) => (
                      <SelectItem
                        key={provider.provider_id}
                        value={provider.provider_id}
                        disabled
                      >
                        <ProviderItem
                          provider={provider}
                          showStatus={showStatus}
                        />
                      </SelectItem>
                    ))}
                </SelectGroup>
              )}
            </SelectContent>
          </Select>
        )}

        {/* Provider Status Badge (when selected) */}
        {currentProvider && showStatus && (
          <div className="flex items-center gap-2 mt-1">
            <ProviderStatusBadge
              status={providerStatus[selectedProvider!] || currentProvider.status}
              healthScore={currentProvider.health_score}
              compact
            />
            {currentProvider.capabilities.length > 0 && (
              <span className="text-xs text-muted-foreground">
                {currentProvider.capabilities.slice(0, 3).join(", ")}
                {currentProvider.capabilities.length > 3 && "..."}
              </span>
            )}
          </div>
        )}
      </div>

      {/* Model Selector */}
      <div className="space-y-1.5">
        <div className="flex items-center justify-between">
          <label className="text-sm font-medium text-foreground">
            Model
          </label>
          {selectedProvider && (
            <Button
              variant="ghost"
              size="sm"
              onClick={() => refreshModels()}
              disabled={isLoadingModels || disabled || !selectedProvider}
              className="h-6 px-2 text-xs"
            >
              <RefreshIcon
                className={cn(
                  "h-3 w-3 mr-1",
                  isLoadingModels && "animate-spin"
                )}
              />
              Refresh
            </Button>
          )}
        </div>

        {modelsError ? (
          <ErrorState
            message="Failed to load models"
            onRetry={refreshModels}
          />
        ) : (
          <Select
            value={selectedModel || ""}
            onValueChange={handleModelChange}
            disabled={disabled || isLoadingModels || !selectedProvider || isSaving}
          >
            <SelectTrigger className="w-full">
              {isLoadingModels ? (
                <SelectSkeleton />
              ) : selectedModel && currentModel ? (
                <div className="flex items-center gap-2">
                  <CpuIcon className="h-4 w-4 text-muted-foreground" />
                  <span>{currentModel.name}</span>
                  {currentModel.is_default && (
                    <span className="text-[10px] px-1.5 py-0.5 rounded-full bg-primary/10 text-primary">
                      Default
                    </span>
                  )}
                </div>
              ) : !selectedProvider ? (
                <span className="text-muted-foreground">
                  Select a provider first
                </span>
              ) : (
                <SelectValue placeholder={modelPlaceholder} />
              )}
            </SelectTrigger>
            <SelectContent>
              {models.length === 0 ? (
                <div className="p-4 text-center text-sm text-muted-foreground">
                  No models available for this provider
                </div>
              ) : (
                <SelectGroup>
                  <SelectLabel>
                    {currentProvider?.display_name || "Provider"} Models
                  </SelectLabel>
                  {models.map((model) => (
                    <SelectItem
                      key={model.model_id}
                      value={model.model_id}
                      disabled={!model.is_available}
                    >
                      <ModelItem
                        model={model}
                        showCapabilities={showCapabilities}
                      />
                    </SelectItem>
                  ))}
                </SelectGroup>
              )}
            </SelectContent>
          </Select>
        )}

        {/* Model Capabilities (when selected) */}
        {currentModel && showCapabilities && currentModel.capabilities.length > 0 && (
          <div className="mt-2">
            <ModelCapabilityBadges
              capabilities={currentModel.capabilities}
              maxVisible={4}
              size="sm"
            />
          </div>
        )}

        {/* Model Details */}
        {currentModel && (
          <div className="flex items-center gap-3 mt-1 text-xs text-muted-foreground">
            {currentModel.context_length && (
              <span>Context: {Math.round(currentModel.context_length / 1000)}K</span>
            )}
            {currentModel.max_output_tokens && (
              <span>Max Output: {Math.round(currentModel.max_output_tokens / 1000)}K</span>
            )}
            {currentModel.pricing && (
              <span>
                ${currentModel.pricing.input.toFixed(4)}/1K in,
                ${currentModel.pricing.output.toFixed(4)}/1K out
              </span>
            )}
          </div>
        )}
      </div>

      {/* Saving Indicator */}
      {isSaving && (
        <div className="flex items-center gap-2 text-xs text-muted-foreground">
          <RefreshIcon className="h-3 w-3 animate-spin" />
          <span>Saving selection...</span>
        </div>
      )}
    </div>
  );
}

// =============================================================================
// Compact Version
// =============================================================================

interface CompactProviderModelSelectorProps {
  onSelectionChange?: (provider: string, model: string) => void;
  disabled?: boolean;
  className?: string;
  sessionId?: string;
}

export function CompactProviderModelSelector({
  onSelectionChange,
  disabled = false,
  className,
  sessionId,
}: CompactProviderModelSelectorProps) {
  // Wrap the callback to handle null values
  const handleSelectionChange = React.useCallback(
    (provider: string | null, model: string | null) => {
      if (provider && model && onSelectionChange) {
        onSelectionChange(provider, model);
      }
    },
    [onSelectionChange]
  );

  const {
    providers,
    models,
    selectedProvider,
    selectedModel,
    isLoadingProviders,
    isLoadingModels,
    selectProvider,
    selectModel,
    providerStatus,
  } = useUnifiedProviderSelection({
    sessionId,
    onSelectionChange: handleSelectionChange,
  });

  const currentProvider = providers.find((p) => p.provider_id === selectedProvider);
  const currentModel = models.find((m) => m.model_id === selectedModel);

  return (
    <div className={cn("flex items-center gap-2", className)}>
      {/* Provider */}
      <Select
        value={selectedProvider || ""}
        onValueChange={selectProvider}
        disabled={disabled || isLoadingProviders}
      >
        <SelectTrigger className="w-[160px]">
          {isLoadingProviders ? (
            <Skeleton className="h-4 w-full" />
          ) : currentProvider ? (
            <div className="flex items-center gap-1.5">
              <ProviderStatusIndicator
                status={providerStatus[selectedProvider!] || currentProvider.status}
              />
              <span className="truncate">{currentProvider.display_name}</span>
            </div>
          ) : (
            <span className="text-muted-foreground">Provider</span>
          )}
        </SelectTrigger>
        <SelectContent>
          {providers
            .filter((p) => p.is_available)
            .map((provider) => (
              <SelectItem key={provider.provider_id} value={provider.provider_id}>
                <div className="flex items-center gap-1.5">
                  <ProviderStatusIndicator
                    status={providerStatus[provider.provider_id] || provider.status}
                  />
                  <span>{provider.display_name}</span>
                </div>
              </SelectItem>
            ))}
        </SelectContent>
      </Select>

      <ChevronDownIcon className="h-4 w-4 text-muted-foreground rotate-[-90deg]" />

      {/* Model */}
      <Select
        value={selectedModel || ""}
        onValueChange={selectModel}
        disabled={disabled || isLoadingModels || !selectedProvider}
      >
        <SelectTrigger className="w-[200px]">
          {isLoadingModels ? (
            <Skeleton className="h-4 w-full" />
          ) : currentModel ? (
            <span className="truncate">{currentModel.name}</span>
          ) : (
            <span className="text-muted-foreground">Model</span>
          )}
        </SelectTrigger>
        <SelectContent>
          {models.map((model) => (
            <SelectItem key={model.model_id} value={model.model_id}>
              <div className="flex items-center gap-2">
                <span>{model.name}</span>
                {model.is_default && (
                  <span className="text-[10px] text-primary">Default</span>
                )}
              </div>
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
    </div>
  );
}

export default CascadingProviderModelSelector;
