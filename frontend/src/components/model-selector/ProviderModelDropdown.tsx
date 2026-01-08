"use client";

/**
 * Provider Model Dropdown Component
 *
 * A dropdown menu component for selecting AI models organized by provider.
 * Features:
 * - Models grouped by provider (OpenAI, Anthropic, Google, etc.)
 * - Visual provider icons and status indicators
 * - Loading and error states
 * - Compact and full-size variants
 * - Keyboard navigation support
 */

import React, { useState, useEffect, useCallback, useMemo } from "react";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuGroup,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";
import {
  ChevronDown,
  Check,
  Loader2,
  AlertCircle,
  RefreshCw,
  Cpu,
  Sparkles,
  Brain,
  Bot,
  Zap,
  Search,
  Server,
} from "lucide-react";
import { toast } from "sonner";

// ============================================================================
// Types
// ============================================================================

export interface Provider {
  provider: string;
  status: string;
  model?: string;
  available_models: string[];
}

export interface ProvidersListResponse {
  providers: Provider[];
  default: string;
  count: number;
}

export interface ProviderModelDropdownProps {
  /** Currently selected provider */
  selectedProvider?: string;
  /** Currently selected model */
  selectedModel?: string;
  /** Callback when selection changes */
  onSelectionChange?: (provider: string, model: string) => void;
  /** Compact mode for smaller UI spaces */
  compact?: boolean;
  /** Disable the dropdown */
  disabled?: boolean;
  /** Show refresh button */
  showRefresh?: boolean;
  /** Custom placeholder text */
  placeholder?: string;
  /** Additional CSS classes */
  className?: string;
  /** Align dropdown content */
  align?: "start" | "center" | "end";
  /** Side of trigger to show dropdown */
  side?: "top" | "right" | "bottom" | "left";
}

interface ProviderGroup {
  provider: string;
  displayName: string;
  icon: React.ReactNode;
  status: "active" | "inactive" | "unknown";
  models: string[];
  defaultModel?: string;
}

const FALLBACK_PROVIDERS: Provider[] = [
  {
    provider: "google",
    status: "unknown",
    model: "gemini-2.5-flash",
    available_models: [
      "gemini-3-pro-preview",
      "gemini-3-pro-image-preview",
      "gemini-2.5-pro",
      "gemini-2.5-pro-preview-06-05",
      "gemini-2.5-flash",
      "gemini-2.5-flash-lite",
      "gemini-2.5-flash-image",
      "gemini-2.0-flash",
      "gemini-2.0-flash-lite",
      "gemini-1.5-pro",
      "gemini-1.5-flash",
      "gemini-1.5-flash-8b",
    ],
  },
  {
    provider: "gemini",
    status: "unknown",
    model: "gemini-2.5-flash",
    available_models: [
      "gemini-3-pro-preview",
      "gemini-3-pro-image-preview",
      "gemini-2.5-pro",
      "gemini-2.5-pro-preview-06-05",
      "gemini-2.5-flash",
      "gemini-2.5-flash-lite",
      "gemini-2.5-flash-image",
      "gemini-2.0-flash",
      "gemini-2.0-flash-lite",
      "gemini-1.5-pro",
      "gemini-1.5-flash",
      "gemini-1.5-flash-8b",
    ],
  },
  {
    provider: "gemini-cli",
    status: "unknown",
    model: "gemini-3-pro-preview",
    available_models: ["gemini-3-pro-preview", "gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.5-flash-lite"],
  },
  {
    provider: "antigravity",
    status: "unknown",
    model: "gemini-claude-sonnet-4-5",
    available_models: ["gemini-claude-sonnet-4-5-thinking", "gemini-claude-sonnet-4-5", "gemini-3-pro-preview", "gemini-2.5-flash"],
  },
  {
    provider: "anthropic",
    status: "unknown",
    model: "claude-sonnet-4",
    available_models: ["claude-opus-4", "claude-sonnet-4", "claude-3-5-sonnet", "claude-3-5-haiku", "claude-3-opus", "claude-3-sonnet", "claude-3-haiku"],
  },
  {
    provider: "openai",
    status: "unknown",
    model: "gpt-4o",
    available_models: ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "o1", "o3"],
  },
  {
    provider: "deepseek",
    status: "unknown",
    model: "deepseek-chat",
    available_models: ["deepseek-chat", "deepseek-reasoner", "deepseek-coder"],
  },
  {
    provider: "qwen",
    status: "unknown",
    model: "qwen3-coder-plus",
    available_models: ["qwen3-coder-plus", "qwen3-coder-flash"],
  },
];

function getFallbackProviders(): Provider[] {
  return FALLBACK_PROVIDERS.map((provider) => ({
    ...provider,
    available_models: [...(provider.available_models ?? [])],
  }));
}

/**
 * Normalize backend responses so the UI always deals with arrays.
 * Some proxy responses omit fields or return null when the backend is down,
 * which otherwise caused `.length` access errors.
 */
type ProviderLike = Partial<Provider> & Record<string, unknown>;

function normalizeProviders(rawProviders: unknown): Provider[] {
  if (!Array.isArray(rawProviders)) {
    return [];
  }

  const normalized = rawProviders
    .filter(
      (provider): provider is ProviderLike =>
        Boolean(provider && typeof provider === "object" && typeof (provider as ProviderLike).provider === "string")
    )
    .map((provider) => ({
      ...provider,
      provider: provider.provider as string,
      status: typeof provider.status === "string" ? provider.status : "unknown",
      model: typeof provider.model === "string" ? provider.model : "",
      available_models: Array.isArray(provider.available_models)
        ? provider.available_models.filter((model): model is string => typeof model === "string")
        : [],
    }));

  return normalized as Provider[];
}

function countModels(models?: string[]): number {
  return Array.isArray(models) ? models.length : 0;
}

/**
 * Fetch providers from the API with proper error handling
 */
async function fetchProvidersFromApi(): Promise<ProvidersListResponse> {
  // Use the backend API endpoint
  const baseUrl = "/api/v1";

  const response = await fetch(`${baseUrl}/providers`, {
    headers: {
      "Content-Type": "application/json",
    },
  });

  if (!response.ok) {
    throw new Error(`Failed to list providers: ${response.statusText}`);
  }

  return response.json();
}

// ============================================================================
// Provider Configuration
// ============================================================================

const PROVIDER_CONFIG: Record<string, { displayName: string; icon: React.ReactNode; color: string }> = {
  // Google Gemini AI Models
  google: {
    displayName: "Gemini AI",
    icon: <Sparkles className="h-4 w-4" />,
    color: "text-blue-500",
  },
  gemini: {
    displayName: "Gemini AI",
    icon: <Sparkles className="h-4 w-4" />,
    color: "text-blue-500",
  },
  "gemini-cli": {
    displayName: "Gemini CLI (OAuth)",
    icon: <Sparkles className="h-4 w-4" />,
    color: "text-blue-400",
  },
  "gemini-openai": {
    displayName: "Gemini (OpenAI Compatible)",
    icon: <Sparkles className="h-4 w-4" />,
    color: "text-blue-300",
  },
  // Antigravity - Hybrid Models
  antigravity: {
    displayName: "Antigravity (Hybrid)",
    icon: <Zap className="h-4 w-4" />,
    color: "text-purple-500",
  },
  // Opels
  openai: {
    displayName: "OpenAI",
    icon: <Bot className="h-4 w-4" />,
    color: "text-green-500",
  },
  // Anthropic Claude Models
  anthropic: {
    displayName: "Anthropic Claude",
    icon: <Brain className="h-4 w-4" />,
    color: "text-orange-500",
  },
  kiro: {
    displayName: "Kiro (Claude)",
    icon: <Brain className="h-4 w-4" />,
    color: "text-orange-400",
  },
  // Qwen Models
  qwen: {
    displayName: "Qwen",
    icon: <Cpu className="h-4 w-4" />,
    color: "text-cyan-500",
  },
  // DeepSeek Models
  deepseek: {
    displayName: "DeepSeek",
    icon: <Search className="h-4 w-4" />,
    color: "text-indigo-500",
  },
  // Cursor Models
  cursor: {
    displayName: "Cursor",
    icon: <Server className="h-4 w-4" />,
    color: "text-pink-500",
  },
};

// ============================================================================
// Helper Functions
// ============================================================================

function getProviderConfig(providerId: string | undefined | null) {
  if (!providerId) {
    return {
      displayName: "Unknown",
      icon: <Cpu className="h-4 w-4" />,
      color: "text-gray-500",
    };
  }
  return (
    PROVIDER_CONFIG[providerId.toLowerCase()] || {
      displayName: providerId.charAt(0).toUpperCase() + providerId.slice(1),
      icon: <Cpu className="h-4 w-4" />,
      color: "text-gray-500",
    }
  );
}

// ============================================================================
// Component
// ============================================================================

export function ProviderModelDropdown({
  selectedProvider,
  selectedModel,
  onSelectionChange,
  compact = false,
  disabled = false,
  showRefresh = true,
  placeholder = "Select a model",
  className,
  align = "start",
  side = "bottom",
}: ProviderModelDropdownProps) {
  const [isOpen, setIsOpen] = useState(false);
  const [isLoading, setIsLoading] = useState(true);
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [providers, setProviders] = useState<Provider[]>([]);
  const [searchQuery, setSearchQuery] = useState("");

  // Fetch providers and models
  const fetchProviders = useCallback(async (showToast = false) => {
    try {
      setIsRefreshing(true);
      setError(null);

      // Use direct fetch instead of enhancedApi to avoid potential circular dependency issues
      const data = await fetchProvidersFromApi();

      // Safely access providers array with fallback to empty array
      const providersList = normalizeProviders(data?.providers);
      setProviders(providersList);

      // Set defaults if nothing selected
      if (!selectedProvider && !selectedModel && data?.default) {
        const defaultProvider = providersList.find((p) => p.provider === data.default);
        if (defaultProvider && defaultProvider.available_models.length > 0) {
          onSelectionChange?.(data.default, defaultProvider.model || defaultProvider.available_models[0]);
        }
      }

      if (showToast) {
        const totalModels = providersList.reduce(
          (acc, p) => acc + countModels(p.available_models),
          0
        );
        toast.success("Models refreshed", {
          description: `Found ${data?.count ?? providersList.length} providers with ${totalModels} models`,
        });
      }
    } catch (err) {
      console.error("Failed to fetch providers:", err);
      const fallbackProviders = normalizeProviders(getFallbackProviders());
      if (fallbackProviders.length > 0) {
        setProviders(fallbackProviders);

        if (!selectedProvider && !selectedModel) {
          const defaultProvider = fallbackProviders.find((p) => p.model) || fallbackProviders[0];
          if (defaultProvider && defaultProvider.available_models.length > 0) {
            onSelectionChange?.(
              defaultProvider.provider,
              defaultProvider.model || defaultProvider.available_models[0]
            );
          }
        }

        if (showToast) {
          toast.warning("Backend unavailable", {
            description: "Showing the default AI model catalog.",
          });
        }
      } else {
        setError("Failed to load models");
        if (showToast) {
          toast.error("Failed to refresh models");
        }
      }
    } finally {
      setIsLoading(false);
      setIsRefreshing(false);
    }
  }, [selectedProvider, selectedModel, onSelectionChange]);

  // Initial fetch
  useEffect(() => {
    fetchProviders();
  }, [fetchProviders]);

  // Group providers with their models
  const providerGroups: ProviderGroup[] = useMemo(() => {
    return providers
      .filter((provider) => provider.provider) // Filter out providers with undefined/null provider field
      .map((provider) => {
        const config = getProviderConfig(provider.provider);
        return {
          provider: provider.provider,
          displayName: config.displayName,
          icon: config.icon,
          status: provider.status as "active" | "inactive" | "unknown",
          models: provider.available_models || [],
          defaultModel: provider.model,
        };
      });
  }, [providers]);

  // Filter models based on search
  const filteredGroups = useMemo(() => {
    if (!searchQuery.trim()) return providerGroups;

    const query = searchQuery.toLowerCase();
    return providerGroups
      .map((group) => ({
        ...group,
        models: group.models.filter(
          (model) =>
            model.toLowerCase().includes(query) ||
            group.displayName.toLowerCase().includes(query)
        ),
      }))
      .filter((group) => group.models.length > 0);
  }, [providerGroups, searchQuery]);

  // Handle model selection
  const handleSelectModel = (provider: string, model: string) => {
    onSelectionChange?.(provider, model);
    setIsOpen(false);
    setSearchQuery("");
  };

  // Get display text for trigger
  const triggerText = useMemo(() => {
    if (selectedModel && selectedProvider) {
      const config = getProviderConfig(selectedProvider);
      return compact ? selectedModel : `${config.displayName}: ${selectedModel}`;
    }
    return placeholder;
  }, [selectedModel, selectedProvider, compact, placeholder]);

  // Get selected provider config
  const selectedProviderConfig = selectedProvider ? getProviderConfig(selectedProvider) : null;

  // Calculate total models count safely
  const totalModelsCount = useMemo(() => {
    return providerGroups.reduce((acc, g) => acc + countModels(g.models), 0);
  }, [providerGroups]);

  // Loading state
  if (isLoading) {
    return (
      <Button
        variant="outline"
        disabled
        className={cn("justify-between", compact ? "w-[200px]" : "w-[300px]", className)}
      >
        <span className="flex items-center gap-2">
          <Loader2 className="h-4 w-4 animate-spin" />
          Loading models...
        </span>
      </Button>
    );
  }

  // Error state
  if (error && providers.length === 0) {
    return (
      <div className={cn("flex items-center gap-2", className)}>
        <Button
          variant="outline"
          className={cn("justify-between text-destructive", compact ? "w-[200px]" : "w-[300px]")}
          onClick={() => fetchProviders(true)}
        >
          <span className="flex items-center gap-2">
            <AlertCircle className="h-4 w-4" />
            {error}
          </span>
          <RefreshCw className="h-4 w-4" />
        </Button>
      </div>
    );
  }

  return (
    <div className={cn("flex items-center gap-2", className)}>
      <DropdownMenu open={isOpen} onOpenChange={setIsOpen}>
        <DropdownMenuTrigger asChild>
          <Button
            variant="outline"
            role="combobox"
            aria-expanded={isOpen}
            disabled={disabled}
            className={cn(
              "justify-between font-normal",
              compact ? "w-[200px]" : "w-[300px]",
              !selectedModel && "text-muted-foreground"
            )}
          >
            <span className="flex items-center gap-2 truncate">
              {selectedProviderConfig && (
                <span className={selectedProviderConfig.color}>
                  {selectedProviderConfig.icon}
                </span>
              )}
              <span className="truncate">{triggerText}</span>
            </span>
            <ChevronDown className="ml-2 h-4 w-4 shrink-0 opacity-50" />
          </Button>
        </DropdownMenuTrigger>

        <DropdownMenuContent
          align={align}
          side={side}
          className={cn("p-0", compact ? "w-[250px]" : "w-[350px]")}
        >
          {/* Search Input */}
          {totalModelsCount > 10 && (
            <div className="p-2 border-b">
              <div className="relative">
                <Search className="absolute left-2.5 top-2.5 h-4 w-4 text-muted-foreground" />
                <input
                  type="text"
                  placeholder="Search models..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  className="w-full pl-8 pr-3 py-2 text-sm border rounded-md bg-background focus:outline-none focus:ring-2 focus:ring-primary/50"
                  autoFocus
                />
              </div>
            </div>
          )}

          <ScrollArea className="max-h-[400px]">
            <div className="p-1">
              {filteredGroups.length === 0 ? (
                <div className="py-6 text-center text-sm text-muted-foreground">
                  No models found
                </div>
              ) : (
                filteredGroups.flatMap((group, index) => {
                  const elements: React.ReactNode[] = [];

                  if (index > 0) {
                    elements.push(
                      <DropdownMenuSeparator key={`sep-${group.provider}`} />
                    );
                  }

                  elements.push(
                    <DropdownMenuGroup key={`group-${group.provider}`}>
                      <DropdownMenuLabel className="flex items-center gap-2 py-2">
                        <span className={getProviderConfig(group.provider).color}>
                          {group.icon}
                        </span>
                        <span>{group.displayName}</span>
                        <Badge
                          variant={group.status === "active" ? "default" : "secondary"}
                          className="ml-auto text-[10px] px-1.5 py-0"
                        >
                          {group.status}
                        </Badge>
                        <span className="text-xs text-muted-foreground">
                          ({group.models?.length ?? 0})
                        </span>
                      </DropdownMenuLabel>

                      {(group.models ?? []).map((model) => {
                        const isSelected =
                          selectedProvider === group.provider && selectedModel === model;
                        const isDefault = model === group.defaultModel;

                        return (
                          <DropdownMenuItem
                            key={`${group.provider}-${model}`}
                            onSelect={() => handleSelectModel(group.provider, model)}
                            className={cn(
                              "flex items-center justify-between cursor-pointer py-2 pl-8",
                              isSelected && "bg-accent"
                            )}
                          >
                            <span className="flex items-center gap-2 truncate">
                              <span className="truncate">{model}</span>
                              {isDefault && (
                                <Badge variant="outline" className="text-[10px] px-1 py-0">
                                  default
                                </Badge>
                              )}
                            </span>
                            {isSelected && <Check className="h-4 w-4 shrink-0" />}
                          </DropdownMenuItem>
                        );
                      })}
                    </DropdownMenuGroup>
                  );

                  return elements;
                })
              )}
            </div>
          </ScrollArea>

          {/* Footer with model count */}
          <div className="border-t p-2 text-xs text-muted-foreground text-center">
            {providers.reduce((acc, p) => acc + countModels(p.available_models), 0)} models across{" "}
            {providers.length} providers
          </div>
        </DropdownMenuContent>
      </DropdownMenu>

      {/* Refresh Button */}
      {showRefresh && (
        <Button
          variant="ghost"
          size="icon"
          onClick={() => fetchProviders(true)}
          disabled={isRefreshing}
          className="h-9 w-9"
        >
          <RefreshCw className={cn("h-4 w-4", isRefreshing && "animate-spin")} />
        </Button>
      )}
    </div>
  );
}

export default ProviderModelDropdown;
