"use client";

import React, { useState, useEffect, useCallback } from "react";
import { Label } from "@/components/ui/label";
import { Loader2, RefreshCw } from "lucide-react";
import { Button } from "@/components/ui/button";
import enhancedApi, { ModelsListResponse } from "@/lib/api-enhanced";
import { ProviderModelDropdown } from "@/components/model-selector/ProviderModelDropdown";

export interface ModelSelection {
  provider: string;
  model: string;
}

interface AIModelSelectorProps {
  /** Callback when model selection changes */
  onSelectionChange: (selection: ModelSelection) => void;
  /** Initial provider (optional) */
  defaultProvider?: string;
  /** Initial model (optional) */
  defaultModel?: string;
  /** Show as inline (horizontal) or stacked (vertical) */
  layout?: "inline" | "stacked";
  /** Show refresh button */
  showRefresh?: boolean;
  /** Label for the selector */
  label?: string;
  /** Disabled state */
  disabled?: boolean;
}

export function AIModelSelector({
  onSelectionChange,
  defaultProvider,
  defaultModel,
  layout = "inline",
  showRefresh = false,
  label,
  disabled = false,
}: AIModelSelectorProps) {
  const [modelsData, setModelsData] = useState<ModelsListResponse | null>(null);
  const [selectedProvider, setSelectedProvider] = useState<string>(defaultProvider || "");
  const [selectedModel, setSelectedModel] = useState<string>(defaultModel || "");
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Load available models
  const loadModels = useCallback(async () => {
    setIsLoading(true);
    setError(null);

    try {
      const response = await enhancedApi.models.list();
      setModelsData(response.data);

      // Set defaults if not already selected
      const provider = defaultProvider || response.data.default_provider;
      const model = defaultModel || response.data.default_model;

      if (!selectedProvider && provider) {
        setSelectedProvider(provider);
      }
      if (!selectedModel && model) {
        setSelectedModel(model);
      }

      // Notify parent of initial selection
      if (provider && model) {
        onSelectionChange({ provider, model });
      }
    } catch (err) {
      console.error("Failed to load models:", err);
      setError("Failed to load models");
    } finally {
      setIsLoading(false);
    }
  }, [defaultProvider, defaultModel, selectedProvider, selectedModel, onSelectionChange]);

  // Initialize on mount
  useEffect(() => {
    loadModels();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Get available models for selected provider
  const getModelsForProvider = (providerId: string): string[] => {
    if (!modelsData) return [];
    const provider = modelsData.providers.find(p => p.provider === providerId);
    return provider?.available_models || [];
  };

  // Handle provider change
  const handleProviderChange = (newProvider: string) => {
    setSelectedProvider(newProvider);

    // Reset model to first available for this provider
    const models = getModelsForProvider(newProvider);
    const newModel = models.length > 0 ? models[0] : "";
    setSelectedModel(newModel);

    // Notify parent
    if (newModel) {
      onSelectionChange({ provider: newProvider, model: newModel });
    }
  };

  // Handle model change
  const handleModelChange = (newModel: string) => {
    setSelectedModel(newModel);
    onSelectionChange({ provider: selectedProvider, model: newModel });
  };

  if (isLoading) {
    return (
      <div className="flex items-center gap-2 text-sm text-muted-foreground">
        <Loader2 className="h-4 w-4 animate-spin" />
        <span>Loading models...</span>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex items-center gap-2 text-sm text-destructive">
        <span>{error}</span>
        <Button variant="ghost" size="sm" onClick={loadModels}>
          <RefreshCw className="h-3 w-3" />
        </Button>
      </div>
    );
  }

  const isInline = layout === "inline";

  // Handle selection change from ProviderModelDropdown
  const handleSelectionChange = (provider: string, model: string) => {
    setSelectedProvider(provider);
    setSelectedModel(model);
    onSelectionChange({ provider, model });
  };

  return (
    <div className={isInline ? "flex items-center gap-3 flex-wrap" : "space-y-3"}>
      {label && (
        <Label className={isInline ? "text-sm font-medium" : ""}>{label}</Label>
      )}

      {/* Provider & Model Selection using ProviderModelDropdown */}
      <ProviderModelDropdown
        selectedProvider={selectedProvider}
        selectedModel={selectedModel}
        onSelectionChange={handleSelectionChange}
        compact={isInline}
        disabled={disabled}
        showRefresh={showRefresh}
        placeholder="Select Gemini AI model"
      />
    </div>
  );
}

export default AIModelSelector;
