"use client";

import React, { useState, useEffect, useCallback, useRef } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import {
  enhancedApi,
  ModelsListResponse,
  ProviderWithModels,
  SessionInfoResponse
} from "@/lib/api-enhanced";
import { toast } from "sonner";
import { Loader2, RefreshCw, Check, AlertCircle, Server, Cpu, Zap } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";

// Fallback providers when backend is unavailable (January 2026 models)
const FALLBACK_MODELS_DATA: ModelsListResponse = {
  providers: [
    {
      provider: "google",
      status: "unknown",
      model: "gemini-3-pro",
      available_models: [
        "gemini-3-pro",
        "gemini-3-flash",
        "gemini-2.5-pro-latest",
        "gemini-2.0-flash",
        "gemini-1.5-pro",
        "gemini-1.5-flash",
      ],
    },
    {
      provider: "gemini",
      status: "unknown",
      model: "gemini-3-pro",
      available_models: [
        "gemini-3-pro",
        "gemini-3-flash",
        "gemini-2.5-pro-latest",
        "gemini-2.0-flash",
        "gemini-1.5-pro",
        "gemini-1.5-flash",
      ],
    },
    {
      provider: "anthropic",
      status: "unknown",
      model: "claude-opus-4.5",
      available_models: [
        "claude-opus-4.5",
        "claude-sonnet-4.5",
        "claude-haiku-4.5",
        "claude-3-5-sonnet-20241022",
      ],
    },
    {
      provider: "openai",
      status: "unknown",
      model: "gpt-5.2",
      available_models: [
        "gpt-5.2",
        "gpt-5.2-codex",
        "o3-mini",
        "gpt-4.5",
        "gpt-4o",
        "gpt-4o-mini",
      ],
    },
    {
      provider: "deepseek",
      status: "unknown",
      model: "deepseek-v4",
      available_models: [
        "deepseek-v4",
        "deepseek-chat",
        "deepseek-reasoner",
      ],
    },
    {
      provider: "bigmodel",
      status: "unknown",
      model: "glm-4.7",
      available_models: [
        "glm-4.7",
        "glm-4.6v",
        "glm-4-plus",
        "glm-4-flash",
      ],
    },
  ],
  default_provider: "gemini",
  default_model: "gemini-3-pro",
  total_models: 30,
};

interface ModelSelectorProps {
  onModelChange?: (provider: string, model: string) => void;
  showSessionInfo?: boolean;
  compact?: boolean;
  autoApply?: boolean;
}

export function ModelSelector({
  onModelChange,
  showSessionInfo = true,
  compact = false,
  autoApply = true,
}: ModelSelectorProps) {
  const [modelsData, setModelsData] = useState<ModelsListResponse | null>(null);
  const [selectedProvider, setSelectedProvider] = useState<string>("");
  const [selectedModel, setSelectedModel] = useState<string>("");
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [sessionInfo, setSessionInfo] = useState<SessionInfoResponse | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [isSaving, setIsSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [saveError, setSaveError] = useState<string | null>(null);
  const [modelsLoaded, setModelsLoaded] = useState(false);
  const [sessionLoaded, setSessionLoaded] = useState(false);

  const lastSyncedRef = useRef<{ provider: string; model: string } | null>(null);
  const saveTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const saveRequestIdRef = useRef(0);

  // Load available models
  const loadModels = useCallback(async () => {
    setIsLoading(true);
    setError(null);

    try {
      const response = await enhancedApi.models.list();
      setModelsData(response.data);

      // Set defaults if not already selected
      if (!selectedProvider && response.data.default_provider) {
        setSelectedProvider(response.data.default_provider);
      }
      if (!selectedModel && response.data.default_model) {
        setSelectedModel(response.data.default_model);
      }
    } catch (err) {
      console.error("Failed to load models:", err);
      setError("Failed to load available models");
    } finally {
      setIsLoading(false);
      setModelsLoaded(true);
    }
  }, [selectedProvider, selectedModel]);

  // Load or create session
  const initializeSession = useCallback(async () => {
    // Check for existing session in localStorage
    const storedSessionId = localStorage.getItem("chimera_session_id");

    if (storedSessionId) {
      try {
        const response = await enhancedApi.session.get(storedSessionId);
        // Session found and valid
        if (response.data) {
          setSessionId(storedSessionId);
          setSessionInfo(response.data);

          // Validate the stored provider/model against available models
          // If invalid, we'll use defaults from modelsData instead
          const sessionProvider = response.data.provider;
          const sessionModel = response.data.model;

          if (sessionProvider && sessionModel) {
            // Pre-validate by checking if provider exists
            // The actual validation will happen during sync
            setSelectedProvider(sessionProvider);
            setSelectedModel(sessionModel);
            lastSyncedRef.current = {
              provider: sessionProvider,
              model: sessionModel,
            };
          }
          // If provider/model not set in session, defaults from loadModels will be used

          setSessionLoaded(true);
          return;
        }
        // Session not found or expired (null response)
        localStorage.removeItem("chimera_session_id");
      } catch {
        // Network or other error, create new session
        localStorage.removeItem("chimera_session_id");
      }
    }

    // Create new session
    try {
      const response = await enhancedApi.session.create();
      const newSessionId = response.data.session_id;
      setSessionId(newSessionId);
      localStorage.setItem("chimera_session_id", newSessionId);
      if (response.data.provider) {
        setSelectedProvider(response.data.provider);
      }
      if (response.data.model) {
        setSelectedModel(response.data.model);
      }
      if (response.data.provider && response.data.model) {
        lastSyncedRef.current = {
          provider: response.data.provider,
          model: response.data.model,
        };
      }

      // Fetch full session info
      const sessionResponse = await enhancedApi.session.get(newSessionId);
      if (sessionResponse.data) {
        setSessionInfo(sessionResponse.data);
      }
    } catch (err) {
      console.error("Failed to create session:", err);
      // Continue without session - will use defaults
    } finally {
      setSessionLoaded(true);
    }
  }, []);

  // Initialize on mount
  useEffect(() => {
    loadModels();
    initializeSession();
  }, [loadModels, initializeSession]);

  // Get available models for selected provider
  const getModelsForProvider = (providerId: string): string[] => {
    if (!modelsData) return [];
    const provider = modelsData.providers.find(p => p.provider === providerId);
    return provider?.available_models || [];
  };

  // Get provider info
  const getProviderInfo = (providerId: string): ProviderWithModels | undefined => {
    return modelsData?.providers.find(p => p.provider === providerId);
  };

  // Handle provider change
  const handleProviderChange = (newProvider: string) => {
    setSelectedProvider(newProvider);
    setSaveError(null);

    // Reset model to first available for this provider
    const models = getModelsForProvider(newProvider);
    if (models.length > 0) {
      setSelectedModel(models[0]);
    } else {
      setSelectedModel("");
    }
  };

  // Handle model change
  const handleModelChange = (newModel: string) => {
    setSelectedModel(newModel);
    setSaveError(null);
  };

  const saveSelection = useCallback(async function saveSelectionInner(
    provider: string,
    model: string,
    options?: { isFallback?: boolean }
  ) {
    if (!provider || !model) {
      toast.error("Please select a provider and model");
      return;
    }

    const lastSynced = lastSyncedRef.current;
    if (lastSynced && lastSynced.provider === provider && lastSynced.model === model) {
      if (saveError) setSaveError(null);
      return;
    }

    const requestId = saveRequestIdRef.current + 1;
    saveRequestIdRef.current = requestId;

    setIsSaving(true);
    setSaveError(null);

    try {
      // Validate the selection first
      const validateResponse = await enhancedApi.models.validate(provider, model);

      if (!validateResponse.data.valid) {
        const message = validateResponse.data.message || "Invalid model selection";
        toast.error("Invalid model selection", { description: message });
        setSaveError(message);

        // Use fallback if provided
        const fallbackProvider = validateResponse.data.fallback_provider;
        const fallbackModel = validateResponse.data.fallback_model;

        if (fallbackProvider && fallbackModel) {
          const shouldRetry = fallbackProvider !== provider || fallbackModel !== model;
          setSelectedProvider(fallbackProvider);
          setSelectedModel(fallbackModel);

          if (shouldRetry && !options?.isFallback) {
            void saveSelectionInner(fallbackProvider, fallbackModel, { isFallback: true });
          }
        }
        return;
      }

      let activeSessionId = sessionId;
      if (!activeSessionId) {
        const sessionResponse = await enhancedApi.session.create({ provider, model });
        if (sessionResponse.data.session_id) {
          activeSessionId = sessionResponse.data.session_id;
          setSessionId(activeSessionId);
          localStorage.setItem("chimera_session_id", activeSessionId);
        }
      }

      if (!activeSessionId) {
        throw new Error("Unable to create a session for model selection");
      }

      const response = await enhancedApi.session.updateModel(
        activeSessionId,
        model,
        provider
      );

      if (saveRequestIdRef.current !== requestId) {
        return;
      }

      if (response.data.success) {
        const nextProvider = response.data.provider || provider;
        const nextModel = response.data.model || model;
        lastSyncedRef.current = { provider: nextProvider, model: nextModel };

        toast.success("Model updated", {
          description: `Now using ${nextModel} from ${nextProvider}`,
        });

        // Refresh session info
        const sessionResponse = await enhancedApi.session.get(activeSessionId);
        setSessionInfo(sessionResponse.data);

        onModelChange?.(nextProvider, nextModel);
      } else {
        const message = response.data.message || "Failed to update selection";
        setSaveError(message);
        toast.warning("Model update issue", { description: message });

        if (response.data.reverted_to_default) {
          if (response.data.provider) setSelectedProvider(response.data.provider);
          if (response.data.model) setSelectedModel(response.data.model);
          if (response.data.provider && response.data.model) {
            lastSyncedRef.current = {
              provider: response.data.provider,
              model: response.data.model,
            };
          }
        } else if (lastSyncedRef.current) {
          setSelectedProvider(lastSyncedRef.current.provider);
          setSelectedModel(lastSyncedRef.current.model);
        }
      }
    } catch (err) {
      if (saveRequestIdRef.current !== requestId) {
        return;
      }

      const message = err instanceof Error ? err.message : "Failed to save model selection";
      console.error("Failed to save model selection:", err);
      setSaveError(message);
      toast.error("Failed to save model selection", { description: message });

      if (lastSyncedRef.current) {
        setSelectedProvider(lastSyncedRef.current.provider);
        setSelectedModel(lastSyncedRef.current.model);
      }
    } finally {
      if (saveRequestIdRef.current === requestId) {
        setIsSaving(false);
      }
    }
  }, [onModelChange, sessionId]);

  const queueSaveSelection = useCallback((provider: string, model: string) => {
    if (saveTimerRef.current) {
      clearTimeout(saveTimerRef.current);
    }

    saveTimerRef.current = setTimeout(() => {
      void saveSelection(provider, model);
    }, 400);
  }, [saveSelection]);

  useEffect(() => {
    if (!autoApply) return;
    if (!modelsLoaded || !sessionLoaded) return;
    if (!selectedProvider || !selectedModel) return;

    const lastSynced = lastSyncedRef.current;
    if (lastSynced && lastSynced.provider === selectedProvider && lastSynced.model === selectedModel) {
      return;
    }

    queueSaveSelection(selectedProvider, selectedModel);
  }, [
    autoApply,
    modelsLoaded,
    sessionLoaded,
    selectedProvider,
    selectedModel,
    queueSaveSelection,
  ]);

  useEffect(() => {
    return () => {
      if (saveTimerRef.current) {
        clearTimeout(saveTimerRef.current);
      }
    };
  }, []);

  // Refresh models list
  const refreshModels = async () => {
    await loadModels();
    toast.success("Models refreshed");
  };

  if (isLoading) {
    return (
      <Card className={compact ? "p-4" : ""}>
        <CardContent className="flex items-center justify-center py-8">
          <Loader2 className="h-6 w-6 animate-spin mr-2" />
          <span>Loading models...</span>
        </CardContent>
      </Card>
    );
  }

  if (error) {
    return (
      <Alert variant="destructive">
        <AlertCircle className="h-4 w-4" />
        <AlertTitle>Error</AlertTitle>
        <AlertDescription>{error}</AlertDescription>
      </Alert>
    );
  }

  if (compact) {
    return (
      <div className="flex items-center gap-4">
        <div className="flex items-center gap-2">
          <Server className="h-4 w-4 text-muted-foreground" />
          <Select value={selectedProvider} onValueChange={handleProviderChange}>
            <SelectTrigger className="w-[140px]">
              <SelectValue placeholder="Provider" />
            </SelectTrigger>
            <SelectContent>
              {modelsData?.providers.map((provider) => (
                <SelectItem key={provider.provider} value={provider.provider}>
                  {provider.provider}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>

        <div className="flex items-center gap-2">
          <Cpu className="h-4 w-4 text-muted-foreground" />
          <Select value={selectedModel} onValueChange={handleModelChange}>
            <SelectTrigger className="w-[180px]">
              <SelectValue placeholder="Model" />
            </SelectTrigger>
            <SelectContent>
              {getModelsForProvider(selectedProvider).map((model) => (
                <SelectItem key={model} value={model}>
                  {model}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>

        <Button
          size="sm"
          onClick={() => saveSelection(selectedProvider, selectedModel)}
          disabled={isSaving}
        >
          {isSaving ? (
            <Loader2 className="h-4 w-4 animate-spin" />
          ) : (
            <Check className="h-4 w-4" />
          )}
        </Button>
      </div>
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
          <Button variant="ghost" size="icon" onClick={refreshModels}>
            <RefreshCw className="h-4 w-4" />
          </Button>
        </div>
      </CardHeader>
      <CardContent className="space-y-6">
        {/* Provider Selection */}
        <div className="space-y-2">
          <Label>Provider</Label>
          <Select value={selectedProvider} onValueChange={handleProviderChange}>
            <SelectTrigger>
              <SelectValue placeholder="Select a provider" />
            </SelectTrigger>
            <SelectContent>
              {modelsData?.providers.map((provider) => (
                <SelectItem key={provider.provider} value={provider.provider}>
                  <div className="flex items-center gap-2">
                    <Server className="h-4 w-4" />
                    <span>{provider.provider}</span>
                    <Badge variant={provider.status === "active" ? "default" : "secondary"} className="ml-2">
                      {provider.status}
                    </Badge>
                  </div>
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
          {selectedProvider && (
            <p className="text-xs text-muted-foreground">
              {getModelsForProvider(selectedProvider).length} models available
            </p>
          )}
        </div>

        {/* Model Selection */}
        <div className="space-y-2">
          <Label>Model</Label>
          <Select value={selectedModel} onValueChange={handleModelChange}>
            <SelectTrigger>
              <SelectValue placeholder="Select a model" />
            </SelectTrigger>
            <SelectContent>
              {getModelsForProvider(selectedProvider).map((model) => (
                <SelectItem key={model} value={model}>
                  <div className="flex items-center gap-2">
                    <Cpu className="h-4 w-4" />
                    <span>{model}</span>
                    {model === modelsData?.default_model && (
                      <Badge variant="outline" className="ml-2">Default</Badge>
                    )}
                  </div>
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>

        {/* Save Button */}
        <Button
          onClick={() => saveSelection(selectedProvider, selectedModel)}
          disabled={isSaving || !selectedProvider || !selectedModel}
          className="w-full"
        >
          {isSaving ? (
            <>
              <Loader2 className="mr-2 h-4 w-4 animate-spin" />
              Syncing...
            </>
          ) : (
            <>
              <Check className="mr-2 h-4 w-4" />
              Apply Selection
            </>
          )}
        </Button>

        {saveError && (
          <Alert variant="destructive">
            <AlertCircle className="h-4 w-4" />
            <AlertTitle>Selection not synced</AlertTitle>
            <AlertDescription className="flex items-center justify-between gap-3">
              <span>{saveError}</span>
              <Button
                variant="outline"
                size="sm"
                onClick={() => saveSelection(selectedProvider, selectedModel)}
                disabled={isSaving}
              >
                Retry
              </Button>
            </AlertDescription>
          </Alert>
        )}

        {/* Session Info */}
        {showSessionInfo && sessionInfo && (
          <>
            <Separator />
            <div className="space-y-2">
              <Label className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
                Session Info
              </Label>
              <div className="grid grid-cols-2 gap-2 text-sm">
                <div>
                  <span className="text-muted-foreground">Session ID:</span>
                  <p className="font-mono text-xs truncate">{sessionInfo.session_id}</p>
                </div>
                <div>
                  <span className="text-muted-foreground">Requests:</span>
                  <p>{sessionInfo.request_count}</p>
                </div>
                <div>
                  <span className="text-muted-foreground">Created:</span>
                  <p className="text-xs">{new Date(sessionInfo.created_at).toLocaleString()}</p>
                </div>
                <div>
                  <span className="text-muted-foreground">Last Activity:</span>
                  <p className="text-xs">{new Date(sessionInfo.last_activity).toLocaleString()}</p>
                </div>
              </div>
            </div>
          </>
        )}

        {/* Summary */}
        <div className="rounded-lg border bg-muted/50 p-3">
          <div className="flex items-center gap-2">
            <Zap className="h-4 w-4 text-primary" />
            <span className="font-medium">Current Selection</span>
          </div>
          <p className="text-sm text-muted-foreground mt-1">
            {selectedProvider && selectedModel
              ? `${selectedModel} via ${selectedProvider}`
              : "No model selected"}
          </p>
        </div>
      </CardContent>
    </Card>
  );
}
