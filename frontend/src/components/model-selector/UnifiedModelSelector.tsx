"use client";

import React, { useState, useEffect, useCallback, useRef, useMemo } from "react";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Separator } from "@/components/ui/separator";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";
import { toast } from "sonner";
import { Loader2, RefreshCw, Check, AlertCircle, Server, Cpu, Zap, Wifi, WifiOff, Clock, ChevronRight, Sparkles, Shield, Activity } from "lucide-react";

// Types
interface ProviderInfo {
  provider: string;
  displayName: string;
  status: "active" | "degraded" | "unavailable" | "unknown";
  isHealthy: boolean;
  models: string[];
  defaultModel: string | null;
  latencyMs: number | null;
}

interface ModelInfo {
  id: string;
  name: string;
  description?: string;
  maxTokens: number;
  isDefault: boolean;
  tier: "standard" | "premium" | "experimental";
}

interface SelectionState {
  provider: string | null;
  model: string | null;
  isDefault: boolean;
  sessionId: string | null;
}

interface SessionInfo {
  sessionId: string;
  provider: string;
  model: string;
  createdAt: string;
  lastActivity: string;
  requestCount: number;
}

type ConnectionStatus = "connected" | "connecting" | "disconnected" | "error";

interface UnifiedModelSelectorProps {
  onSelectionChange?: (provider: string, model: string) => void;
  className?: string;
  variant?: "default" | "compact" | "minimal";
  showSessionInfo?: boolean;
  showHealthStatus?: boolean;
  autoSync?: boolean;
}

// Constants
const API_BASE = "/api/v1";
const WS_RECONNECT_DELAY = 3000;
const WS_MAX_RECONNECT_ATTEMPTS = 5;
const DEBOUNCE_DELAY = 400;
const SESSION_STORAGE_KEY = "chimera_session_id";

// Hooks
function useDebounce<T>(value: T, delay: number): T {
  const [debouncedValue, setDebouncedValue] = useState<T>(value);
  useEffect(() => {
    const timer = setTimeout(() => setDebouncedValue(value), delay);
    return () => clearTimeout(timer);
  }, [value, delay]);
  return debouncedValue;
}

// Sub-components
function ConnectionIndicator({ status }: { status: ConnectionStatus }) {
  const configs: Record<ConnectionStatus, { icon: typeof Wifi; color: string; pulse: boolean; label: string }> = {
    connected: { icon: Wifi, color: "text-green-500", pulse: true, label: "Live sync" },
    connecting: { icon: Loader2, color: "text-yellow-500", pulse: false, label: "Connecting..." },
    disconnected: { icon: WifiOff, color: "text-muted-foreground", pulse: false, label: "Offline" },
    error: { icon: AlertCircle, color: "text-red-500", pulse: false, label: "Connection error" },
  };
  const config = configs[status];
  const Icon = config.icon;

  return (
    <TooltipProvider>
      <Tooltip>
        <TooltipTrigger asChild>
          <div className={cn("flex items-center gap-1.5 text-xs", config.color)}>
            {config.pulse && (
              <span className="relative flex h-2 w-2">
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-green-400 opacity-75" />
                <span className="relative inline-flex rounded-full h-2 w-2 bg-green-500" />
              </span>
            )}
            <Icon className={cn("h-3.5 w-3.5", status === "connecting" && "animate-spin")} />
            <span className="hidden sm:inline">{config.label}</span>
          </div>
        </TooltipTrigger>
        <TooltipContent><p>{config.label}</p></TooltipContent>
      </Tooltip>
    </TooltipProvider>
  );
}

function ModelTierBadge({ tier }: { tier: string }) {
  const configs: Record<string, { icon: typeof Shield; className: string }> = {
    premium: { icon: Sparkles, className: "bg-purple-100 text-purple-800 dark:bg-purple-900/30 dark:text-purple-300" },
    experimental: { icon: Zap, className: "bg-orange-100 text-orange-800 dark:bg-orange-900/30 dark:text-orange-300" },
    standard: { icon: Shield, className: "bg-gray-100 text-gray-800 dark:bg-gray-800 dark:text-gray-300" },
  };
  const config = configs[tier] || configs.standard;
  const Icon = config.icon;
  return (
    <span className={cn("inline-flex items-center gap-1 px-2 py-0.5 rounded text-[10px] font-medium", config.className)}>
      <Icon className="h-2.5 w-2.5" />{tier}
    </span>
  );
}

function LoadingSkeleton({ variant }: { variant: string }) {
  if (variant === "minimal") return <div className="flex items-center gap-2"><Skeleton className="h-9 w-32" /><Skeleton className="h-9 w-40" /></div>;
  if (variant === "compact") return <div className="flex items-center gap-4"><Skeleton className="h-9 w-36" /><Skeleton className="h-9 w-44" /><Skeleton className="h-9 w-9" /></div>;
  return (
    <Card>
      <CardHeader><div className="flex items-center justify-between"><div className="space-y-2"><Skeleton className="h-5 w-32" /><Skeleton className="h-4 w-48" /></div><Skeleton className="h-9 w-9" /></div></CardHeader>
      <CardContent className="space-y-6"><div className="space-y-2"><Skeleton className="h-4 w-16" /><Skeleton className="h-10 w-full" /></div><div className="space-y-2"><Skeleton className="h-4 w-12" /><Skeleton className="h-10 w-full" /></div><Skeleton className="h-10 w-full" /></CardContent>
    </Card>
  );
}

// Main Component
export function UnifiedModelSelector({
  onSelectionChange,
  className,
  variant = "default",
  showSessionInfo = true,
  showHealthStatus = true,
  autoSync = true,
}: UnifiedModelSelectorProps) {
  const [providers, setProviders] = useState<ProviderInfo[]>([]);
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [selection, setSelection] = useState<SelectionState>({ provider: null, model: null, isDefault: true, sessionId: null });
  const [sessionInfo, setSessionInfo] = useState<SessionInfo | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [isSaving, setIsSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [saveError, setSaveError] = useState<string | null>(null);
  const [isInitialized, setIsInitialized] = useState(false);
  const [wsStatus, setWsStatus] = useState<ConnectionStatus>("disconnected");

  const lastSyncedRef = useRef<{ provider: string; model: string } | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectAttemptsRef = useRef(0);

  const selectedProvider = useMemo(() => providers.find((p) => p.provider === selection.provider), [providers, selection.provider]);
  const availableModels = useMemo(() => selectedProvider ? models.filter((m) => selectedProvider.models.includes(m.id)) : [], [selectedProvider, models]);

  // API Functions
  const fetchProviders = useCallback(async () => {
    const response = await fetch(`${API_BASE}/providers/available`);
    if (!response.ok) throw new Error("Failed to fetch providers");
    const data = await response.json();
    const mapped: ProviderInfo[] = data.providers.map((p: Record<string, unknown>) => ({
      provider: p.provider as string, displayName: p.display_name as string,
      status: p.is_healthy ? "active" : "unavailable", isHealthy: p.is_healthy as boolean,
      models: p.models as string[], defaultModel: p.default_model as string | null, latencyMs: p.latency_ms as number | null,
    }));
    setProviders(mapped);
    if (!selection.provider) setSelection((prev) => ({ ...prev, provider: data.default_provider, model: data.default_model }));
    return { defaultProvider: data.default_provider, defaultModel: data.default_model };
  }, [selection.provider]);

  const fetchModelsForProvider = useCallback(async (provider: string) => {
    try {
      const response = await fetch(`${API_BASE}/providers/${provider}/models`);
      if (!response.ok) return;
      const data = await response.json();
      const mapped: ModelInfo[] = data.models.map((m: Record<string, unknown>) => ({
        id: m.id as string, name: m.name as string, description: m.description as string | undefined,
        maxTokens: (m.max_tokens as number) || 4096, isDefault: m.is_default as boolean,
        tier: ((m.tier as string) || "standard") as "standard" | "premium" | "experimental",
      }));
      setModels(mapped);
    } catch (e) { console.error("Failed to fetch models:", e); }
  }, []);

  const ensureSession = useCallback(async () => {
    const storedSessionId = localStorage.getItem(SESSION_STORAGE_KEY);
    if (storedSessionId) {
      try {
        const response = await fetch(`${API_BASE}/session/${storedSessionId}`);
        if (response.ok) {
          const data = await response.json();
          if (data) {
            setSelection((prev) => ({ ...prev, sessionId: storedSessionId, provider: data.provider || prev.provider, model: data.model || prev.model, isDefault: false }));
            setSessionInfo({ sessionId: data.session_id, provider: data.provider, model: data.model, createdAt: data.created_at, lastActivity: data.last_activity, requestCount: data.request_count });
            lastSyncedRef.current = { provider: data.provider, model: data.model };
            return storedSessionId;
          }
        }
      } catch { /* Session invalid */ }
      localStorage.removeItem(SESSION_STORAGE_KEY);
    }
    try {
      const response = await fetch(`${API_BASE}/session`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ provider: selection.provider, model: selection.model }) });
      if (response.ok) {
        const data = await response.json();
        localStorage.setItem(SESSION_STORAGE_KEY, data.session_id);
        setSelection((prev) => ({ ...prev, sessionId: data.session_id, provider: data.provider, model: data.model }));
        lastSyncedRef.current = { provider: data.provider, model: data.model };
        return data.session_id;
      }
    } catch (e) { console.error("Failed to create session:", e); }
    return null;
  }, [selection.provider, selection.model]);

  const saveSelection = useCallback(async (provider: string, model: string) => {
    if (lastSyncedRef.current?.provider === provider && lastSyncedRef.current?.model === model) return true;
    setIsSaving(true); setSaveError(null);
    try {
      const sessionId = selection.sessionId || (await ensureSession());
      if (!sessionId) throw new Error("Failed to establish session");
      const response = await fetch(`${API_BASE}/providers/select`, { method: "POST", headers: { "Content-Type": "application/json", "X-Session-ID": sessionId }, body: JSON.stringify({ provider, model }) });
      if (!response.ok) { const errorData = await response.json(); throw new Error(errorData.detail?.message || errorData.detail || "Failed to save selection"); }
      const data = await response.json();
      lastSyncedRef.current = { provider: data.provider, model: data.model };
      setSelection((prev) => ({ ...prev, provider: data.provider, model: data.model, sessionId: data.session_id, isDefault: false }));
      toast.success("Model updated", { description: `Now using ${data.model} from ${data.provider}` });
      onSelectionChange?.(data.provider, data.model);
      return true;
    } catch (e) {
      const message = e instanceof Error ? e.message : "Failed to save selection";
      setSaveError(message); toast.error("Failed to update model", { description: message });
      if (lastSyncedRef.current) setSelection((prev) => ({ ...prev, provider: lastSyncedRef.current!.provider, model: lastSyncedRef.current!.model }));
      return false;
    } finally { setIsSaving(false); }
  }, [selection.sessionId, ensureSession, onSelectionChange]);

  // WebSocket
  useEffect(() => {
    if (!autoSync || !isInitialized || typeof window === "undefined") return;
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const wsUrl = `${protocol}//${window.location.host}${API_BASE}/providers/ws/selection`;
    
    const connect = () => {
      try {
        setWsStatus("connecting");
        const ws = new WebSocket(wsUrl);
        ws.onopen = () => { setWsStatus("connected"); reconnectAttemptsRef.current = 0; };
        ws.onmessage = (event) => {
          try {
            const message = JSON.parse(event.data);
            if (message.type === "selection_change" && message.data.session_id === selection.sessionId) {
              setSelection((prev) => ({ ...prev, provider: message.data.provider, model: message.data.model, isDefault: false }));
              lastSyncedRef.current = { provider: message.data.provider, model: message.data.model };
            } else if (message.type === "health_update" && message.data.providers) {
              setProviders((prev) => prev.map((p) => {
                const health = message.data.providers.find((h: { provider: string }) => h.provider === p.provider);
                return health ? { ...p, isHealthy: health.is_healthy, latencyMs: health.latency_ms, status: health.is_healthy ? "active" : "unavailable" } : p;
              }));
            }
          } catch { /* ignore parse errors */ }
        };
        ws.onclose = () => { setWsStatus("disconnected"); if (reconnectAttemptsRef.current < WS_MAX_RECONNECT_ATTEMPTS) { reconnectAttemptsRef.current++; setTimeout(connect, WS_RECONNECT_DELAY); } };
        ws.onerror = () => setWsStatus("error");
        wsRef.current = ws;
      } catch { setWsStatus("error"); }
    };
    connect();
    return () => { wsRef.current?.close(); };
  }, [autoSync, isInitialized, selection.sessionId]);

  // Event Handlers
  const handleProviderChange = useCallback(async (newProvider: string) => {
    setSelection((prev) => ({ ...prev, provider: newProvider })); setSaveError(null);
    await fetchModelsForProvider(newProvider);
    const providerInfo = providers.find((p) => p.provider === newProvider);
    if (providerInfo?.defaultModel) setSelection((prev) => ({ ...prev, model: providerInfo.defaultModel }));
  }, [providers, fetchModelsForProvider]);

  const handleModelChange = useCallback((newModel: string) => { setSelection((prev) => ({ ...prev, model: newModel })); setSaveError(null); }, []);
  const handleApply = useCallback(() => { if (selection.provider && selection.model) saveSelection(selection.provider, selection.model); }, [selection.provider, selection.model, saveSelection]);
  const handleRefresh = useCallback(async () => { setIsLoading(true); try { await fetchProviders(); if (selection.provider) await fetchModelsForProvider(selection.provider); toast.success("Refreshed"); } catch { toast.error("Failed to refresh"); } finally { setIsLoading(false); } }, [fetchProviders, fetchModelsForProvider, selection.provider]);

  // Auto-save
  const debouncedSelection = useDebounce(selection, DEBOUNCE_DELAY);
  useEffect(() => { if (autoSync && isInitialized && debouncedSelection.provider && debouncedSelection.model) saveSelection(debouncedSelection.provider, debouncedSelection.model); }, [autoSync, isInitialized, debouncedSelection.provider, debouncedSelection.model, saveSelection]);

  // Initialization
  useEffect(() => {
    const initialize = async () => {
      setIsLoading(true); setError(null);
      try {
        const { defaultProvider } = await fetchProviders();
        await ensureSession();
        if (defaultProvider) await fetchModelsForProvider(selection.provider || defaultProvider);
        setIsInitialized(true);
      } catch (e) { setError("Failed to initialize model selector"); console.error("Initialization error:", e); }
      finally { setIsLoading(false); }
    };
    initialize();
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => { if (selection.provider && isInitialized) fetchModelsForProvider(selection.provider); }, [selection.provider, isInitialized, fetchModelsForProvider]);

  // Render
  if (isLoading && !isInitialized) return <LoadingSkeleton variant={variant} />;
  if (error && !isInitialized) return (
    <Alert variant="destructive"><AlertCircle className="h-4 w-4" /><AlertTitle>Error</AlertTitle>
      <AlertDescription className="flex items-center justify-between"><span>{error}</span><Button variant="outline" size="sm" onClick={handleRefresh}><RefreshCw className="h-4 w-4 mr-1" />Retry</Button></AlertDescription>
    </Alert>
  );

  if (variant === "minimal") return (
    <div className={cn("flex items-center gap-2", className)}>
      <Select value={selection.provider || ""} onValueChange={handleProviderChange}><SelectTrigger className="w-32 h-8 text-xs"><SelectValue placeholder="Provider" /></SelectTrigger><SelectContent>{providers.map((p) => <SelectItem key={p.provider} value={p.provider} className="text-xs">{p.displayName}</SelectItem>)}</SelectContent></Select>
      <Select value={selection.model || ""} onValueChange={handleModelChange}><SelectTrigger className="w-40 h-8 text-xs"><SelectValue placeholder="Model" /></SelectTrigger><SelectContent>{availableModels.map((m) => <SelectItem key={m.id} value={m.id} className="text-xs">{m.name}</SelectItem>)}</SelectContent></Select>
      {isSaving && <Loader2 className="h-4 w-4 animate-spin text-muted-foreground" />}
    </div>
  );

  if (variant === "compact") return (
    <div className={cn("flex items-center gap-4", className)}>
      <div className="flex items-center gap-2"><Server className="h-4 w-4 text-muted-foreground" />
        <Select value={selection.provider || ""} onValueChange={handleProviderChange}><SelectTrigger className="w-36"><SelectValue placeholder="Provider" /></SelectTrigger><SelectContent>{providers.map((p) => <SelectItem key={p.provider} value={p.provider}><div className="flex items-center gap-2"><span>{p.displayName}</span>{showHealthStatus && <span className={cn("h-1.5 w-1.5 rounded-full", p.isHealthy ? "bg-green-500" : "bg-red-500")} />}</div></SelectItem>)}</SelectContent></Select>
      </div>
      <div className="flex items-center gap-2"><Cpu className="h-4 w-4 text-muted-foreground" />
        <Select value={selection.model || ""} onValueChange={handleModelChange}><SelectTrigger className="w-44"><SelectValue placeholder="Model" /></SelectTrigger><SelectContent>{availableModels.map((m) => <SelectItem key={m.id} value={m.id}><div className="flex items-center gap-2"><span>{m.name}</span><ModelTierBadge tier={m.tier} /></div></SelectItem>)}</SelectContent></Select>
      </div>
      <Button size="sm" onClick={handleApply} disabled={isSaving}>{isSaving ? <Loader2 className="h-4 w-4 animate-spin" /> : <Check className="h-4 w-4" />}</Button>
      <ConnectionIndicator status={wsStatus} />
    </div>
  );

  // Default variant
  return (
    <Card className={className}>
      <CardHeader>
        <div className="flex items-center justify-between">
          <div><CardTitle className="flex items-center gap-2"><Cpu className="h-5 w-5" />Model Selection</CardTitle><CardDescription>Choose the AI provider and model for your requests</CardDescription></div>
          <div className="flex items-center gap-2"><ConnectionIndicator status={wsStatus} /><Button variant="ghost" size="icon" onClick={handleRefresh}><RefreshCw className="h-4 w-4" /></Button></div>
        </div>
      </CardHeader>
      <CardContent className="space-y-6">
        <div className="space-y-2"><Label>Provider</Label>
          <Select value={selection.provider || ""} onValueChange={handleProviderChange}><SelectTrigger><SelectValue placeholder="Select a provider" /></SelectTrigger>
            <SelectContent>{providers.map((p) => <SelectItem key={p.provider} value={p.provider}><div className="flex items-center gap-2"><Server className="h-4 w-4" /><span>{p.displayName}</span>{showHealthStatus && <Badge variant={p.isHealthy ? "default" : "secondary"} className="ml-2">{p.status}</Badge>}{p.latencyMs && <span className="text-xs text-muted-foreground ml-1">{p.latencyMs}ms</span>}</div></SelectItem>)}</SelectContent>
          </Select>
          {selectedProvider && <p className="text-xs text-muted-foreground">{selectedProvider.models.length} models available</p>}
        </div>
        <div className="space-y-2"><Label>Model</Label>
          <Select value={selection.model || ""} onValueChange={handleModelChange}><SelectTrigger><SelectValue placeholder="Select a model" /></SelectTrigger>
            <SelectContent>{availableModels.map((m) => <SelectItem key={m.id} value={m.id}><div className="flex items-center gap-2"><Cpu className="h-4 w-4" /><span>{m.name}</span><ModelTierBadge tier={m.tier} />{m.isDefault && <Badge variant="outline" className="ml-2">Default</Badge>}</div></SelectItem>)}</SelectContent>
          </Select>
        </div>
        <Button onClick={handleApply} disabled={isSaving || !selection.provider || !selection.model} className="w-full">{isSaving ? <><Loader2 className="mr-2 h-4 w-4 animate-spin" />Syncing...</> : <><Check className="mr-2 h-4 w-4" />Apply Selection</>}</Button>
        {saveError && <Alert variant="destructive"><AlertCircle className="h-4 w-4" /><AlertTitle>Selection not synced</AlertTitle><AlertDescription className="flex items-center justify-between gap-3"><span>{saveError}</span><Button variant="outline" size="sm" onClick={handleApply} disabled={isSaving}>Retry</Button></AlertDescription></Alert>}
        {showSessionInfo && sessionInfo && <><Separator /><div className="space-y-2"><Label className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">Session Info</Label><div className="grid grid-cols-2 gap-2 text-sm"><div><span className="text-muted-foreground">Session ID:</span><p className="font-mono text-xs truncate">{sessionInfo.sessionId}</p></div><div><span className="text-muted-foreground">Requests:</span><p>{sessionInfo.requestCount}</p></div><div><span className="text-muted-foreground">Created:</span><p className="text-xs">{new Date(sessionInfo.createdAt).toLocaleString()}</p></div><div><span className="text-muted-foreground">Last Activity:</span><p className="text-xs">{new Date(sessionInfo.lastActivity).toLocaleString()}</p></div></div></div></>}
        <div className="rounded-lg border bg-muted/50 p-3"><div className="flex items-center gap-2"><Zap className="h-4 w-4 text-primary" /><span className="font-medium">Current Selection</span></div><p className="text-sm text-muted-foreground mt-1">{selection.provider && selection.model ? `${selection.model} via ${selection.provider}` : "No model selected"}</p></div>
      </CardContent>
    </Card>
  );
}

export default UnifiedModelSelector;