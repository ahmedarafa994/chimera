"use client";

import * as React from "react";
import { useState, useCallback, useMemo } from "react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Skeleton } from "@/components/ui/skeleton";
import { ScrollArea, ScrollBar } from "@/components/ui/scroll-area";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { cn } from "@/lib/utils";
import {
  Key,
  Shield,
  CheckCircle2,
  XCircle,
  AlertCircle,
  RefreshCw,
  Plus,
  Settings,
  Activity,
  Zap,
} from "lucide-react";
import { ApiKeyForm, type ApiKeyFormData, type ProviderId } from "./ApiKeyForm";
import { ApiKeyList, type ApiKeyItem } from "./ApiKeyList";

// =============================================================================
// Types
// =============================================================================

export interface ProviderKeySummary {
  provider_id: ProviderId;
  provider_name: string;
  total_keys: number;
  active_keys: number;
  primary_key_id: string | null;
  backup_key_ids: string[];
  has_valid_key: boolean;
  status: "configured" | "unconfigured" | "error";
}

export interface ApiKeyManagerProps {
  // Data
  keys: ApiKeyItem[];
  providers: ProviderKeySummary[];
  isLoading?: boolean;
  error?: Error | null;

  // Callbacks
  onCreateKey: (data: ApiKeyFormData) => Promise<void>;
  onUpdateKey: (keyId: string, data: Partial<ApiKeyFormData>) => Promise<void>;
  onDeleteKey: (keyId: string) => Promise<void>;
  onTestKey: (keyId: string) => Promise<{ success: boolean; message: string; latency_ms?: number }>;
  onTestNewKey: (data: ApiKeyFormData) => Promise<{ success: boolean; message: string; latency_ms?: number }>;
  onActivateKey: (keyId: string) => Promise<void>;
  onDeactivateKey: (keyId: string) => Promise<void>;
  onRevokeKey: (keyId: string) => Promise<void>;
  onRefresh: () => Promise<void>;

  // Optional
  defaultProvider?: ProviderId;
  className?: string;
}

// =============================================================================
// Provider Configuration
// =============================================================================

const providerInfo: Record<
  ProviderId,
  { name: string; icon: React.ElementType; color: string; bgColor: string }
> = {
  openai: {
    name: "OpenAI",
    icon: Zap,
    color: "text-green-500",
    bgColor: "bg-green-500/10",
  },
  anthropic: {
    name: "Anthropic",
    icon: Zap,
    color: "text-orange-500",
    bgColor: "bg-orange-500/10",
  },
  google: {
    name: "Google",
    icon: Zap,
    color: "text-blue-500",
    bgColor: "bg-blue-500/10",
  },
  deepseek: {
    name: "DeepSeek",
    icon: Zap,
    color: "text-purple-500",
    bgColor: "bg-purple-500/10",
  },
  qwen: {
    name: "Qwen",
    icon: Zap,
    color: "text-cyan-500",
    bgColor: "bg-cyan-500/10",
  },
  bigmodel: {
    name: "BigModel",
    icon: Zap,
    color: "text-indigo-500",
    bgColor: "bg-indigo-500/10",
  },
  routeway: {
    name: "Routeway",
    icon: Zap,
    color: "text-pink-500",
    bgColor: "bg-pink-500/10",
  },
  cursor: {
    name: "Cursor",
    icon: Zap,
    color: "text-teal-500",
    bgColor: "bg-teal-500/10",
  },
};

// All supported providers
const ALL_PROVIDERS: ProviderId[] = [
  "openai",
  "anthropic",
  "google",
  "deepseek",
  "qwen",
  "bigmodel",
  "routeway",
  "cursor",
];

// =============================================================================
// Provider Tab Component
// =============================================================================

interface ProviderTabProps {
  providerId: ProviderId;
  summary: ProviderKeySummary | undefined;
  isActive: boolean;
}

function ProviderTab({ providerId, summary, isActive }: ProviderTabProps) {
  const info = providerInfo[providerId];
  const Icon = info.icon;

  const statusIcon = useMemo(() => {
    if (!summary || summary.status === "unconfigured") {
      return <AlertCircle className="h-3 w-3 text-muted-foreground" />;
    }
    if (summary.status === "error" || !summary.has_valid_key) {
      return <XCircle className="h-3 w-3 text-red-500" />;
    }
    return <CheckCircle2 className="h-3 w-3 text-emerald-500" />;
  }, [summary]);

  return (
    <div className="flex items-center gap-2">
      <div className={cn("p-1 rounded", info.bgColor)}>
        <Icon className={cn("h-3 w-3", info.color)} />
      </div>
      <span className={cn("font-medium", isActive && "text-primary")}>
        {info.name}
      </span>
      {statusIcon}
      {summary && summary.total_keys > 0 && (
        <Badge variant="secondary" className="text-[10px] h-4 px-1">
          {summary.total_keys}
        </Badge>
      )}
    </div>
  );
}

// =============================================================================
// Stats Card Component
// =============================================================================

interface StatsCardProps {
  providers: ProviderKeySummary[];
  keys: ApiKeyItem[];
}

function StatsCard({ providers, keys }: StatsCardProps) {
  const stats = useMemo(() => {
    const totalKeys = keys.length;
    const activeKeys = keys.filter((k) => k.status === "active").length;
    const configuredProviders = providers.filter((p) => p.has_valid_key).length;
    const totalProviders = ALL_PROVIDERS.length;

    return {
      totalKeys,
      activeKeys,
      configuredProviders,
      totalProviders,
    };
  }, [providers, keys]);

  return (
    <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
      <Card>
        <CardHeader className="pb-2">
          <CardDescription className="text-xs">Total Keys</CardDescription>
          <CardTitle className="text-2xl flex items-center gap-2">
            <Key className="h-5 w-5 text-muted-foreground" />
            {stats.totalKeys}
          </CardTitle>
        </CardHeader>
      </Card>

      <Card>
        <CardHeader className="pb-2">
          <CardDescription className="text-xs">Active Keys</CardDescription>
          <CardTitle className="text-2xl flex items-center gap-2">
            <CheckCircle2 className="h-5 w-5 text-emerald-500" />
            {stats.activeKeys}
          </CardTitle>
        </CardHeader>
      </Card>

      <Card>
        <CardHeader className="pb-2">
          <CardDescription className="text-xs">Configured Providers</CardDescription>
          <CardTitle className="text-2xl flex items-center gap-2">
            <Activity className="h-5 w-5 text-blue-500" />
            {stats.configuredProviders}/{stats.totalProviders}
          </CardTitle>
        </CardHeader>
      </Card>

      <Card>
        <CardHeader className="pb-2">
          <CardDescription className="text-xs">Coverage</CardDescription>
          <CardTitle className="text-2xl flex items-center gap-2">
            <Shield className="h-5 w-5 text-purple-500" />
            {Math.round((stats.configuredProviders / stats.totalProviders) * 100)}%
          </CardTitle>
        </CardHeader>
      </Card>
    </div>
  );
}

// =============================================================================
// Main API Key Manager Component
// =============================================================================

export function ApiKeyManager({
  keys,
  providers,
  isLoading,
  error,
  onCreateKey,
  onUpdateKey,
  onDeleteKey,
  onTestKey,
  onTestNewKey,
  onActivateKey,
  onDeactivateKey,
  onRevokeKey,
  onRefresh,
  defaultProvider = "openai",
  className,
}: ApiKeyManagerProps) {
  const [activeTab, setActiveTab] = useState<ProviderId | "all">("all");
  const [isFormOpen, setIsFormOpen] = useState(false);
  const [editingKey, setEditingKey] = useState<ApiKeyItem | null>(null);
  const [formProviderId, setFormProviderId] = useState<ProviderId | undefined>(undefined);
  const [isSaving, setIsSaving] = useState(false);
  const [isRefreshing, setIsRefreshing] = useState(false);

  // Get provider summaries map for quick lookup
  const providerSummaryMap = useMemo(() => {
    const map = new Map<ProviderId, ProviderKeySummary>();
    providers.forEach((p) => map.set(p.provider_id, p));
    return map;
  }, [providers]);

  // Filter keys by active tab
  const filteredKeys = useMemo(() => {
    if (activeTab === "all") return keys;
    return keys.filter((k) => k.provider_id === activeTab);
  }, [keys, activeTab]);

  // Handle add key
  const handleAddKey = useCallback((providerId?: ProviderId) => {
    setEditingKey(null);
    setFormProviderId(providerId);
    setIsFormOpen(true);
  }, []);

  // Handle edit key
  const handleEditKey = useCallback((key: ApiKeyItem) => {
    setEditingKey(key);
    setFormProviderId(key.provider_id);
    setIsFormOpen(true);
  }, []);

  // Handle save key
  const handleSaveKey = useCallback(
    async (data: ApiKeyFormData) => {
      setIsSaving(true);
      try {
        if (editingKey) {
          await onUpdateKey(editingKey.id, data);
        } else {
          await onCreateKey(data);
        }
        setIsFormOpen(false);
        setEditingKey(null);
      } finally {
        setIsSaving(false);
      }
    },
    [editingKey, onCreateKey, onUpdateKey]
  );

  // Handle test key (for new keys in form)
  const handleTestNewKey = useCallback(
    async (data: ApiKeyFormData) => {
      return onTestNewKey(data);
    },
    [onTestNewKey]
  );

  // Handle refresh
  const handleRefresh = useCallback(async () => {
    setIsRefreshing(true);
    try {
      await onRefresh();
    } finally {
      setIsRefreshing(false);
    }
  }, [onRefresh]);

  // Loading state
  if (isLoading) {
    return (
      <div className={cn("space-y-6", className)}>
        <div className="flex items-center justify-between">
          <div>
            <Skeleton className="h-8 w-48" />
            <Skeleton className="h-4 w-64 mt-2" />
          </div>
          <Skeleton className="h-10 w-24" />
        </div>
        <div className="grid grid-cols-4 gap-4">
          <Skeleton className="h-20" />
          <Skeleton className="h-20" />
          <Skeleton className="h-20" />
          <Skeleton className="h-20" />
        </div>
        <Skeleton className="h-64" />
      </div>
    );
  }

  // Error state
  if (error) {
    return (
      <div className={cn("space-y-6", className)}>
        <Card className="border-destructive">
          <CardHeader>
            <CardTitle className="text-destructive flex items-center gap-2">
              <AlertCircle className="h-5 w-5" />
              Error Loading API Keys
            </CardTitle>
            <CardDescription>{error.message}</CardDescription>
          </CardHeader>
          <CardContent>
            <Button onClick={handleRefresh}>
              <RefreshCw className="h-4 w-4 mr-2" />
              Retry
            </Button>
          </CardContent>
        </Card>
      </div>
    );
  }

  return (
    <div className={cn("space-y-6", className)}>
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-semibold tracking-tight flex items-center gap-2">
            <Shield className="h-6 w-6" />
            API Key Management
          </h2>
          <p className="text-muted-foreground text-sm">
            Securely manage your LLM provider API keys with encryption at rest
          </p>
        </div>
        <div className="flex items-center gap-2">
          <TooltipProvider>
            <Tooltip>
              <TooltipTrigger asChild>
                <Button
                  variant="outline"
                  size="icon"
                  onClick={handleRefresh}
                  disabled={isRefreshing}
                >
                  <RefreshCw
                    className={cn("h-4 w-4", isRefreshing && "animate-spin")}
                  />
                </Button>
              </TooltipTrigger>
              <TooltipContent>Refresh</TooltipContent>
            </Tooltip>
          </TooltipProvider>
          <Button onClick={() => handleAddKey()}>
            <Plus className="h-4 w-4 mr-2" />
            Add API Key
          </Button>
        </div>
      </div>

      {/* Stats */}
      <StatsCard providers={providers} keys={keys} />

      {/* Provider Tabs */}
      <Tabs value={activeTab} onValueChange={(v) => setActiveTab(v as ProviderId | "all")}>
        <ScrollArea className="w-full">
          <TabsList className="inline-flex h-auto p-1 bg-muted/50">
            <TabsTrigger value="all" className="px-4 py-2">
              <div className="flex items-center gap-2">
                <Settings className="h-3.5 w-3.5" />
                <span>All Providers</span>
                <Badge variant="secondary" className="text-[10px] h-4 px-1">
                  {keys.length}
                </Badge>
              </div>
            </TabsTrigger>
            {ALL_PROVIDERS.map((providerId) => (
              <TabsTrigger key={providerId} value={providerId} className="px-4 py-2">
                <ProviderTab
                  providerId={providerId}
                  summary={providerSummaryMap.get(providerId)}
                  isActive={activeTab === providerId}
                />
              </TabsTrigger>
            ))}
          </TabsList>
          <ScrollBar orientation="horizontal" />
        </ScrollArea>

        {/* All Providers Tab */}
        <TabsContent value="all" className="mt-4">
          <ApiKeyList
            keys={filteredKeys}
            onEdit={handleEditKey}
            onDelete={onDeleteKey}
            onTest={onTestKey}
            onActivate={onActivateKey}
            onDeactivate={onDeactivateKey}
            onRevoke={onRevokeKey}
            onAdd={() => handleAddKey()}
            onRefresh={handleRefresh}
            isRefreshing={isRefreshing}
            emptyMessage="No API keys configured. Add your first key to get started."
          />
        </TabsContent>

        {/* Provider-specific Tabs */}
        {ALL_PROVIDERS.map((providerId) => (
          <TabsContent key={providerId} value={providerId} className="mt-4">
            <ApiKeyList
              keys={filteredKeys}
              onEdit={handleEditKey}
              onDelete={onDeleteKey}
              onTest={onTestKey}
              onActivate={onActivateKey}
              onDeactivate={onDeactivateKey}
              onRevoke={onRevokeKey}
              onAdd={() => handleAddKey(providerId)}
              onRefresh={handleRefresh}
              isRefreshing={isRefreshing}
              emptyMessage={`No API keys configured for ${providerInfo[providerId].name}. Add a key to enable this provider.`}
            />
          </TabsContent>
        ))}
      </Tabs>

      {/* Add/Edit Form Dialog */}
      <ApiKeyForm
        isOpen={isFormOpen}
        onClose={() => {
          setIsFormOpen(false);
          setEditingKey(null);
          setFormProviderId(undefined);
        }}
        onSave={handleSaveKey}
        onTest={handleTestNewKey}
        initialData={
          editingKey
            ? {
                provider_id: editingKey.provider_id,
                name: editingKey.name,
                role: editingKey.role,
                priority: editingKey.priority,
                description: editingKey.description,
                tags: editingKey.tags,
              }
            : undefined
        }
        isEditing={!!editingKey}
        isSaving={isSaving}
        providerId={formProviderId}
      />
    </div>
  );
}

export default ApiKeyManager;
