"use client";

import * as React from "react";
import { useState, useCallback } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { cn } from "@/lib/utils";
import { ProviderType } from "@/lib/api/validation";
import {
  Eye,
  EyeOff,
  Save,
  Trash2,
  TestTube,
  Loader2,
  CheckCircle2,
  XCircle,
  AlertCircle,
  Key,
  Globe,
  Settings,
  Plus,
  RefreshCw,
} from "lucide-react";

// =============================================================================
// Types
// =============================================================================

export interface ProviderConfigData {
  id?: string;
  name: string;
  type: ProviderType;
  apiKey?: string;
  baseUrl?: string;
  organizationId?: string;
  projectId?: string;
  maxRetries?: number;
  timeout?: number;
  customHeaders?: Record<string, string>;
  enabled?: boolean;
}

export interface ProviderConfigFormProps {
  isOpen: boolean;
  onClose: () => void;
  onSave: (config: ProviderConfigData) => Promise<void>;
  onDelete?: (providerId: string) => Promise<void>;
  onTest?: (config: ProviderConfigData) => Promise<{ success: boolean; message: string }>;
  initialData?: Partial<ProviderConfigData>;
  isEditing?: boolean;
  isSaving?: boolean;
  isTesting?: boolean;
  isDeleting?: boolean;
}

// =============================================================================
// Provider Type Configuration
// =============================================================================

const providerTypeConfig: Record<
  ProviderType,
  {
    name: string;
    description: string;
    requiresApiKey: boolean;
    requiresBaseUrl: boolean;
    defaultBaseUrl?: string;
    supportsOrganization?: boolean;
    supportsProject?: boolean;
  }
> = {
  openai: {
    name: "OpenAI",
    description: "GPT-4, GPT-3.5, and other OpenAI models",
    requiresApiKey: true,
    requiresBaseUrl: false,
    defaultBaseUrl: "https://api.openai.com/v1",
    supportsOrganization: true,
    supportsProject: true,
  },
  anthropic: {
    name: "Anthropic",
    description: "Claude models (Claude 3, Claude 2, etc.)",
    requiresApiKey: true,
    requiresBaseUrl: false,
    defaultBaseUrl: "https://api.anthropic.com",
  },
  google: {
    name: "Google AI",
    description: "Google AI Studio / Vertex AI models",
    requiresApiKey: true,
    requiresBaseUrl: false,
    defaultBaseUrl: "https://generativelanguage.googleapis.com",
    supportsProject: true,
  },
  gemini: {
    name: "Gemini",
    description: "Google Gemini models",
    requiresApiKey: true,
    requiresBaseUrl: false,
    defaultBaseUrl: "https://generativelanguage.googleapis.com",
  },
  deepseek: {
    name: "DeepSeek",
    description: "DeepSeek AI models",
    requiresApiKey: true,
    requiresBaseUrl: false,
    defaultBaseUrl: "https://api.deepseek.com",
  },
  qwen: {
    name: "Qwen",
    description: "Alibaba Qwen models",
    requiresApiKey: true,
    requiresBaseUrl: false,
    defaultBaseUrl: "https://dashscope.aliyuncs.com/api/v1",
  },
  "gemini-cli": {
    name: "Gemini CLI",
    description: "Google Gemini via CLI interface",
    requiresApiKey: true,
    requiresBaseUrl: false,
    defaultBaseUrl: "https://generativelanguage.googleapis.com",
  },
  antigravity: {
    name: "AntiGravity",
    description: "AntiGravity AI models",
    requiresApiKey: true,
    requiresBaseUrl: false,
  },
  kiro: {
    name: "Kiro",
    description: "Kiro AI assistant models",
    requiresApiKey: true,
    requiresBaseUrl: false,
  },
  cursor: {
    name: "Cursor",
    description: "Cursor AI coding assistant",
    requiresApiKey: true,
    requiresBaseUrl: false,
  },
  xai: {
    name: "xAI",
    description: "xAI Grok models",
    requiresApiKey: true,
    requiresBaseUrl: false,
    defaultBaseUrl: "https://api.x.ai/v1",
  },
  mock: {
    name: "Mock Provider",
    description: "Mock provider for testing purposes",
    requiresApiKey: false,
    requiresBaseUrl: false,
  },
};

// =============================================================================
// API Key Input Component
// =============================================================================

interface ApiKeyInputProps {
  value: string;
  onChange: (value: string) => void;
  placeholder?: string;
  disabled?: boolean;
  hasExistingKey?: boolean;
}

function ApiKeyInput({
  value,
  onChange,
  placeholder = "sk-...",
  disabled = false,
  hasExistingKey = false,
}: ApiKeyInputProps) {
  const [showKey, setShowKey] = useState(false);

  return (
    <div className="relative">
      <Input
        type={showKey ? "text" : "password"}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder={hasExistingKey ? "••••••••••••••••" : placeholder}
        disabled={disabled}
        className="pr-10 font-mono text-sm"
      />
      <Button
        type="button"
        variant="ghost"
        size="icon"
        className="absolute right-0 top-0 h-full px-3 hover:bg-transparent"
        onClick={() => setShowKey(!showKey)}
        disabled={disabled}
      >
        {showKey ? (
          <EyeOff className="h-4 w-4 text-muted-foreground" />
        ) : (
          <Eye className="h-4 w-4 text-muted-foreground" />
        )}
      </Button>
    </div>
  );
}

// =============================================================================
// Custom Headers Editor
// =============================================================================

interface CustomHeadersEditorProps {
  headers: Record<string, string>;
  onChange: (headers: Record<string, string>) => void;
  disabled?: boolean;
}

function CustomHeadersEditor({
  headers,
  onChange,
  disabled = false,
}: CustomHeadersEditorProps) {
  const [newKey, setNewKey] = useState("");
  const [newValue, setNewValue] = useState("");

  const addHeader = useCallback(() => {
    if (newKey.trim() && newValue.trim()) {
      onChange({ ...headers, [newKey.trim()]: newValue.trim() });
      setNewKey("");
      setNewValue("");
    }
  }, [headers, newKey, newValue, onChange]);

  const removeHeader = useCallback(
    (key: string) => {
      const newHeaders = { ...headers };
      delete newHeaders[key];
      onChange(newHeaders);
    },
    [headers, onChange]
  );

  return (
    <div className="space-y-3">
      {/* Existing headers */}
      {Object.entries(headers).map(([key, value]) => (
        <div key={key} className="flex items-center gap-2">
          <Input
            value={key}
            disabled
            className="flex-1 font-mono text-xs"
          />
          <Input
            value={value}
            disabled
            className="flex-1 font-mono text-xs"
          />
          <Button
            type="button"
            variant="ghost"
            size="icon"
            className="h-8 w-8 shrink-0"
            onClick={() => removeHeader(key)}
            disabled={disabled}
          >
            <Trash2 className="h-3.5 w-3.5 text-destructive" />
          </Button>
        </div>
      ))}

      {/* Add new header */}
      <div className="flex items-center gap-2">
        <Input
          value={newKey}
          onChange={(e) => setNewKey(e.target.value)}
          placeholder="Header name"
          disabled={disabled}
          className="flex-1 font-mono text-xs"
        />
        <Input
          value={newValue}
          onChange={(e) => setNewValue(e.target.value)}
          placeholder="Header value"
          disabled={disabled}
          className="flex-1 font-mono text-xs"
        />
        <Button
          type="button"
          variant="outline"
          size="icon"
          className="h-8 w-8 shrink-0"
          onClick={addHeader}
          disabled={disabled || !newKey.trim() || !newValue.trim()}
        >
          <Plus className="h-3.5 w-3.5" />
        </Button>
      </div>
    </div>
  );
}

// =============================================================================
// Test Result Display
// =============================================================================

interface TestResultProps {
  result: { success: boolean; message: string } | null;
}

function TestResult({ result }: TestResultProps) {
  if (!result) return null;

  return (
    <div
      className={cn(
        "flex items-center gap-2 rounded-md p-3 text-sm",
        result.success
          ? "bg-emerald-500/10 text-emerald-600 dark:text-emerald-400"
          : "bg-destructive/10 text-destructive"
      )}
    >
      {result.success ? (
        <CheckCircle2 className="h-4 w-4 shrink-0" />
      ) : (
        <XCircle className="h-4 w-4 shrink-0" />
      )}
      <span>{result.message}</span>
    </div>
  );
}

// =============================================================================
// Main Provider Config Form Component
// =============================================================================

export function ProviderConfigForm({
  isOpen,
  onClose,
  onSave,
  onDelete,
  onTest,
  initialData,
  isEditing = false,
  isSaving = false,
  isTesting = false,
  isDeleting = false,
}: ProviderConfigFormProps) {
  // Form state
  const [formData, setFormData] = useState<ProviderConfigData>({
    name: initialData?.name || "",
    type: initialData?.type || "openai",
    apiKey: "",
    baseUrl: initialData?.baseUrl || "",
    organizationId: initialData?.organizationId || "",
    projectId: initialData?.projectId || "",
    maxRetries: initialData?.maxRetries || 3,
    timeout: initialData?.timeout || 30,
    customHeaders: initialData?.customHeaders || {},
    enabled: initialData?.enabled ?? true,
    ...initialData,
  });

  const [testResult, setTestResult] = useState<{
    success: boolean;
    message: string;
  } | null>(null);

  const [activeTab, setActiveTab] = useState("basic");

  // Get provider type config
  const typeConfig = providerTypeConfig[formData.type];

  // Update form field
  const updateField = useCallback(
    <K extends keyof ProviderConfigData>(
      field: K,
      value: ProviderConfigData[K]
    ) => {
      setFormData((prev) => ({ ...prev, [field]: value }));
      setTestResult(null);
    },
    []
  );

  // Handle type change
  const handleTypeChange = useCallback((type: ProviderType) => {
    const config = providerTypeConfig[type];
    setFormData((prev) => ({
      ...prev,
      type,
      baseUrl: config.defaultBaseUrl || prev.baseUrl,
      name: prev.name || config.name,
    }));
    setTestResult(null);
  }, []);

  // Handle save
  const handleSave = useCallback(async () => {
    try {
      await onSave(formData);
      onClose();
    } catch (error) {
      console.error("Failed to save provider config:", error);
    }
  }, [formData, onSave, onClose]);

  // Handle test
  const handleTest = useCallback(async () => {
    if (!onTest) return;
    try {
      const result = await onTest(formData);
      setTestResult(result);
    } catch (error) {
      setTestResult({
        success: false,
        message: error instanceof Error ? error.message : "Test failed",
      });
    }
  }, [formData, onTest]);

  // Handle delete
  const handleDelete = useCallback(async () => {
    if (!onDelete || !initialData?.id) return;
    try {
      await onDelete(initialData.id);
      onClose();
    } catch (error) {
      console.error("Failed to delete provider:", error);
    }
  }, [onDelete, initialData?.id, onClose]);

  // Validation
  const isValid =
    formData.name.trim() &&
    formData.type &&
    (!typeConfig.requiresApiKey || formData.apiKey || isEditing) &&
    (!typeConfig.requiresBaseUrl || formData.baseUrl);

  return (
    <Dialog open={isOpen} onOpenChange={(open) => !open && onClose()}>
      <DialogContent className="max-w-lg">
        <DialogHeader>
          <DialogTitle>
            {isEditing ? "Edit Provider" : "Add New Provider"}
          </DialogTitle>
          <DialogDescription>
            {isEditing
              ? "Update the configuration for this AI provider."
              : "Configure a new AI provider to use with Chimera."}
          </DialogDescription>
        </DialogHeader>

        <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
          <TabsList className="grid w-full grid-cols-2">
            <TabsTrigger value="basic" className="gap-2">
              <Key className="h-3.5 w-3.5" />
              Basic
            </TabsTrigger>
            <TabsTrigger value="advanced" className="gap-2">
              <Settings className="h-3.5 w-3.5" />
              Advanced
            </TabsTrigger>
          </TabsList>

          {/* Basic Settings Tab */}
          <TabsContent value="basic" className="space-y-4 pt-4">
            {/* Provider Type */}
            <div className="space-y-2">
              <label className="text-sm font-medium">Provider Type</label>
              <Select
                value={formData.type}
                onValueChange={(v) => handleTypeChange(v as ProviderType)}
                disabled={isEditing}
              >
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {Object.entries(providerTypeConfig).map(([type, config]) => (
                    <SelectItem key={type} value={type}>
                      <div className="flex flex-col">
                        <span>{config.name}</span>
                        <span className="text-[10px] text-muted-foreground">
                          {config.description}
                        </span>
                      </div>
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            {/* Provider Name */}
            <div className="space-y-2">
              <label className="text-sm font-medium">Display Name</label>
              <Input
                value={formData.name}
                onChange={(e) => updateField("name", e.target.value)}
                placeholder={typeConfig.name}
              />
            </div>

            {/* API Key */}
            {typeConfig.requiresApiKey && (
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <label className="text-sm font-medium">API Key</label>
                  {isEditing && (
                    <Badge variant="outline" className="text-[10px]">
                      Leave empty to keep existing
                    </Badge>
                  )}
                </div>
                <ApiKeyInput
                  value={formData.apiKey || ""}
                  onChange={(v) => updateField("apiKey", v)}
                  hasExistingKey={isEditing}
                />
              </div>
            )}

            {/* Base URL */}
            {typeConfig.requiresBaseUrl && (
              <div className="space-y-2">
                <label className="text-sm font-medium">Base URL</label>
                <div className="relative">
                  <Globe className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
                  <Input
                    value={formData.baseUrl || ""}
                    onChange={(e) => updateField("baseUrl", e.target.value)}
                    placeholder={typeConfig.defaultBaseUrl || "https://api.example.com"}
                    className="pl-10"
                  />
                </div>
              </div>
            )}

            {/* Organization ID (OpenAI) */}
            {typeConfig.supportsOrganization && (
              <div className="space-y-2">
                <label className="text-sm font-medium">
                  Organization ID{" "}
                  <span className="text-muted-foreground">(optional)</span>
                </label>
                <Input
                  value={formData.organizationId || ""}
                  onChange={(e) => updateField("organizationId", e.target.value)}
                  placeholder="org-..."
                />
              </div>
            )}

            {/* Project ID */}
            {typeConfig.supportsProject && (
              <div className="space-y-2">
                <label className="text-sm font-medium">
                  Project ID{" "}
                  <span className="text-muted-foreground">(optional)</span>
                </label>
                <Input
                  value={formData.projectId || ""}
                  onChange={(e) => updateField("projectId", e.target.value)}
                  placeholder="project-..."
                />
              </div>
            )}

            {/* Test Result */}
            <TestResult result={testResult} />
          </TabsContent>

          {/* Advanced Settings Tab */}
          <TabsContent value="advanced" className="space-y-4 pt-4">
            {/* Timeout */}
            <div className="space-y-2">
              <label className="text-sm font-medium">
                Request Timeout (seconds)
              </label>
              <Input
                type="number"
                value={formData.timeout || 30}
                onChange={(e) =>
                  updateField("timeout", parseInt(e.target.value) || 30)
                }
                min={5}
                max={300}
              />
            </div>

            {/* Max Retries */}
            <div className="space-y-2">
              <label className="text-sm font-medium">Max Retries</label>
              <Input
                type="number"
                value={formData.maxRetries || 3}
                onChange={(e) =>
                  updateField("maxRetries", parseInt(e.target.value) || 3)
                }
                min={0}
                max={10}
              />
            </div>

            {/* Custom Headers */}
            <div className="space-y-2">
              <label className="text-sm font-medium">Custom Headers</label>
              <Card className="p-3">
                <CustomHeadersEditor
                  headers={formData.customHeaders || {}}
                  onChange={(headers) => updateField("customHeaders", headers)}
                />
              </Card>
            </div>

            {/* Enabled Toggle */}
            <div className="flex items-center justify-between rounded-lg border p-3">
              <div>
                <p className="text-sm font-medium">Enable Provider</p>
                <p className="text-xs text-muted-foreground">
                  Disabled providers will not be available for selection
                </p>
              </div>
              <Button
                type="button"
                variant={formData.enabled ? "default" : "outline"}
                size="sm"
                onClick={() => updateField("enabled", !formData.enabled)}
              >
                {formData.enabled ? "Enabled" : "Disabled"}
              </Button>
            </div>
          </TabsContent>
        </Tabs>

        <DialogFooter className="flex-col gap-2 sm:flex-row">
          {/* Delete button (only for editing) */}
          {isEditing && onDelete && (
            <Button
              type="button"
              variant="destructive"
              onClick={handleDelete}
              disabled={isDeleting || isSaving}
              className="sm:mr-auto"
            >
              {isDeleting ? (
                <Loader2 className="h-4 w-4 animate-spin mr-2" />
              ) : (
                <Trash2 className="h-4 w-4 mr-2" />
              )}
              Delete
            </Button>
          )}

          {/* Test button */}
          {onTest && (
            <Button
              type="button"
              variant="outline"
              onClick={handleTest}
              disabled={isTesting || !isValid}
            >
              {isTesting ? (
                <Loader2 className="h-4 w-4 animate-spin mr-2" />
              ) : (
                <TestTube className="h-4 w-4 mr-2" />
              )}
              Test Connection
            </Button>
          )}

          {/* Cancel button */}
          <Button type="button" variant="outline" onClick={onClose}>
            Cancel
          </Button>

          {/* Save button */}
          <Button
            type="button"
            onClick={handleSave}
            disabled={isSaving || !isValid}
          >
            {isSaving ? (
              <Loader2 className="h-4 w-4 animate-spin mr-2" />
            ) : (
              <Save className="h-4 w-4 mr-2" />
            )}
            {isEditing ? "Update" : "Add Provider"}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

// =============================================================================
// Provider List Component
// =============================================================================

export interface ProviderListProps {
  providers: Array<{
    id: string;
    name: string;
    type: ProviderType;
    status: "available" | "unavailable" | "degraded" | "unknown";
    hasApiKey: boolean;
    enabled: boolean;
  }>;
  onEdit: (providerId: string) => void;
  onAdd: () => void;
  onRefresh?: () => void;
  isRefreshing?: boolean;
}

export function ProviderList({
  providers,
  onEdit,
  onAdd,
  onRefresh,
  isRefreshing = false,
}: ProviderListProps) {
  const statusIcon = {
    available: <CheckCircle2 className="h-4 w-4 text-emerald-500" />,
    unavailable: <XCircle className="h-4 w-4 text-destructive" />,
    degraded: <AlertCircle className="h-4 w-4 text-amber-500" />,
    unknown: <AlertCircle className="h-4 w-4 text-muted-foreground" />,
  };

  return (
    <Card>
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <div>
            <CardTitle className="text-base">AI Providers</CardTitle>
            <CardDescription className="text-xs">
              Manage your AI provider configurations
            </CardDescription>
          </div>
          <div className="flex items-center gap-2">
            {onRefresh && (
              <Button
                variant="ghost"
                size="icon"
                className="h-8 w-8"
                onClick={onRefresh}
                disabled={isRefreshing}
              >
                <RefreshCw
                  className={cn("h-4 w-4", isRefreshing && "animate-spin")}
                />
              </Button>
            )}
            <Button size="sm" onClick={onAdd}>
              <Plus className="h-4 w-4 mr-1" />
              Add
            </Button>
          </div>
        </div>
      </CardHeader>
      <CardContent className="space-y-2">
        {providers.length === 0 ? (
          <div className="text-center py-6 text-muted-foreground text-sm">
            No providers configured. Add one to get started.
          </div>
        ) : (
          providers.map((provider) => (
            <div
              key={provider.id}
              className={cn(
                "flex items-center justify-between rounded-lg border p-3 cursor-pointer transition-colors hover:bg-accent/50",
                !provider.enabled && "opacity-50"
              )}
              onClick={() => onEdit(provider.id)}
            >
              <div className="flex items-center gap-3">
                {statusIcon[provider.status]}
                <div>
                  <p className="text-sm font-medium">{provider.name}</p>
                  <p className="text-xs text-muted-foreground capitalize">
                    {provider.type}
                  </p>
                </div>
              </div>
              <div className="flex items-center gap-2">
                {!provider.hasApiKey && (
                  <Badge variant="outline" className="text-[10px]">
                    No API Key
                  </Badge>
                )}
                {!provider.enabled && (
                  <Badge variant="secondary" className="text-[10px]">
                    Disabled
                  </Badge>
                )}
              </div>
            </div>
          ))
        )}
      </CardContent>
    </Card>
  );
}

export default ProviderConfigForm;
