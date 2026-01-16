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
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Textarea } from "@/components/ui/textarea";
import { cn } from "@/lib/utils";
import {
  Eye,
  EyeOff,
  Save,
  TestTube,
  Loader2,
  CheckCircle2,
  XCircle,
  Key,
  AlertCircle,
} from "lucide-react";

// =============================================================================
// Types
// =============================================================================

export type ApiKeyRole = "primary" | "backup" | "fallback";
export type ApiKeyStatus = "active" | "inactive" | "expired" | "rate_limited" | "invalid" | "revoked";
export type ProviderId = "google" | "openai" | "anthropic" | "deepseek" | "qwen" | "bigmodel" | "routeway" | "cursor";

export interface ApiKeyFormData {
  provider_id: ProviderId;
  api_key: string;
  name: string;
  role: ApiKeyRole;
  priority: number;
  description?: string;
  tags?: string[];
}

export interface ApiKeyFormProps {
  isOpen: boolean;
  onClose: () => void;
  onSave: (data: ApiKeyFormData) => Promise<void>;
  onTest?: (data: ApiKeyFormData) => Promise<{ success: boolean; message: string; latency_ms?: number }>;
  initialData?: Partial<ApiKeyFormData>;
  isEditing?: boolean;
  isSaving?: boolean;
  providerId?: ProviderId;
}

// =============================================================================
// Provider Configuration
// =============================================================================

const providerConfig: Record<ProviderId, { name: string; description: string; placeholder: string }> = {
  openai: {
    name: "OpenAI",
    description: "GPT-4, GPT-3.5-turbo, and other OpenAI models",
    placeholder: "sk-...",
  },
  anthropic: {
    name: "Anthropic",
    description: "Claude 3 Opus, Sonnet, Haiku models",
    placeholder: "sk-ant-...",
  },
  google: {
    name: "Google AI",
    description: "Gemini Pro, Gemini Flash models",
    placeholder: "AIzaSy...",
  },
  deepseek: {
    name: "DeepSeek",
    description: "DeepSeek Chat and Coder models",
    placeholder: "sk-...",
  },
  qwen: {
    name: "Qwen",
    description: "Alibaba Qwen models",
    placeholder: "sk-...",
  },
  bigmodel: {
    name: "BigModel",
    description: "GLM-4 and other BigModel models",
    placeholder: "...",
  },
  routeway: {
    name: "Routeway",
    description: "Routeway API gateway",
    placeholder: "...",
  },
  cursor: {
    name: "Cursor",
    description: "Cursor AI coding assistant",
    placeholder: "...",
  },
};

const roleConfig: Record<ApiKeyRole, { label: string; description: string; color: string }> = {
  primary: {
    label: "Primary",
    description: "Main key used for all requests",
    color: "text-emerald-500",
  },
  backup: {
    label: "Backup",
    description: "Used when primary is rate limited",
    color: "text-amber-500",
  },
  fallback: {
    label: "Fallback",
    description: "Last resort when backup is unavailable",
    color: "text-blue-500",
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
  placeholder = "Enter API key...",
  disabled = false,
  hasExistingKey = false,
}: ApiKeyInputProps) {
  const [showKey, setShowKey] = useState(false);

  return (
    <div className="relative">
      <Key className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
      <Input
        type={showKey ? "text" : "password"}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder={hasExistingKey ? "••••••••••••••••" : placeholder}
        disabled={disabled}
        className="pl-10 pr-10 font-mono text-sm"
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
// Test Result Display
// =============================================================================

interface TestResultProps {
  result: { success: boolean; message: string; latency_ms?: number } | null;
  isTesting?: boolean;
}

function TestResult({ result, isTesting }: TestResultProps) {
  if (isTesting) {
    return (
      <div className="flex items-center gap-2 rounded-md p-3 text-sm bg-muted/50">
        <Loader2 className="h-4 w-4 animate-spin text-muted-foreground" />
        <span className="text-muted-foreground">Testing connection...</span>
      </div>
    );
  }

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
      <span className="flex-1">{result.message}</span>
      {result.latency_ms && result.success && (
        <Badge variant="outline" className="text-[10px]">
          {result.latency_ms.toFixed(0)}ms
        </Badge>
      )}
    </div>
  );
}

// =============================================================================
// Main API Key Form Component
// =============================================================================

export function ApiKeyForm({
  isOpen,
  onClose,
  onSave,
  onTest,
  initialData,
  isEditing = false,
  isSaving = false,
  providerId,
}: ApiKeyFormProps) {
  // Form state
  const [formData, setFormData] = useState<ApiKeyFormData>({
    provider_id: providerId || initialData?.provider_id || "openai",
    api_key: "",
    name: initialData?.name || "",
    role: initialData?.role || "primary",
    priority: initialData?.priority ?? 0,
    description: initialData?.description || "",
    tags: initialData?.tags || [],
  });

  const [testResult, setTestResult] = useState<{
    success: boolean;
    message: string;
    latency_ms?: number;
  } | null>(null);

  const [isTesting, setIsTesting] = useState(false);
  const [tagInput, setTagInput] = useState("");

  // Get provider config
  const provider = providerConfig[formData.provider_id];

  // Update form field
  const updateField = useCallback(
    <K extends keyof ApiKeyFormData>(field: K, value: ApiKeyFormData[K]) => {
      setFormData((prev) => ({ ...prev, [field]: value }));
      setTestResult(null);
    },
    []
  );

  // Handle provider change
  const handleProviderChange = useCallback((newProvider: ProviderId) => {
    setFormData((prev) => ({
      ...prev,
      provider_id: newProvider,
      name: prev.name || providerConfig[newProvider].name,
    }));
    setTestResult(null);
  }, []);

  // Handle save
  const handleSave = useCallback(async () => {
    try {
      await onSave(formData);
      onClose();
    } catch (error) {
      // Error handled by parent
    }
  }, [formData, onSave, onClose]);

  // Handle test
  const handleTest = useCallback(async () => {
    if (!onTest) return;
    setIsTesting(true);
    try {
      const result = await onTest(formData);
      setTestResult(result);
    } catch (error) {
      setTestResult({
        success: false,
        message: error instanceof Error ? error.message : "Test failed",
      });
    } finally {
      setIsTesting(false);
    }
  }, [formData, onTest]);

  // Handle tag input
  const handleAddTag = useCallback(() => {
    if (tagInput.trim() && !formData.tags?.includes(tagInput.trim())) {
      updateField("tags", [...(formData.tags || []), tagInput.trim().toLowerCase()]);
      setTagInput("");
    }
  }, [tagInput, formData.tags, updateField]);

  const handleRemoveTag = useCallback(
    (tag: string) => {
      updateField(
        "tags",
        formData.tags?.filter((t) => t !== tag) || []
      );
    },
    [formData.tags, updateField]
  );

  // Validation
  const isValid =
    formData.name.trim() &&
    formData.provider_id &&
    (formData.api_key || isEditing);

  return (
    <Dialog open={isOpen} onOpenChange={(open) => !open && onClose()}>
      <DialogContent className="max-w-lg">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <Key className="h-5 w-5" />
            {isEditing ? "Edit API Key" : "Add API Key"}
          </DialogTitle>
          <DialogDescription>
            {isEditing
              ? "Update the configuration for this API key."
              : "Add a new API key for secure provider access."}
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-4 py-4">
          {/* Provider Selection */}
          {!providerId && (
            <div className="space-y-2">
              <label className="text-sm font-medium">Provider</label>
              <Select
                value={formData.provider_id}
                onValueChange={(v) => handleProviderChange(v as ProviderId)}
                disabled={isEditing}
              >
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {Object.entries(providerConfig).map(([id, config]) => (
                    <SelectItem key={id} value={id}>
                      <div className="flex flex-col">
                        <span className="font-medium">{config.name}</span>
                        <span className="text-[10px] text-muted-foreground">
                          {config.description}
                        </span>
                      </div>
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
          )}

          {/* Key Name */}
          <div className="space-y-2">
            <label className="text-sm font-medium">Name</label>
            <Input
              value={formData.name}
              onChange={(e) => updateField("name", e.target.value)}
              placeholder={`${provider?.name || "Provider"} Key`}
            />
            <p className="text-xs text-muted-foreground">
              A descriptive name to identify this key
            </p>
          </div>

          {/* API Key */}
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
              value={formData.api_key}
              onChange={(v) => updateField("api_key", v)}
              placeholder={provider?.placeholder}
              hasExistingKey={isEditing}
            />
          </div>

          {/* Role Selection */}
          <div className="space-y-2">
            <label className="text-sm font-medium">Role</label>
            <Select
              value={formData.role}
              onValueChange={(v) => updateField("role", v as ApiKeyRole)}
            >
              <SelectTrigger>
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {Object.entries(roleConfig).map(([role, config]) => (
                  <SelectItem key={role} value={role}>
                    <div className="flex items-center gap-2">
                      <span className={cn("font-medium", config.color)}>
                        {config.label}
                      </span>
                      <span className="text-[10px] text-muted-foreground">
                        — {config.description}
                      </span>
                    </div>
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          {/* Priority (for backup/fallback keys) */}
          {formData.role !== "primary" && (
            <div className="space-y-2">
              <label className="text-sm font-medium">Priority</label>
              <Input
                type="number"
                value={formData.priority}
                onChange={(e) => updateField("priority", parseInt(e.target.value) || 0)}
                min={0}
                max={100}
              />
              <p className="text-xs text-muted-foreground">
                Lower priority = tried first during failover (0-100)
              </p>
            </div>
          )}

          {/* Description */}
          <div className="space-y-2">
            <label className="text-sm font-medium">
              Description <span className="text-muted-foreground">(optional)</span>
            </label>
            <Textarea
              value={formData.description || ""}
              onChange={(e) => updateField("description", e.target.value)}
              placeholder="Add notes about this API key..."
              className="min-h-[80px] resize-none"
            />
          </div>

          {/* Tags */}
          <div className="space-y-2">
            <label className="text-sm font-medium">
              Tags <span className="text-muted-foreground">(optional)</span>
            </label>
            <div className="flex gap-2">
              <Input
                value={tagInput}
                onChange={(e) => setTagInput(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && (e.preventDefault(), handleAddTag())}
                placeholder="Add tag..."
                className="flex-1"
              />
              <Button
                type="button"
                variant="outline"
                size="sm"
                onClick={handleAddTag}
                disabled={!tagInput.trim()}
              >
                Add
              </Button>
            </div>
            {formData.tags && formData.tags.length > 0 && (
              <div className="flex flex-wrap gap-1.5 mt-2">
                {formData.tags.map((tag) => (
                  <Badge
                    key={tag}
                    variant="secondary"
                    className="text-xs cursor-pointer hover:bg-destructive/20"
                    onClick={() => handleRemoveTag(tag)}
                  >
                    {tag} ×
                  </Badge>
                ))}
              </div>
            )}
          </div>

          {/* Test Result */}
          <TestResult result={testResult} isTesting={isTesting} />
        </div>

        <DialogFooter className="flex-col gap-2 sm:flex-row">
          {/* Test button */}
          {onTest && (
            <Button
              type="button"
              variant="outline"
              onClick={handleTest}
              disabled={isTesting || !formData.api_key}
              className="sm:mr-auto"
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
            {isEditing ? "Update Key" : "Add Key"}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

export default ApiKeyForm;
