"use client";

/**
 * API Key Manager Component
 *
 * A component for managing user API keys including:
 * - List existing API keys (masked)
 * - Create new key with name
 * - Copy key to clipboard on create
 * - Delete/revoke key
 * - Show last used timestamp
 *
 * @module components/profile/APIKeyManager
 */

import React, { useState, useCallback, useEffect } from "react";
import {
  Key,
  Plus,
  Trash2,
  Copy,
  Check,
  Loader2,
  AlertCircle,
  Clock,
  Calendar,
  Activity,
  Shield,
  RefreshCw,
} from "lucide-react";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
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
import { getApiConfig, getApiHeaders } from "@/lib/api-config";

// =============================================================================
// Types
// =============================================================================

interface APIKey {
  id: number;
  name: string | null;
  key_prefix: string;
  is_active: boolean;
  expires_at: string | null;
  last_used_at: string | null;
  usage_count: number;
  created_at: string;
  revoked_at: string | null;
}

interface APIKeyManagerProps {
  /** Additional class names */
  className?: string;
}

// =============================================================================
// Helper Functions
// =============================================================================

/**
 * Format date for display
 */
function formatDate(dateStr?: string | null): string {
  if (!dateStr) return "Never";
  try {
    const date = new Date(dateStr);
    return date.toLocaleDateString("en-US", {
      year: "numeric",
      month: "short",
      day: "numeric",
    });
  } catch {
    return "Unknown";
  }
}

/**
 * Format relative time
 */
function formatRelativeTime(dateStr?: string | null): string {
  if (!dateStr) return "Never";
  try {
    const date = new Date(dateStr);
    const now = new Date();
    const diffMs = now.getTime() - date.getTime();
    const diffDays = Math.floor(diffMs / (1000 * 60 * 60 * 24));
    const diffHours = Math.floor(diffMs / (1000 * 60 * 60));
    const diffMinutes = Math.floor(diffMs / (1000 * 60));

    if (diffMinutes < 1) return "Just now";
    if (diffMinutes < 60) return `${diffMinutes}m ago`;
    if (diffHours < 24) return `${diffHours}h ago`;
    if (diffDays < 7) return `${diffDays}d ago`;
    return formatDate(dateStr);
  } catch {
    return "Unknown";
  }
}

/**
 * Check if key is expired
 */
function isExpired(expiresAt?: string | null): boolean {
  if (!expiresAt) return false;
  try {
    return new Date(expiresAt) < new Date();
  } catch {
    return false;
  }
}

// =============================================================================
// Component
// =============================================================================

export function APIKeyManager({ className }: APIKeyManagerProps) {
  // State
  const [apiKeys, setApiKeys] = useState<APIKey[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Create dialog state
  const [isCreateDialogOpen, setIsCreateDialogOpen] = useState(false);
  const [newKeyName, setNewKeyName] = useState("");
  const [newKeyExpiry, setNewKeyExpiry] = useState<number | "">("");
  const [isCreating, setIsCreating] = useState(false);
  const [createdKey, setCreatedKey] = useState<string | null>(null);
  const [isCopied, setIsCopied] = useState(false);

  // Delete state
  const [deletingKeyId, setDeletingKeyId] = useState<number | null>(null);

  // ==========================================================================
  // Fetch API Keys
  // ==========================================================================

  const fetchApiKeys = useCallback(async () => {
    setIsLoading(true);
    setError(null);

    try {
      const apiConfig = getApiConfig();
      const headers = getApiHeaders();

      const response = await fetch(`${apiConfig.backendApiUrl}/api/v1/users/me/api-keys`, {
        method: "GET",
        headers,
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail?.error || data.detail || "Failed to fetch API keys");
      }

      setApiKeys(data.api_keys || []);
    } catch (err: unknown) {
      const error = err as { message?: string };
      setError(error.message || "Failed to load API keys");
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchApiKeys();
  }, [fetchApiKeys]);

  // ==========================================================================
  // Create API Key
  // ==========================================================================

  const handleCreateKey = useCallback(async () => {
    setIsCreating(true);
    setError(null);

    try {
      const apiConfig = getApiConfig();
      const headers = getApiHeaders();

      const body: Record<string, unknown> = {};
      if (newKeyName.trim()) {
        body.name = newKeyName.trim();
      }
      if (newKeyExpiry && newKeyExpiry > 0) {
        body.expires_in_days = newKeyExpiry;
      }

      const response = await fetch(`${apiConfig.backendApiUrl}/api/v1/users/me/api-keys`, {
        method: "POST",
        headers: {
          ...headers,
          "Content-Type": "application/json",
        },
        body: JSON.stringify(body),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail?.error || data.detail || "Failed to create API key");
      }

      // Store the created key for display
      setCreatedKey(data.api_key);

      // Refresh the list
      await fetchApiKeys();
    } catch (err: unknown) {
      const error = err as { message?: string };
      setError(error.message || "Failed to create API key");
    } finally {
      setIsCreating(false);
    }
  }, [newKeyName, newKeyExpiry, fetchApiKeys]);

  // ==========================================================================
  // Copy to Clipboard
  // ==========================================================================

  const handleCopyKey = useCallback(async (key: string) => {
    try {
      await navigator.clipboard.writeText(key);
      setIsCopied(true);
      setTimeout(() => setIsCopied(false), 2000);
    } catch {
      setError("Failed to copy to clipboard");
    }
  }, []);

  // ==========================================================================
  // Delete API Key
  // ==========================================================================

  const handleDeleteKey = useCallback(
    async (keyId: number) => {
      setDeletingKeyId(keyId);
      setError(null);

      try {
        const apiConfig = getApiConfig();
        const headers = getApiHeaders();

        const response = await fetch(
          `${apiConfig.backendApiUrl}/api/v1/users/me/api-keys/${keyId}`,
          {
            method: "DELETE",
            headers,
          }
        );

        const data = await response.json();

        if (!response.ok) {
          throw new Error(data.detail?.error || data.detail || "Failed to revoke API key");
        }

        // Refresh the list
        await fetchApiKeys();
      } catch (err: unknown) {
        const error = err as { message?: string };
        setError(error.message || "Failed to revoke API key");
      } finally {
        setDeletingKeyId(null);
      }
    },
    [fetchApiKeys]
  );

  // ==========================================================================
  // Close Create Dialog
  // ==========================================================================

  const handleCloseDialog = useCallback(() => {
    setIsCreateDialogOpen(false);
    setNewKeyName("");
    setNewKeyExpiry("");
    setCreatedKey(null);
    setIsCopied(false);
  }, []);

  // ==========================================================================
  // Render Loading State
  // ==========================================================================

  if (isLoading) {
    return (
      <div className={cn("space-y-4", className)}>
        <div className="flex items-center justify-between">
          <Skeleton className="h-6 w-32" />
          <Skeleton className="h-10 w-32" />
        </div>
        {[1, 2].map((i) => (
          <Card key={i} className="animate-pulse">
            <CardContent className="p-4">
              <div className="flex items-center justify-between">
                <div className="space-y-2">
                  <Skeleton className="h-5 w-40" />
                  <Skeleton className="h-4 w-24" />
                </div>
                <Skeleton className="h-8 w-8" />
              </div>
            </CardContent>
          </Card>
        ))}
      </div>
    );
  }

  // ==========================================================================
  // Render
  // ==========================================================================

  const activeKeys = apiKeys.filter((k) => k.is_active && !isExpired(k.expires_at));

  return (
    <div className={cn("space-y-6", className)}>
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="space-y-1">
          <h3 className="text-lg font-semibold">API Keys</h3>
          <p className="text-sm text-muted-foreground">
            Manage your API keys for programmatic access
          </p>
        </div>

        <Dialog open={isCreateDialogOpen} onOpenChange={setIsCreateDialogOpen}>
          <DialogTrigger asChild>
            <Button className="gap-2" disabled={activeKeys.length >= 10}>
              <Plus className="h-4 w-4" />
              Create Key
            </Button>
          </DialogTrigger>
          <DialogContent className="sm:max-w-md">
            <DialogHeader>
              <DialogTitle>
                {createdKey ? "API Key Created" : "Create API Key"}
              </DialogTitle>
              <DialogDescription>
                {createdKey
                  ? "Copy this key now. It won't be shown again."
                  : "Create a new API key for programmatic access."}
              </DialogDescription>
            </DialogHeader>

            {createdKey ? (
              <div className="space-y-4">
                <Alert className="bg-yellow-500/10 border-yellow-500/20">
                  <Shield className="h-4 w-4 text-yellow-400" />
                  <AlertTitle className="text-yellow-400">
                    Save This Key
                  </AlertTitle>
                  <AlertDescription className="text-muted-foreground">
                    This is the only time you&apos;ll see this key. Store it securely.
                  </AlertDescription>
                </Alert>

                <div className="flex items-center gap-2">
                  <Input
                    value={createdKey}
                    readOnly
                    className="font-mono text-sm"
                  />
                  <Button
                    type="button"
                    variant="outline"
                    size="icon"
                    onClick={() => handleCopyKey(createdKey)}
                  >
                    {isCopied ? (
                      <Check className="h-4 w-4 text-green-400" />
                    ) : (
                      <Copy className="h-4 w-4" />
                    )}
                  </Button>
                </div>

                <DialogFooter>
                  <Button onClick={handleCloseDialog}>Done</Button>
                </DialogFooter>
              </div>
            ) : (
              <div className="space-y-4">
                <div className="space-y-2">
                  <Label htmlFor="key-name">Name (optional)</Label>
                  <Input
                    id="key-name"
                    placeholder="e.g., Production, CI/CD"
                    value={newKeyName}
                    onChange={(e) => setNewKeyName(e.target.value)}
                    maxLength={100}
                  />
                </div>

                <div className="space-y-2">
                  <Label htmlFor="key-expiry">Expiration (days, optional)</Label>
                  <Input
                    id="key-expiry"
                    type="number"
                    placeholder="Leave empty for no expiration"
                    value={newKeyExpiry}
                    onChange={(e) =>
                      setNewKeyExpiry(e.target.value ? parseInt(e.target.value) : "")
                    }
                    min={1}
                    max={365}
                  />
                  <p className="text-xs text-muted-foreground">1-365 days</p>
                </div>

                <DialogFooter>
                  <Button variant="outline" onClick={handleCloseDialog}>
                    Cancel
                  </Button>
                  <Button onClick={handleCreateKey} disabled={isCreating}>
                    {isCreating ? (
                      <>
                        <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                        Creating...
                      </>
                    ) : (
                      "Create Key"
                    )}
                  </Button>
                </DialogFooter>
              </div>
            )}
          </DialogContent>
        </Dialog>
      </div>

      {/* Error Alert */}
      {error && (
        <Alert variant="destructive">
          <AlertCircle className="h-4 w-4" />
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      {/* Key Limit Warning */}
      {activeKeys.length >= 10 && (
        <Alert className="bg-yellow-500/10 border-yellow-500/20">
          <AlertCircle className="h-4 w-4 text-yellow-400" />
          <AlertDescription className="text-yellow-400">
            You have reached the maximum of 10 active API keys. Revoke unused
            keys to create new ones.
          </AlertDescription>
        </Alert>
      )}

      {/* API Keys List */}
      {apiKeys.length === 0 ? (
        <Card className="border-dashed">
          <CardContent className="flex flex-col items-center justify-center py-12 text-center">
            <Key className="h-12 w-12 text-muted-foreground/50 mb-4" />
            <h3 className="text-lg font-medium mb-2">No API Keys</h3>
            <p className="text-sm text-muted-foreground mb-4">
              Create an API key to access Chimera programmatically
            </p>
            <Button onClick={() => setIsCreateDialogOpen(true)} className="gap-2">
              <Plus className="h-4 w-4" />
              Create Your First Key
            </Button>
          </CardContent>
        </Card>
      ) : (
        <div className="space-y-3">
          {apiKeys.map((apiKey) => {
            const expired = isExpired(apiKey.expires_at);
            const inactive = !apiKey.is_active || expired;

            return (
              <Card
                key={apiKey.id}
                className={cn(
                  "transition-colors",
                  inactive && "opacity-60 bg-muted/30"
                )}
              >
                <CardContent className="p-4">
                  <div className="flex items-start justify-between gap-4">
                    <div className="flex-1 min-w-0 space-y-2">
                      {/* Key Name and Prefix */}
                      <div className="flex items-center gap-2 flex-wrap">
                        <code className="text-sm font-mono bg-muted px-2 py-0.5 rounded">
                          {apiKey.key_prefix}...
                        </code>
                        {apiKey.name && (
                          <span className="font-medium text-foreground">
                            {apiKey.name}
                          </span>
                        )}
                        {!apiKey.is_active && (
                          <Badge
                            variant="outline"
                            className="bg-red-500/10 text-red-400 border-red-500/30"
                          >
                            Revoked
                          </Badge>
                        )}
                        {expired && apiKey.is_active && (
                          <Badge
                            variant="outline"
                            className="bg-orange-500/10 text-orange-400 border-orange-500/30"
                          >
                            Expired
                          </Badge>
                        )}
                      </div>

                      {/* Key Metadata */}
                      <div className="flex items-center gap-4 text-xs text-muted-foreground flex-wrap">
                        <div className="flex items-center gap-1">
                          <Calendar className="h-3 w-3" />
                          Created {formatDate(apiKey.created_at)}
                        </div>
                        {apiKey.expires_at && (
                          <div className="flex items-center gap-1">
                            <Clock className="h-3 w-3" />
                            {expired ? "Expired" : "Expires"}{" "}
                            {formatDate(apiKey.expires_at)}
                          </div>
                        )}
                        <div className="flex items-center gap-1">
                          <Activity className="h-3 w-3" />
                          Used {apiKey.usage_count} times
                        </div>
                        {apiKey.last_used_at && (
                          <div className="flex items-center gap-1">
                            Last used {formatRelativeTime(apiKey.last_used_at)}
                          </div>
                        )}
                      </div>
                    </div>

                    {/* Actions */}
                    <div className="flex-shrink-0">
                      {apiKey.is_active && !expired && (
                        <Button
                          variant="ghost"
                          size="icon"
                          className="text-destructive hover:text-destructive hover:bg-destructive/10"
                          onClick={() => handleDeleteKey(apiKey.id)}
                          disabled={deletingKeyId === apiKey.id}
                        >
                          {deletingKeyId === apiKey.id ? (
                            <Loader2 className="h-4 w-4 animate-spin" />
                          ) : (
                            <Trash2 className="h-4 w-4" />
                          )}
                        </Button>
                      )}
                    </div>
                  </div>
                </CardContent>
              </Card>
            );
          })}
        </div>
      )}

      {/* Refresh Button */}
      <div className="flex justify-center">
        <Button
          variant="ghost"
          size="sm"
          onClick={fetchApiKeys}
          className="gap-2 text-muted-foreground"
        >
          <RefreshCw className="h-4 w-4" />
          Refresh
        </Button>
      </div>
    </div>
  );
}

export default APIKeyManager;
