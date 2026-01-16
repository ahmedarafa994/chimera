"use client";

import * as React from "react";
import { useState, useCallback } from "react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { ScrollArea } from "@/components/ui/scroll-area";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { Skeleton } from "@/components/ui/skeleton";
import { cn } from "@/lib/utils";
import {
  Key,
  MoreVertical,
  TestTube,
  Trash2,
  Edit,
  CheckCircle2,
  XCircle,
  AlertCircle,
  Clock,
  Shield,
  ShieldAlert,
  ShieldOff,
  Loader2,
  RefreshCw,
  Plus,
  Ban,
  Play,
} from "lucide-react";
import type { ApiKeyRole, ApiKeyStatus, ProviderId } from "./ApiKeyForm";

// =============================================================================
// Types
// =============================================================================

export interface ApiKeyItem {
  id: string;
  provider_id: ProviderId;
  name: string;
  masked_key: string;
  role: ApiKeyRole;
  status: ApiKeyStatus;
  priority: number;
  created_at: string;
  updated_at: string;
  last_used_at?: string;
  request_count: number;
  successful_requests: number;
  failed_requests: number;
  description?: string;
  tags?: string[];
  is_expired: boolean;
  is_rate_limited: boolean;
}

export interface ApiKeyListProps {
  keys: ApiKeyItem[];
  isLoading?: boolean;
  onEdit?: (key: ApiKeyItem) => void;
  onDelete?: (keyId: string) => Promise<void>;
  onTest?: (keyId: string) => Promise<{ success: boolean; message: string; latency_ms?: number }>;
  onActivate?: (keyId: string) => Promise<void>;
  onDeactivate?: (keyId: string) => Promise<void>;
  onRevoke?: (keyId: string) => Promise<void>;
  onAdd?: () => void;
  onRefresh?: () => void;
  isRefreshing?: boolean;
  emptyMessage?: string;
}

// =============================================================================
// Status Configuration
// =============================================================================

const statusConfig: Record<
  ApiKeyStatus,
  { label: string; icon: React.ElementType; color: string; bgColor: string }
> = {
  active: {
    label: "Active",
    icon: CheckCircle2,
    color: "text-emerald-500",
    bgColor: "bg-emerald-500/10",
  },
  inactive: {
    label: "Inactive",
    icon: Clock,
    color: "text-zinc-500",
    bgColor: "bg-zinc-500/10",
  },
  expired: {
    label: "Expired",
    icon: XCircle,
    color: "text-red-500",
    bgColor: "bg-red-500/10",
  },
  rate_limited: {
    label: "Rate Limited",
    icon: AlertCircle,
    color: "text-amber-500",
    bgColor: "bg-amber-500/10",
  },
  invalid: {
    label: "Invalid",
    icon: ShieldAlert,
    color: "text-red-500",
    bgColor: "bg-red-500/10",
  },
  revoked: {
    label: "Revoked",
    icon: ShieldOff,
    color: "text-zinc-400",
    bgColor: "bg-zinc-400/10",
  },
};

const roleConfig: Record<ApiKeyRole, { label: string; color: string }> = {
  primary: { label: "Primary", color: "text-emerald-500" },
  backup: { label: "Backup", color: "text-amber-500" },
  fallback: { label: "Fallback", color: "text-blue-500" },
};

// =============================================================================
// API Key Item Component
// =============================================================================

interface ApiKeyItemCardProps {
  keyItem: ApiKeyItem;
  onEdit?: () => void;
  onDelete?: () => void;
  onTest?: () => Promise<{ success: boolean; message: string; latency_ms?: number }>;
  onActivate?: () => void;
  onDeactivate?: () => void;
  onRevoke?: () => void;
}

function ApiKeyItemCard({
  keyItem,
  onEdit,
  onDelete,
  onTest,
  onActivate,
  onDeactivate,
  onRevoke,
}: ApiKeyItemCardProps) {
  const [isTesting, setIsTesting] = useState(false);
  const [testResult, setTestResult] = useState<{
    success: boolean;
    message: string;
    latency_ms?: number;
  } | null>(null);
  const [showDeleteDialog, setShowDeleteDialog] = useState(false);

  const status = statusConfig[keyItem.status];
  const role = roleConfig[keyItem.role];
  const StatusIcon = status.icon;

  const handleTest = useCallback(async () => {
    if (!onTest) return;
    setIsTesting(true);
    setTestResult(null);
    try {
      const result = await onTest();
      setTestResult(result);
      // Clear result after 5 seconds
      setTimeout(() => setTestResult(null), 5000);
    } catch (error) {
      setTestResult({
        success: false,
        message: error instanceof Error ? error.message : "Test failed",
      });
    } finally {
      setIsTesting(false);
    }
  }, [onTest]);

  const formatDate = (dateStr: string | undefined) => {
    if (!dateStr) return "Never";
    const date = new Date(dateStr);
    return date.toLocaleDateString(undefined, {
      month: "short",
      day: "numeric",
      year: "numeric",
    });
  };

  const formatTimeAgo = (dateStr: string | undefined) => {
    if (!dateStr) return "Never";
    const date = new Date(dateStr);
    const now = new Date();
    const diff = now.getTime() - date.getTime();
    const minutes = Math.floor(diff / 60000);
    const hours = Math.floor(minutes / 60);
    const days = Math.floor(hours / 24);

    if (minutes < 1) return "Just now";
    if (minutes < 60) return `${minutes}m ago`;
    if (hours < 24) return `${hours}h ago`;
    if (days < 7) return `${days}d ago`;
    return formatDate(dateStr);
  };

  const successRate =
    keyItem.request_count > 0
      ? ((keyItem.successful_requests / keyItem.request_count) * 100).toFixed(1)
      : "N/A";

  return (
    <>
      <div
        className={cn(
          "group relative rounded-lg border p-4 transition-all hover:shadow-md",
          keyItem.status === "revoked" && "opacity-60",
          status.bgColor
        )}
      >
        {/* Header */}
        <div className="flex items-start justify-between gap-3">
          <div className="flex items-start gap-3 min-w-0">
            <div className={cn("p-2 rounded-lg", status.bgColor)}>
              <Key className={cn("h-4 w-4", status.color)} />
            </div>
            <div className="min-w-0 flex-1">
              <div className="flex items-center gap-2 flex-wrap">
                <h4 className="font-medium text-sm truncate">{keyItem.name}</h4>
                <Badge variant="outline" className={cn("text-[10px]", role.color)}>
                  {role.label}
                </Badge>
                <Badge
                  variant="outline"
                  className={cn("text-[10px] flex items-center gap-1", status.color)}
                >
                  <StatusIcon className="h-3 w-3" />
                  {status.label}
                </Badge>
              </div>
              <p className="text-xs text-muted-foreground font-mono mt-1">
                {keyItem.masked_key}
              </p>
            </div>
          </div>

          {/* Actions Menu */}
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button variant="ghost" size="icon" className="h-8 w-8 shrink-0">
                <MoreVertical className="h-4 w-4" />
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end">
              {onTest && (
                <DropdownMenuItem onClick={handleTest} disabled={isTesting}>
                  {isTesting ? (
                    <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                  ) : (
                    <TestTube className="h-4 w-4 mr-2" />
                  )}
                  Test Connection
                </DropdownMenuItem>
              )}
              {onEdit && (
                <DropdownMenuItem onClick={onEdit}>
                  <Edit className="h-4 w-4 mr-2" />
                  Edit
                </DropdownMenuItem>
              )}
              <DropdownMenuSeparator />
              {keyItem.status === "inactive" && onActivate && (
                <DropdownMenuItem onClick={onActivate}>
                  <Play className="h-4 w-4 mr-2" />
                  Activate
                </DropdownMenuItem>
              )}
              {keyItem.status === "active" && onDeactivate && (
                <DropdownMenuItem onClick={onDeactivate}>
                  <Clock className="h-4 w-4 mr-2" />
                  Deactivate
                </DropdownMenuItem>
              )}
              {keyItem.status !== "revoked" && onRevoke && (
                <DropdownMenuItem onClick={onRevoke} className="text-amber-600">
                  <Ban className="h-4 w-4 mr-2" />
                  Revoke
                </DropdownMenuItem>
              )}
              <DropdownMenuSeparator />
              {onDelete && (
                <DropdownMenuItem
                  onClick={() => setShowDeleteDialog(true)}
                  className="text-destructive"
                >
                  <Trash2 className="h-4 w-4 mr-2" />
                  Delete
                </DropdownMenuItem>
              )}
            </DropdownMenuContent>
          </DropdownMenu>
        </div>

        {/* Test Result */}
        {(isTesting || testResult) && (
          <div
            className={cn(
              "mt-3 flex items-center gap-2 rounded-md p-2 text-xs",
              isTesting && "bg-muted/50",
              testResult?.success && "bg-emerald-500/10 text-emerald-600",
              testResult && !testResult.success && "bg-destructive/10 text-destructive"
            )}
          >
            {isTesting ? (
              <>
                <Loader2 className="h-3 w-3 animate-spin" />
                <span>Testing...</span>
              </>
            ) : testResult ? (
              <>
                {testResult.success ? (
                  <CheckCircle2 className="h-3 w-3" />
                ) : (
                  <XCircle className="h-3 w-3" />
                )}
                <span className="flex-1">{testResult.message}</span>
                {testResult.latency_ms && (
                  <Badge variant="outline" className="text-[9px]">
                    {testResult.latency_ms.toFixed(0)}ms
                  </Badge>
                )}
              </>
            ) : null}
          </div>
        )}

        {/* Stats */}
        <div className="mt-3 grid grid-cols-3 gap-3 text-xs">
          <TooltipProvider>
            <Tooltip>
              <TooltipTrigger asChild>
                <div className="flex flex-col">
                  <span className="text-muted-foreground">Requests</span>
                  <span className="font-medium">
                    {keyItem.request_count.toLocaleString()}
                  </span>
                </div>
              </TooltipTrigger>
              <TooltipContent>
                <p>
                  {keyItem.successful_requests.toLocaleString()} successful,{" "}
                  {keyItem.failed_requests.toLocaleString()} failed
                </p>
              </TooltipContent>
            </Tooltip>
          </TooltipProvider>

          <div className="flex flex-col">
            <span className="text-muted-foreground">Success Rate</span>
            <span
              className={cn(
                "font-medium",
                parseFloat(successRate) >= 95
                  ? "text-emerald-500"
                  : parseFloat(successRate) >= 80
                  ? "text-amber-500"
                  : "text-red-500"
              )}
            >
              {successRate}%
            </span>
          </div>

          <TooltipProvider>
            <Tooltip>
              <TooltipTrigger asChild>
                <div className="flex flex-col">
                  <span className="text-muted-foreground">Last Used</span>
                  <span className="font-medium">
                    {formatTimeAgo(keyItem.last_used_at)}
                  </span>
                </div>
              </TooltipTrigger>
              <TooltipContent>
                <p>Created: {formatDate(keyItem.created_at)}</p>
              </TooltipContent>
            </Tooltip>
          </TooltipProvider>
        </div>

        {/* Tags */}
        {keyItem.tags && keyItem.tags.length > 0 && (
          <div className="mt-3 flex flex-wrap gap-1">
            {keyItem.tags.slice(0, 3).map((tag) => (
              <Badge key={tag} variant="secondary" className="text-[9px]">
                {tag}
              </Badge>
            ))}
            {keyItem.tags.length > 3 && (
              <Badge variant="secondary" className="text-[9px]">
                +{keyItem.tags.length - 3} more
              </Badge>
            )}
          </div>
        )}
      </div>

      {/* Delete Confirmation Dialog */}
      <AlertDialog open={showDeleteDialog} onOpenChange={setShowDeleteDialog}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Delete API Key</AlertDialogTitle>
            <AlertDialogDescription>
              Are you sure you want to delete &quot;{keyItem.name}&quot;? This action cannot be
              undone and the key will be permanently removed.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Cancel</AlertDialogCancel>
            <AlertDialogAction
              onClick={() => {
                onDelete?.();
                setShowDeleteDialog(false);
              }}
              className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
            >
              Delete
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </>
  );
}

// =============================================================================
// Loading Skeleton
// =============================================================================

function ApiKeyListSkeleton() {
  return (
    <div className="space-y-3">
      {[1, 2, 3].map((i) => (
        <div key={i} className="rounded-lg border p-4 space-y-3">
          <div className="flex items-start gap-3">
            <Skeleton className="h-8 w-8 rounded-lg" />
            <div className="flex-1 space-y-2">
              <Skeleton className="h-4 w-32" />
              <Skeleton className="h-3 w-24" />
            </div>
          </div>
          <div className="grid grid-cols-3 gap-3">
            <Skeleton className="h-8" />
            <Skeleton className="h-8" />
            <Skeleton className="h-8" />
          </div>
        </div>
      ))}
    </div>
  );
}

// =============================================================================
// Main API Key List Component
// =============================================================================

export function ApiKeyList({
  keys,
  isLoading,
  onEdit,
  onDelete,
  onTest,
  onActivate,
  onDeactivate,
  onRevoke,
  onAdd,
  onRefresh,
  isRefreshing,
  emptyMessage = "No API keys configured",
}: ApiKeyListProps) {
  const [deletingIds, setDeletingIds] = useState<Set<string>>(new Set());

  const handleDelete = useCallback(
    async (keyId: string) => {
      if (!onDelete) return;
      setDeletingIds((prev) => new Set(prev).add(keyId));
      try {
        await onDelete(keyId);
      } finally {
        setDeletingIds((prev) => {
          const next = new Set(prev);
          next.delete(keyId);
          return next;
        });
      }
    },
    [onDelete]
  );

  if (isLoading) {
    return (
      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-base flex items-center gap-2">
            <Shield className="h-4 w-4" />
            API Keys
          </CardTitle>
          <CardDescription>Loading your API keys...</CardDescription>
        </CardHeader>
        <CardContent>
          <ApiKeyListSkeleton />
        </CardContent>
      </Card>
    );
  }

  return (
    <Card>
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <div>
            <CardTitle className="text-base flex items-center gap-2">
              <Shield className="h-4 w-4" />
              API Keys
            </CardTitle>
            <CardDescription>
              {keys.length} key{keys.length !== 1 ? "s" : ""} configured
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
            {onAdd && (
              <Button size="sm" onClick={onAdd}>
                <Plus className="h-4 w-4 mr-1" />
                Add Key
              </Button>
            )}
          </div>
        </div>
      </CardHeader>
      <CardContent>
        {keys.length === 0 ? (
          <div className="flex flex-col items-center justify-center py-8 text-center">
            <Shield className="h-10 w-10 text-muted-foreground/50 mb-3" />
            <p className="text-sm text-muted-foreground">{emptyMessage}</p>
            {onAdd && (
              <Button variant="outline" className="mt-4" onClick={onAdd}>
                <Plus className="h-4 w-4 mr-2" />
                Add Your First Key
              </Button>
            )}
          </div>
        ) : (
          <ScrollArea className="max-h-[500px]">
            <div className="space-y-3">
              {keys.map((keyItem) => (
                <ApiKeyItemCard
                  key={keyItem.id}
                  keyItem={keyItem}
                  onEdit={onEdit ? () => onEdit(keyItem) : undefined}
                  onDelete={
                    onDelete && !deletingIds.has(keyItem.id)
                      ? () => handleDelete(keyItem.id)
                      : undefined
                  }
                  onTest={onTest ? () => onTest(keyItem.id) : undefined}
                  onActivate={onActivate ? () => onActivate(keyItem.id) : undefined}
                  onDeactivate={onDeactivate ? () => onDeactivate(keyItem.id) : undefined}
                  onRevoke={onRevoke ? () => onRevoke(keyItem.id) : undefined}
                />
              ))}
            </div>
          </ScrollArea>
        )}
      </CardContent>
    </Card>
  );
}

export default ApiKeyList;
