"use client";

import Link from "next/link";
import { ApiKeyManager } from "@/components/api-keys";
import { useApiKeys } from "@/hooks";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import { Skeleton } from "@/components/ui/skeleton";
import {
  Key,
  Shield,
  ArrowLeft,
  Activity,
  CheckCircle2,
  AlertCircle,
  AlertTriangle,
  RefreshCw,
  Settings,
  Zap,
} from "lucide-react";

/**
 * API Keys Dashboard Page
 *
 * Main dashboard for managing LLM provider API keys with:
 * - Overview cards showing key statistics
 * - Full API Key Manager component with tabbed provider view
 * - Quick links to related settings
 */
export default function ApiKeysPage() {
  const {
    keys,
    providers,
    isLoading,
    error,
    refresh,
    createKey,
    updateKey,
    deleteKey,
    testKey,
    testNewKey,
    activateKey,
    deactivateKey,
    revokeKey,
  } = useApiKeys({ autoRefresh: true, refreshInterval: 60000 });

  // Calculate statistics
  const totalKeys = keys.length;
  const activeKeys = keys.filter((k) => k.status === "active").length;
  const configuredProviders = providers.filter((p) => p.has_valid_key).length;
  const totalProviders = providers.length;
  const hasIssues = keys.some((k) => k.status === "rate_limited" || k.status === "expired");

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
        <div className="flex items-center gap-4">
          <Link href="/dashboard/settings">
            <Button variant="ghost" size="icon" className="hidden sm:flex">
              <ArrowLeft className="h-4 w-4" />
            </Button>
          </Link>
          <div>
            <h1 className="text-3xl font-bold tracking-tight flex items-center gap-2">
              <Shield className="h-8 w-8 text-primary" />
              API Key Management
            </h1>
            <p className="text-muted-foreground">
              Securely manage your LLM provider API keys with encrypted storage and automatic failover.
            </p>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <Link href="/dashboard/health">
            <Button variant="outline" size="sm">
              <Activity className="h-4 w-4 mr-2" />
              Provider Health
            </Button>
          </Link>
          <Link href="/dashboard/settings">
            <Button variant="outline" size="sm">
              <Settings className="h-4 w-4 mr-2" />
              Settings
            </Button>
          </Link>
        </div>
      </div>

      <Separator />

      {/* Overview Cards */}
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
        {/* Total Keys Card */}
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Total API Keys</CardTitle>
            <Key className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            {isLoading ? (
              <Skeleton className="h-8 w-16" />
            ) : (
              <div className="text-2xl font-bold">{totalKeys}</div>
            )}
            <p className="text-xs text-muted-foreground">
              Across all providers
            </p>
          </CardContent>
        </Card>

        {/* Active Keys Card */}
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Active Keys</CardTitle>
            <CheckCircle2 className="h-4 w-4 text-emerald-500" />
          </CardHeader>
          <CardContent>
            {isLoading ? (
              <Skeleton className="h-8 w-16" />
            ) : (
              <div className="text-2xl font-bold text-emerald-500">{activeKeys}</div>
            )}
            <p className="text-xs text-muted-foreground">
              Ready for use
            </p>
          </CardContent>
        </Card>

        {/* Configured Providers Card */}
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Configured Providers</CardTitle>
            <Zap className="h-4 w-4 text-blue-500" />
          </CardHeader>
          <CardContent>
            {isLoading ? (
              <Skeleton className="h-8 w-16" />
            ) : (
              <div className="text-2xl font-bold text-blue-500">
                {configuredProviders}/{totalProviders}
              </div>
            )}
            <p className="text-xs text-muted-foreground">
              With valid keys
            </p>
          </CardContent>
        </Card>

        {/* Status Card */}
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Status</CardTitle>
            {hasIssues ? (
              <AlertTriangle className="h-4 w-4 text-yellow-500" />
            ) : error ? (
              <AlertCircle className="h-4 w-4 text-red-500" />
            ) : (
              <CheckCircle2 className="h-4 w-4 text-emerald-500" />
            )}
          </CardHeader>
          <CardContent>
            {isLoading ? (
              <Skeleton className="h-8 w-16" />
            ) : error ? (
              <Badge variant="destructive">Error</Badge>
            ) : hasIssues ? (
              <Badge variant="secondary" className="bg-yellow-500/10 text-yellow-500 hover:bg-yellow-500/20">
                Issues Detected
              </Badge>
            ) : (
              <Badge variant="secondary" className="bg-emerald-500/10 text-emerald-500 hover:bg-emerald-500/20">
                All Good
              </Badge>
            )}
            <p className="text-xs text-muted-foreground mt-1">
              {error ? "Connection error" : hasIssues ? "Some keys need attention" : "No issues detected"}
            </p>
          </CardContent>
        </Card>
      </div>

      {/* API Key Manager Component */}
      <Card>
        <CardHeader className="pb-4">
          <div className="flex items-center justify-between">
            <div>
              <CardTitle className="flex items-center gap-2">
                <Key className="h-5 w-5" />
                Manage API Keys
              </CardTitle>
              <CardDescription>
                Add, edit, test, and manage API keys for all LLM providers.
              </CardDescription>
            </div>
            <Button
              variant="outline"
              size="sm"
              onClick={() => refresh()}
              disabled={isLoading}
            >
              <RefreshCw className={`h-4 w-4 mr-2 ${isLoading ? "animate-spin" : ""}`} />
              Refresh
            </Button>
          </div>
        </CardHeader>
        <CardContent>
          <ApiKeyManager
            keys={keys}
            providers={providers}
            isLoading={isLoading}
            error={error}
            onCreateKey={createKey}
            onUpdateKey={updateKey}
            onDeleteKey={deleteKey}
            onTestKey={testKey}
            onTestNewKey={testNewKey}
            onActivateKey={activateKey}
            onDeactivateKey={deactivateKey}
            onRevokeKey={revokeKey}
            onRefresh={refresh}
          />
        </CardContent>
      </Card>

      {/* Quick Links */}
      <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
        <Card className="hover:bg-muted/50 transition-colors">
          <Link href="/dashboard/health">
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-base">
                <Activity className="h-5 w-5 text-blue-500" />
                Provider Health Dashboard
              </CardTitle>
              <CardDescription>
                Monitor real-time health status, latency, and error rates for all configured providers.
              </CardDescription>
            </CardHeader>
          </Link>
        </Card>

        <Card className="hover:bg-muted/50 transition-colors">
          <Link href="/dashboard/providers">
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-base">
                <Zap className="h-5 w-5 text-emerald-500" />
                Provider Settings
              </CardTitle>
              <CardDescription>
                Configure default models, rate limits, and failover strategies for each provider.
              </CardDescription>
            </CardHeader>
          </Link>
        </Card>
      </div>
    </div>
  );
}
