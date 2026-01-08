"use client";

import React, { useState, useEffect, useCallback } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Separator } from "@/components/ui/separator";
import { apiClient } from "@/lib/api-enhanced";
import { toast } from "sonner";
import { 
  Loader2, 
  RefreshCw, 
  CheckCircle, 
  XCircle, 
  AlertTriangle,
  Activity,
  Server,
  Database,
  Cpu,
  Clock,
  Zap,
  Heart
} from "lucide-react";

interface HealthCheckResult {
  name: string;
  status: "healthy" | "degraded" | "unhealthy" | "unknown";
  message: string;
  latency_ms: number;
  details: Record<string, unknown>;
  timestamp: string;
}

interface OverallHealth {
  status: "healthy" | "degraded" | "unhealthy" | "unknown";
  version: string;
  environment: string;
  uptime_seconds: number;
  checks: HealthCheckResult[];
  timestamp: string;
}

interface HealthDashboardProps {
  autoRefresh?: boolean;
  refreshInterval?: number;
  compact?: boolean;
}

const statusConfig = {
  healthy: {
    icon: CheckCircle,
    color: "text-green-500",
    bgColor: "bg-green-500/10",
    borderColor: "border-green-500",
    label: "Healthy",
  },
  degraded: {
    icon: AlertTriangle,
    color: "text-yellow-500",
    bgColor: "bg-yellow-500/10",
    borderColor: "border-yellow-500",
    label: "Degraded",
  },
  unhealthy: {
    icon: XCircle,
    color: "text-red-500",
    bgColor: "bg-red-500/10",
    borderColor: "border-red-500",
    label: "Unhealthy",
  },
  unknown: {
    icon: AlertTriangle,
    color: "text-gray-500",
    bgColor: "bg-gray-500/10",
    borderColor: "border-gray-500",
    label: "Unknown",
  },
};

const serviceIcons: Record<string, React.ElementType> = {
  database: Database,
  redis: Server,
  llm_service: Cpu,
  transformation_engine: Zap,
  cache: Activity,
  liveness: Heart,
  readiness: CheckCircle,
};

function formatUptime(seconds: number): string {
  const days = Math.floor(seconds / 86400);
  const hours = Math.floor((seconds % 86400) / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  const secs = Math.floor(seconds % 60);

  const parts = [];
  if (days > 0) parts.push(`${days}d`);
  if (hours > 0) parts.push(`${hours}h`);
  if (minutes > 0) parts.push(`${minutes}m`);
  if (secs > 0 || parts.length === 0) parts.push(`${secs}s`);

  return parts.join(" ");
}

export function HealthDashboard({ 
  autoRefresh = true, 
  refreshInterval = 30000,
  compact = false 
}: HealthDashboardProps) {
  const [health, setHealth] = useState<OverallHealth | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [lastRefresh, setLastRefresh] = useState<Date | null>(null);

  const fetchHealth = useCallback(async () => {
    try {
      const response = await apiClient.get<OverallHealth>("/health");
      setHealth(response.data);
      setError(null);
      setLastRefresh(new Date());
    } catch (err) {
      console.error("Failed to fetch health status:", err);
      setError("Failed to connect to backend");
      setHealth(null);
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchHealth();

    if (autoRefresh) {
      const interval = setInterval(fetchHealth, refreshInterval);
      return () => clearInterval(interval);
    }
  }, [fetchHealth, autoRefresh, refreshInterval]);

  const handleRefresh = async () => {
    setIsLoading(true);
    await fetchHealth();
    toast.success("Health status refreshed");
  };

  if (isLoading && !health) {
    return (
      <Card className={compact ? "p-4" : ""}>
        <CardContent className="flex items-center justify-center py-8">
          <Loader2 className="h-6 w-6 animate-spin mr-2" />
          <span>Loading health status...</span>
        </CardContent>
      </Card>
    );
  }

  if (error && !health) {
    return (
      <Card className="border-red-500">
        <CardContent className="flex flex-col items-center justify-center py-8">
          <XCircle className="h-12 w-12 text-red-500 mb-4" />
          <p className="text-red-500 font-medium">{error}</p>
          <Button variant="outline" onClick={handleRefresh} className="mt-4">
            <RefreshCw className="h-4 w-4 mr-2" />
            Retry
          </Button>
        </CardContent>
      </Card>
    );
  }

  if (!health) return null;

  const config = statusConfig[health.status];
  const StatusIcon = config.icon;

  if (compact) {
    return (
      <div className={`flex items-center gap-3 p-3 rounded-lg border ${config.bgColor} ${config.borderColor}`}>
        <StatusIcon className={`h-5 w-5 ${config.color}`} />
        <div className="flex-1">
          <span className="font-medium">{config.label}</span>
          <span className="text-sm text-muted-foreground ml-2">
            v{health.version} • {formatUptime(health.uptime_seconds)}
          </span>
        </div>
        <Button variant="ghost" size="sm" onClick={handleRefresh} disabled={isLoading}>
          <RefreshCw className={`h-4 w-4 ${isLoading ? "animate-spin" : ""}`} />
        </Button>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Overall Status Card */}
      <Card className={`border-2 ${config.borderColor}`}>
        <CardHeader>
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className={`p-3 rounded-full ${config.bgColor}`}>
                <StatusIcon className={`h-8 w-8 ${config.color}`} />
              </div>
              <div>
                <CardTitle className="text-2xl">System Health</CardTitle>
                <CardDescription>
                  Overall status of the Chimera platform
                </CardDescription>
              </div>
            </div>
            <div className="flex items-center gap-2">
              <Badge variant="outline" className={config.color}>
                {config.label}
              </Badge>
              <Button variant="ghost" size="icon" onClick={handleRefresh} disabled={isLoading}>
                <RefreshCw className={`h-4 w-4 ${isLoading ? "animate-spin" : ""}`} />
              </Button>
            </div>
          </div>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="text-center p-3 rounded-lg bg-muted/50">
              <div className="text-2xl font-bold">{health.version}</div>
              <div className="text-xs text-muted-foreground">Version</div>
            </div>
            <div className="text-center p-3 rounded-lg bg-muted/50">
              <div className="text-2xl font-bold capitalize">{health.environment}</div>
              <div className="text-xs text-muted-foreground">Environment</div>
            </div>
            <div className="text-center p-3 rounded-lg bg-muted/50">
              <div className="text-2xl font-bold">{formatUptime(health.uptime_seconds)}</div>
              <div className="text-xs text-muted-foreground">Uptime</div>
            </div>
            <div className="text-center p-3 rounded-lg bg-muted/50">
              <div className="text-2xl font-bold">{Array.isArray(health.checks) ? health.checks.length : 0}</div>
              <div className="text-xs text-muted-foreground">Services</div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Service Health Checks */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Activity className="h-5 w-5" />
            Service Health Checks
          </CardTitle>
          <CardDescription>
            Individual service status and latency
          </CardDescription>
        </CardHeader>
        <CardContent>
          <ScrollArea className="h-[400px]">
            <div className="space-y-3">
              {(Array.isArray(health.checks) ? health.checks : []).map((check) => {
                const checkConfig = statusConfig[check.status];
                const CheckStatusIcon = checkConfig.icon;
                const ServiceIcon = serviceIcons[check.name] || Server;

                return (
                  <div
                    key={check.name}
                    className={`p-4 rounded-lg border ${checkConfig.bgColor} ${checkConfig.borderColor}`}
                  >
                    <div className="flex items-center justify-between mb-2">
                      <div className="flex items-center gap-3">
                        <ServiceIcon className="h-5 w-5 text-muted-foreground" />
                        <span className="font-medium capitalize">
                          {check.name.replace(/_/g, " ")}
                        </span>
                      </div>
                      <div className="flex items-center gap-2">
                        <Badge variant="outline" className={checkConfig.color}>
                          <CheckStatusIcon className="h-3 w-3 mr-1" />
                          {checkConfig.label}
                        </Badge>
                        <Badge variant="secondary">
                          <Clock className="h-3 w-3 mr-1" />
                          {check.latency_ms.toFixed(1)}ms
                        </Badge>
                      </div>
                    </div>
                    
                    {check.message && (
                      <p className="text-sm text-muted-foreground mb-2">
                        {check.message}
                      </p>
                    )}

                    {/* Latency Progress Bar */}
                    <div className="space-y-1">
                      <div className="flex justify-between text-xs text-muted-foreground">
                        <span>Latency</span>
                        <span>{check.latency_ms.toFixed(1)}ms</span>
                      </div>
                      <Progress 
                        value={Math.min(check.latency_ms / 10, 100)} 
                        className="h-1"
                      />
                    </div>

                    {/* Details */}
                    {Object.keys(check.details).length > 0 && (
                      <>
                        <Separator className="my-2" />
                        <div className="grid grid-cols-2 gap-2 text-xs">
                          {Object.entries(check.details).map(([key, value]) => (
                            <div key={key} className="flex justify-between">
                              <span className="text-muted-foreground capitalize">
                                {key.replace(/_/g, " ")}:
                              </span>
                              <span className="font-mono">
                                {typeof value === "object" 
                                  ? JSON.stringify(value).slice(0, 30) 
                                  : String(value).slice(0, 30)}
                              </span>
                            </div>
                          ))}
                        </div>
                      </>
                    )}
                  </div>
                );
              })}
            </div>
          </ScrollArea>
        </CardContent>
      </Card>

      {/* Last Refresh Info */}
      {lastRefresh && (
        <div className="text-center text-sm text-muted-foreground">
          Last updated: {lastRefresh.toLocaleTimeString()}
          {autoRefresh && (
            <span className="ml-2">
              (auto-refresh every {refreshInterval / 1000}s)
            </span>
          )}
        </div>
      )}
    </div>
  );
}

// Compact health indicator for use in headers/sidebars
export function HealthIndicator() {
  const [status, setStatus] = useState<"healthy" | "degraded" | "unhealthy" | "unknown">("unknown");
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    const checkHealth = async () => {
      try {
        const response = await apiClient.get<{ status: string }>("/health/live");
        setStatus(response.data.status as typeof status);
      } catch {
        setStatus("unhealthy");
      } finally {
        setIsLoading(false);
      }
    };

    checkHealth();
    const interval = setInterval(checkHealth, 60000); // Check every minute
    return () => clearInterval(interval);
  }, []);

  if (isLoading) {
    return <Loader2 className="h-4 w-4 animate-spin text-muted-foreground" />;
  }

  const config = statusConfig[status];
  const StatusIcon = config.icon;

  return (
    <div className="flex items-center gap-1" title={`System: ${config.label}`}>
      <StatusIcon className={`h-4 w-4 ${config.color}`} />
    </div>
  );
}