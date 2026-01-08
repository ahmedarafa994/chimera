"use client";

import { useQuery } from "@tanstack/react-query";
import enhancedApi from "@/lib/api-enhanced";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import {
  Activity,
  Server,
  Database,
  Cpu,
  Clock,
  CheckCircle2,
  XCircle,
  AlertCircle,
  RefreshCw,
  Zap,
  Shield,
  Box
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";

export function MetricsDashboard() {
  const { data: metricsData, isLoading: metricsLoading, error: metricsError, refetch: refetchMetrics } = useQuery({
    queryKey: ["metrics"],
    queryFn: () => enhancedApi.metrics(),
    refetchInterval: 30000, // Refresh every 30 seconds
  });

  const { data: healthData, isLoading: healthLoading, error: healthError, refetch: refetchHealth } = useQuery({
    queryKey: ["health"],
    queryFn: () => enhancedApi.health(),
    refetchInterval: 30000,
  });

  const { data: providersData, isLoading: providersLoading } = useQuery({
    queryKey: ["providers"],
    queryFn: () => enhancedApi.providers.list(),
    refetchInterval: 60000,
  });

  const { data: techniquesData, isLoading: techniquesLoading } = useQuery({
    queryKey: ["techniques"],
    queryFn: () => enhancedApi.techniques(),
  });

  const isLoading = metricsLoading || healthLoading || providersLoading || techniquesLoading;

  const handleRefresh = () => {
    refetchMetrics();
    refetchHealth();
  };

  const metrics = metricsData?.metrics;
  const health = healthData;
  const providers = providersData?.data?.providers || [];
  const techniques = techniquesData?.techniques;

  const activeProviders = providers.filter((p) => p.status === "active").length;
  const totalProviders = providers.length;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold tracking-tight">System Metrics</h2>
          <p className="text-muted-foreground">Real-time monitoring and analytics</p>
        </div>
        <Button variant="outline" onClick={handleRefresh} disabled={isLoading}>
          <RefreshCw className={`mr-2 h-4 w-4 ${isLoading ? "animate-spin" : ""}`} />
          Refresh
        </Button>
      </div>

      {/* Status Cards */}
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
        {/* System Status */}
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">System Status</CardTitle>
            <Activity className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="flex items-center gap-2">
              {health?.status === "healthy" ? (
                <>
                  <CheckCircle2 className="h-5 w-5 text-green-500" />
                  <span className="text-2xl font-bold text-green-500">Healthy</span>
                </>
              ) : healthError ? (
                <>
                  <XCircle className="h-5 w-5 text-red-500" />
                  <span className="text-2xl font-bold text-red-500">Offline</span>
                </>
              ) : (
                <>
                  <AlertCircle className="h-5 w-5 text-yellow-500" />
                  <span className="text-2xl font-bold text-yellow-500">Unknown</span>
                </>
              )}
            </div>
            <p className="text-xs text-muted-foreground mt-1">
              Last check: {health?.timestamp ? new Date(health.timestamp).toLocaleTimeString() : "N/A"}
            </p>
          </CardContent>
        </Card>

        {/* Active Providers */}
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Active Providers</CardTitle>
            <Server className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {activeProviders} / {totalProviders}
            </div>
            <Progress value={(activeProviders / totalProviders) * 100} className="mt-2" />
            <p className="text-xs text-muted-foreground mt-1">
              LLM integrations online
            </p>
          </CardContent>
        </Card>

        {/* Cache Status */}
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Cache Status</CardTitle>
            <Database className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="flex items-center gap-2">
              {metrics?.cache?.enabled ? (
                <Badge variant="default" className="bg-green-500">Enabled</Badge>
              ) : (
                <Badge variant="secondary">Disabled</Badge>
              )}
            </div>
            <p className="text-2xl font-bold mt-1">
              {metrics?.cache?.entries || 0}
            </p>
            <p className="text-xs text-muted-foreground">
              Cached entries
            </p>
          </CardContent>
        </Card>

        {/* Techniques Available */}
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Techniques</CardTitle>
            <Zap className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {techniques?.length || 0}
            </div>
            <p className="text-xs text-muted-foreground mt-1">
              Transformation suites available
            </p>
          </CardContent>
        </Card>
      </div>

      {/* Provider Status Grid */}
      <Card>
        <CardHeader>
          <CardTitle>Provider Status</CardTitle>
          <CardDescription>Real-time status of all connected LLM providers</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
            {providers.map((provider, index) => (
              <div key={`provider-${provider.provider}-${index}`} className="flex items-center justify-between p-4 rounded-lg border bg-card">
                <div className="flex items-center gap-3">
                  <div className={`p-2 rounded-full ${provider.status === "active"
                    ? "bg-green-500/10 text-green-500"
                    : "bg-gray-500/10 text-gray-500"
                    }`}>
                    <Box className="h-4 w-4" />
                  </div>
                  <div>
                    <p className="font-medium capitalize">{provider.provider}</p>
                    <p className="text-xs text-muted-foreground truncate max-w-[120px]">
                      {provider.model}
                    </p>
                  </div>
                </div>
                <Badge variant={provider.status === "active" ? "default" : "secondary"}>
                  {provider.status}
                </Badge>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Detailed Metrics */}
      <div className="grid gap-4 md:grid-cols-2">
        {/* Service Metrics */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Cpu className="h-5 w-5" />
              Service Metrics
            </CardTitle>
            <CardDescription>Current service status and configuration</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="flex items-center justify-between py-2 border-b">
              <span className="text-sm text-muted-foreground">Status</span>
              <Badge variant={metrics?.status === "operational" ? "default" : "secondary"}>
                {metrics?.status || "Unknown"}
              </Badge>
            </div>
            <div className="flex items-center justify-between py-2 border-b">
              <span className="text-sm text-muted-foreground">Cache Enabled</span>
              <span className="font-medium">{metrics?.cache?.enabled ? "Yes" : "No"}</span>
            </div>
            <div className="flex items-center justify-between py-2 border-b">
              <span className="text-sm text-muted-foreground">Cache Entries</span>
              <span className="font-medium">{metrics?.cache?.entries || 0}</span>
            </div>
            <div className="flex items-center justify-between py-2">
              <span className="text-sm text-muted-foreground">Last Updated</span>
              <span className="font-medium text-xs">
                {metricsData?.timestamp
                  ? new Date(metricsData.timestamp).toLocaleString()
                  : "N/A"}
              </span>
            </div>
          </CardContent>
        </Card>

        {/* Provider Metrics */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Shield className="h-5 w-5" />
              Provider Metrics
            </CardTitle>
            <CardDescription>Provider-specific status information</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            {metrics?.providers ? (
              Object.entries(metrics.providers).map(([key, value]) => (
                <div key={key} className="flex items-center justify-between py-2 border-b last:border-0">
                  <span className="text-sm text-muted-foreground capitalize">{key}</span>
                  <Badge variant={value === "active" || value === "healthy" ? "default" : "secondary"}>
                    {String(value)}
                  </Badge>
                </div>
              ))
            ) : (
              <div className="text-center text-muted-foreground py-4">
                No provider metrics available
              </div>
            )}
          </CardContent>
        </Card>
      </div>

      {/* Quick Stats */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Clock className="h-5 w-5" />
            System Overview
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid gap-4 md:grid-cols-3">
            <div className="text-center p-4 rounded-lg bg-muted/50">
              <p className="text-3xl font-bold text-blue-500">{totalProviders}</p>
              <p className="text-sm text-muted-foreground">Total Providers</p>
            </div>
            <div className="text-center p-4 rounded-lg bg-muted/50">
              <p className="text-3xl font-bold text-green-500">{activeProviders}</p>
              <p className="text-sm text-muted-foreground">Active Providers</p>
            </div>
            <div className="text-center p-4 rounded-lg bg-muted/50">
              <p className="text-3xl font-bold text-purple-500">{techniques?.length || 0}</p>
              <p className="text-sm text-muted-foreground">Technique Suites</p>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}