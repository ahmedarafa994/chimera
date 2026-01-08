"use client";

import { useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import enhancedApi, {
  ConnectionStatusResponse,
  ConnectionTestResponse,
} from "@/lib/api-enhanced";
import { saveApiConfig, getApiConfig } from "@/lib/api-config";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Label } from "@/components/ui/label";
import { toast } from "sonner";
import {
  Wifi,
  WifiOff,
  Cloud,
  RefreshCw,
  CheckCircle2,
  Server,
  XCircle,
  Clock,
  Zap,
  AlertCircle,
} from "lucide-react";
import { ScrollArea } from "@/components/ui/scroll-area";

// Direct mode only - proxy mode removed
type ConnectionMode = "direct";

interface ConnectionStatusCardProps {
  title: string;
  status: ConnectionStatusResponse | null;
  isActive: boolean;
  icon: React.ReactNode;
  onSelect: () => void;
  isLoading?: boolean;
}

function ConnectionStatusCard({
  title,
  status,
  isActive,
  icon,
  onSelect,
  isLoading,
}: ConnectionStatusCardProps) {
  return (
    <Card
      className={`cursor-pointer transition-all ${
        isActive
          ? "border-primary ring-2 ring-primary/20"
          : "hover:border-muted-foreground/50"
      }`}
      onClick={onSelect}
    >
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            {icon}
            <CardTitle className="text-lg">{title}</CardTitle>
          </div>
          {isActive && (
            <Badge variant="default" className="bg-primary">
              Active
            </Badge>
          )}
        </div>
      </CardHeader>
      <CardContent>
        {isLoading ? (
          <div className="flex items-center gap-2 text-muted-foreground">
            <RefreshCw className="h-4 w-4 animate-spin" />
            <span>Checking connection...</span>
          </div>
        ) : status ? (
          <div className="space-y-3">
            <div className="flex items-center gap-2">
              {status.is_connected ? (
                <CheckCircle2 className="h-4 w-4 text-green-500" />
              ) : (
                <XCircle className="h-4 w-4 text-red-500" />
              )}
              <span
                className={
                  status.is_connected ? "text-green-600" : "text-red-600"
                }
              >
                {status.is_connected ? "Connected" : "Disconnected"}
              </span>
            </div>

            <div className="text-sm text-muted-foreground">
              <div className="flex items-center gap-1">
                <Server className="h-3 w-3" />
                <span className="truncate">{status.base_url}</span>
              </div>
            </div>

            {status.latency_ms && (
              <div className="flex items-center gap-1 text-sm">
                <Clock className="h-3 w-3 text-muted-foreground" />
                <span>{status.latency_ms.toFixed(0)}ms latency</span>
              </div>
            )}

            {status.error_message && (
              <div className="flex items-start gap-1 text-sm text-red-500">
                <AlertCircle className="h-3 w-3 mt-0.5 flex-shrink-0" />
                <span className="break-words">{status.error_message}</span>
              </div>
            )}

            {status.available_models && status.available_models.length > 0 && (
              <div className="mt-2">
                <p className="text-xs text-muted-foreground mb-1">
                  Available Models ({status.available_models.length})
                </p>
                <ScrollArea className="h-[60px]">
                  <div className="flex flex-wrap gap-1">
                    {status.available_models.slice(0, 10).map((model) => (
                      <Badge
                        key={model}
                        variant="outline"
                        className="text-[10px]"
                      >
                        {model}
                      </Badge>
                    ))}
                    {status.available_models.length > 10 && (
                      <Badge variant="secondary" className="text-[10px]">
                        +{status.available_models.length - 10} more
                      </Badge>
                    )}
                  </div>
                </ScrollArea>
              </div>
            )}
          </div>
        ) : (
          <div className="text-sm text-muted-foreground">
            Click to test connection
          </div>
        )}
      </CardContent>
    </Card>
  );
}

export function ConnectionConfig() {
  const queryClient = useQueryClient();
  const [testResults, setTestResults] = useState<ConnectionTestResponse | null>(
    null
  );
  const [currentProvider, setCurrentProvider] = useState<"gemini" | "deepseek">(
    () => getApiConfig().aiProvider
  );

  // Fetch current connection config
  const {
    data: configData,
    isLoading: configLoading,
    error: configError,
  } = useQuery({
    queryKey: ["connection-config"],
    queryFn: () => enhancedApi.connection.getConfig(),
    retry: 1,
  });

  // Fetch current connection status
  const {
    data: statusData,
    isLoading: statusLoading,
    refetch: refetchStatus,
  } = useQuery({
    queryKey: ["connection-status"],
    queryFn: () => enhancedApi.connection.getStatus(),
    retry: 1,
    refetchInterval: 30000, // Refresh every 30 seconds
  });

  // Mutation to change connection mode
  const setModeMutation = useMutation({
    mutationFn: (mode: ConnectionMode) =>
      enhancedApi.connection.setMode({ mode }),
    onSuccess: (response) => {
      toast.success("Connection Mode Changed", {
        description: response.data.message,
      });
      queryClient.invalidateQueries({ queryKey: ["connection-config"] });
      queryClient.invalidateQueries({ queryKey: ["connection-status"] });
    },
    onError: (error: any) => {
      toast.error("Failed to change mode", {
        description: error.response?.data?.detail || error.message,
      });
    },
  });

  // Mutation to test both connections
  const testMutation = useMutation({
    mutationFn: () => enhancedApi.connection.test(),
    onSuccess: (response) => {
      setTestResults(response.data);
      toast.success("Connection Test Complete", {
        description: `Recommended: ${response.data.recommended} mode`,
      });
    },
    onError: (error: any) => {
      toast.error("Connection test failed", {
        description: error.response?.data?.detail || error.message,
      });
    },
  });

  const currentMode = configData?.data?.current_mode || "direct";
  const currentStatus = statusData?.data;

  const handleModeChange = (mode: ConnectionMode, provider?: "gemini" | "deepseek") => {
    if (provider) {
      saveApiConfig({ aiProvider: provider });
      setCurrentProvider(provider);
      // Update local API client immediately
      enhancedApi.utils.updateConfig(mode);
    }

    if (mode !== currentMode) {
      setModeMutation.mutate(mode);
    } else if (provider && provider !== currentProvider) {
      // Refresh status if provider changed but mode stayed same
      queryClient.invalidateQueries({ queryKey: ["connection-config"] });
      queryClient.invalidateQueries({ queryKey: ["connection-status"] });
    }
  };

  const handleTestConnections = () => {
    testMutation.mutate();
  };

  const handleRefreshStatus = () => {
    refetchStatus();
  };

  // Show loading state
  if (configLoading) {
    return (
      <div className="p-8 text-center text-muted-foreground">
        <RefreshCw className="h-8 w-8 animate-spin mx-auto mb-2" />
        Loading connection configuration...
      </div>
    );
  }

  // Show error state
  if (configError) {
    return (
      <Card className="border-destructive">
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-destructive">
            <AlertCircle className="h-5 w-5" />
            Connection Error
          </CardTitle>
          <CardDescription>
            Unable to fetch connection configuration from the backend server.
            Make sure the Chimera backend is running.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <Button onClick={() => queryClient.invalidateQueries()}>
            <RefreshCw className="h-4 w-4 mr-2" />
            Retry
          </Button>
        </CardContent>
      </Card>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header with status indicator */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold">API Connection</h2>
          <p className="text-muted-foreground">
            Configure how the application connects to AI services
          </p>
        </div>
        <div className="flex items-center gap-2">
          {currentStatus?.is_connected ? (
            <Badge variant="default" className="bg-green-500">
              <Wifi className="h-3 w-3 mr-1" />
              Connected
            </Badge>
          ) : (
            <Badge variant="destructive">
              <WifiOff className="h-3 w-3 mr-1" />
              Disconnected
            </Badge>
          )}
        </div>
      </div>

      {/* Connection Mode Cards - Direct Only */}
      <div className="grid gap-4 md:grid-cols-2">
        <ConnectionStatusCard
          title="Direct Gemini API"
          status={
            testResults?.direct ||
            (currentMode === "direct" && currentProvider === "gemini" ? currentStatus || null : null)
          }
          isActive={currentMode === "direct" && currentProvider === "gemini"}
          icon={<Cloud className="h-5 w-5 text-purple-500" />}
          onSelect={() => handleModeChange("direct", "gemini")}
          isLoading={setModeMutation.isPending && setModeMutation.variables === "direct" && currentProvider === "gemini"}
        />

        <ConnectionStatusCard
          title="Direct DeepSeek API"
          status={
            testResults?.direct ||
            (currentMode === "direct" && currentProvider === "deepseek" ? currentStatus || null : null)
          }
          isActive={currentMode === "direct" && currentProvider === "deepseek"}
          icon={<Zap className="h-5 w-5 text-indigo-500" />}
          onSelect={() => handleModeChange("direct", "deepseek")}
          isLoading={setModeMutation.isPending && setModeMutation.variables === "direct" && currentProvider === "deepseek"}
        />
      </div>

      {/* Test Results Recommendation */}
      {testResults && (
        <Card className="border-primary/50 bg-primary/5">
          <CardHeader className="pb-2">
            <CardTitle className="text-lg flex items-center gap-2">
              <Zap className="h-5 w-5 text-yellow-500" />
              Connection Status
            </CardTitle>
          </CardHeader>
          <CardContent>
            <p>
              {testResults.direct?.is_connected
                ? (
                    <span>
                      Direct API connection is working with{" "}
                      <strong className="text-primary">
                        {testResults.direct?.latency_ms?.toFixed(0) || "N/A"}ms
                      </strong>{" "}
                      latency.
                    </span>
                  )
                : "Direct API connection failed. Please check your API key configuration."}
            </p>
          </CardContent>
        </Card>
      )}

      {/* Action Buttons */}
      <div className="flex gap-3">
        <Button
          variant="outline"
          onClick={handleTestConnections}
          disabled={testMutation.isPending}
        >
          {testMutation.isPending ? (
            <RefreshCw className="h-4 w-4 mr-2 animate-spin" />
          ) : (
            <Zap className="h-4 w-4 mr-2" />
          )}
          Test Connection
        </Button>

        <Button
          variant="ghost"
          onClick={handleRefreshStatus}
          disabled={statusLoading}
        >
          <RefreshCw
            className={`h-4 w-4 mr-2 ${statusLoading ? "animate-spin" : ""}`}
          />
          Refresh Status
        </Button>
      </div>

      {/* Configuration Details */}
      <Card>
        <CardHeader>
          <CardTitle>Configuration Details</CardTitle>
          <CardDescription>
            Current API connection settings
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <div>
              <Label className="text-muted-foreground">
                {currentProvider === "deepseek" ? "DeepSeek API" : "Gemini API"}
              </Label>
              <p className="font-mono text-sm">
                {currentProvider === "deepseek"
                  ? (getApiConfig().deepseekApiUrl || "https://api.deepseek.com/v1")
                  : (configData?.data?.direct?.url || "https://generativelanguage.googleapis.com")}
              </p>
              <div className="flex items-center gap-2 mt-1">
                <Badge
                  variant={
                    (currentProvider === "deepseek"
                      ? !!getApiConfig().deepseekApiKey
                      : configData?.data?.direct?.api_key_configured)
                      ? "default"
                      : "secondary"
                  }
                >
                  {(currentProvider === "deepseek"
                    ? !!getApiConfig().deepseekApiKey
                    : configData?.data?.direct?.api_key_configured)
                    ? "API Key Set"
                    : "No API Key"}
                </Badge>
                {configData?.data?.direct?.api_key_preview && currentProvider !== "deepseek" && (
                  <span className="text-xs text-muted-foreground font-mono">
                    {configData?.data?.direct?.api_key_preview}
                  </span>
                )}
              </div>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}

export default ConnectionConfig;
