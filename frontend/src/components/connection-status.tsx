"use client";

import { useEffect, useState } from "react";
import { Badge } from "@/components/ui/badge";
import { Wifi, WifiOff, RefreshCw, Settings, Info, ExternalLink, Zap } from "lucide-react";
import { Button } from "@/components/ui/button";
import enhancedApi from "@/lib/api-enhanced";
import { API_MODE_LABELS, ApiMode, saveApiConfig, getApiConfig } from "@/lib/api-config";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";

export function ConnectionStatus() {
  const [isConnected, setIsConnected] = useState<boolean | null>(null);
  const [isChecking, setIsChecking] = useState(false);
  const [currentMode, setCurrentMode] = useState<ApiMode>("direct");
  const [currentProvider, setCurrentProvider] = useState<"gemini" | "deepseek">("gemini");
  const [currentUrl, setCurrentUrl] = useState<string>("");
  const [lastChecked, setLastChecked] = useState<Date | null>(null);
  const [responseTime, setResponseTime] = useState<number | null>(null);

  const checkConnection = async () => {
    setIsChecking(true);
    const startTime = Date.now();
    try {
      const connected = await enhancedApi.utils.checkConnection();
      setResponseTime(Date.now() - startTime);
      setIsConnected(connected);
      setCurrentMode(enhancedApi.utils.getCurrentMode());
      setCurrentProvider(getApiConfig().aiProvider);
      setCurrentUrl(enhancedApi.utils.getCurrentUrl());
      setLastChecked(new Date());
    } catch {
      setIsConnected(false);
      setResponseTime(null);
    } finally {
      setIsChecking(false);
    }
  };

  const switchMode = (mode: ApiMode, provider?: "gemini" | "deepseek") => {
    if (provider) {
      saveApiConfig({ aiProvider: provider });
      setCurrentProvider(provider);
    }
    enhancedApi.utils.updateConfig(mode);
    setCurrentMode(mode);
    setCurrentUrl(enhancedApi.utils.getCurrentUrl());
    checkConnection();
  };

  useEffect(() => {
    setCurrentMode(enhancedApi.utils.getCurrentMode());
    setCurrentProvider(getApiConfig().aiProvider);
    setCurrentUrl(enhancedApi.utils.getCurrentUrl());
    checkConnection();
    // Check connection every 30 seconds
    const interval = setInterval(checkConnection, 30000);
    return () => clearInterval(interval);
  }, []);

  // Status color based on connection and response time
  const getStatusColor = () => {
    if (!isConnected) return "bg-red-600";
    if (responseTime && responseTime > 1000) return "bg-yellow-600";
    return "bg-green-600";
  };

  if (isConnected === null) {
    return (
      <Badge variant="outline" className="gap-1">
        <RefreshCw className="h-3 w-3 animate-spin" />
        Checking...
      </Badge>
    );
  }

  return (
    <div className="flex items-center gap-2">
      {/* Connection Status Badge */}
      {isConnected ? (
        <Badge variant="default" className={`gap-1 ${getStatusColor()}`}>
          <Wifi className="h-3 w-3" />
          {currentProvider === "deepseek" ? "Direct DeepSeek" : "Direct Gemini"}
        </Badge>
      ) : (
        <Badge variant="secondary" className="gap-1 bg-red-600 text-white">
          <WifiOff className="h-3 w-3" />
          Disconnected
        </Badge>
      )}

      {/* Connection Details Popover */}
      <Popover>
        <PopoverTrigger asChild>
          <Button variant="ghost" size="icon" className="h-6 w-6">
            <Info className="h-3 w-3" />
          </Button>
        </PopoverTrigger>
        <PopoverContent className="w-72" align="end">
          <div className="space-y-3">
            <h4 className="font-medium text-sm">Connection Details</h4>

            <div className="space-y-2 text-sm">
              <div className="flex justify-between">
                <span className="text-muted-foreground">Status:</span>
                <span className={isConnected ? "text-green-600" : "text-red-600"}>
                  {isConnected ? "Connected" : "Disconnected"}
                </span>
              </div>

              <div className="flex justify-between">
                <span className="text-muted-foreground">Mode:</span>
                <span>
                  {currentProvider === "deepseek"
                    ? "Direct DeepSeek"
                    : "Direct Gemini"}
                </span>
              </div>

              <div className="flex justify-between">
                <span className="text-muted-foreground">Endpoint:</span>
                <span className="text-xs font-mono truncate max-w-[150px]" title={currentUrl}>
                  {currentUrl}
                </span>
              </div>

              {responseTime !== null && (
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Response Time:</span>
                  <span className={responseTime > 1000 ? "text-yellow-600" : "text-green-600"}>
                    {responseTime}ms
                  </span>
                </div>
              )}

              {lastChecked && (
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Last Checked:</span>
                  <span className="text-xs">
                    {lastChecked.toLocaleTimeString()}
                  </span>
                </div>
              )}
            </div>

            {!isConnected && (
              <div className="pt-2 border-t">
                <p className="text-xs text-muted-foreground mb-2">
                  Backend server may not be running.
                </p>
                <a
                  href="http://localhost:8001/docs"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-xs text-primary hover:underline inline-flex items-center gap-1"
                >
                  <ExternalLink className="h-3 w-3" />
                  Open API Docs
                </a>
              </div>
            )}
          </div>
        </PopoverContent>
      </Popover>

      {/* Settings Dropdown */}
      <DropdownMenu>
        <DropdownMenuTrigger asChild>
          <Button
            variant="ghost"
            size="icon"
            className="h-6 w-6"
          >
            <Settings className="h-3 w-3" />
          </Button>
        </DropdownMenuTrigger>
        <DropdownMenuContent align="end" className="w-64">
          <DropdownMenuLabel>API Connection Mode</DropdownMenuLabel>
          <DropdownMenuSeparator />
          <DropdownMenuItem
            onClick={() => switchMode("direct", "gemini")}
            className={currentProvider === "gemini" ? "bg-accent" : ""}
          >
            <div className="flex flex-col">
              <span className="font-medium">Direct Gemini API</span>
              <span className="text-xs text-muted-foreground">Google AI</span>
            </div>
          </DropdownMenuItem>
          <DropdownMenuItem
            onClick={() => switchMode("direct", "deepseek")}
            className={currentProvider === "deepseek" ? "bg-accent" : ""}
          >
            <div className="flex flex-col">
              <span className="font-medium">Direct DeepSeek API</span>
              <span className="text-xs text-muted-foreground">DeepSeek AI</span>
            </div>
          </DropdownMenuItem>
          <DropdownMenuSeparator />
          <div className="px-2 py-1.5 text-xs text-muted-foreground">
            Current: {currentUrl}
          </div>
        </DropdownMenuContent>
      </DropdownMenu>

      {/* Refresh Button */}
      <Button
        variant="ghost"
        size="icon"
        className="h-6 w-6"
        onClick={checkConnection}
        disabled={isChecking}
        title="Refresh connection"
      >
        <RefreshCw className={`h-3 w-3 ${isChecking ? "animate-spin" : ""}`} />
      </Button>
    </div>
  );
}
