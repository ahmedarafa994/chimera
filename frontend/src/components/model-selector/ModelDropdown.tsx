"use client";

/**
 * Model Dropdown Component
 *
 * Compact dropdown for header/toolbar use showing current selection
 * with quick switch capability.
 */

import React, { useState, useRef, useEffect } from "react";
import { cn } from "@/lib/utils";
import { useModelSelection } from "@/lib/stores/model-selection-store";
import { ProviderStatusBadge } from "@/components/ui/provider-status-badge";

export interface ModelDropdownProps {
  className?: string;
  onSelectionChange?: (provider: string, model: string) => void;
}

export function ModelDropdown({ className, onSelectionChange }: ModelDropdownProps) {
  const {
    providers,
    selectedProvider,
    selectedModel,
    isLoading,
    wsConnected,
    selectProvider,
    selectModel,
    getSelectedProviderInfo,
  } = useModelSelection();

  const [isOpen, setIsOpen] = useState(false);
  const dropdownRef = useRef<HTMLDivElement>(null);

  // Close dropdown when clicking outside
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
        setIsOpen(false);
      }
    };

    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  // Close on escape key
  useEffect(() => {
    const handleEscape = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        setIsOpen(false);
      }
    };

    document.addEventListener("keydown", handleEscape);
    return () => document.removeEventListener("keydown", handleEscape);
  }, []);

  const selectedProviderInfo = getSelectedProviderInfo();

  const providerIcons: Record<string, string> = {
    // Gemini AI
    gemini: "✨",
    google: "✨",
    "gemini-cli": "✨",
    "gemini-openai": "✨",
    // Hybrid
    antigravity: "⚡",
    // Other providers
    deepseek: "🔍",
    openai: "🤖",
    anthropic: "🧠",
    kiro: "🧠",
    qwen: "💻",
    cursor: "🖱️",
  };

  const handleProviderModelSelect = async (provider: string, model: string) => {
    if (provider !== selectedProvider) {
      await selectProvider(provider);
    }
    const success = await selectModel(model);
    if (success) {
      setIsOpen(false);
      onSelectionChange?.(provider, model);
    }
  };

  const getProviderStatus = (isHealthy: boolean, status: string): "healthy" | "degraded" | "unavailable" | "unknown" => {
    if (!isHealthy) return "unavailable";
    if (status === "active") return "healthy";
    if (status === "degraded") return "degraded";
    return "unknown";
  };

  return (
    <div ref={dropdownRef} className={cn("relative", className)}>
      {/* Trigger Button */}
      <button
        onClick={() => setIsOpen(!isOpen)}
        disabled={isLoading}
        className={cn(
          "flex items-center gap-2 px-3 py-2 rounded-lg border transition-all",
          "hover:bg-accent focus:outline-none focus:ring-2 focus:ring-primary/50",
          "bg-background border-border",
          isLoading && "opacity-50 cursor-not-allowed"
        )}
      >
        {/* Provider icon */}
        <span className="text-lg">
          {providerIcons[selectedProvider?.toLowerCase() || ""] || "🔮"}
        </span>

        {/* Selection text */}
        <div className="flex flex-col items-start">
          <span className="text-xs text-muted-foreground">
            {selectedProviderInfo?.displayName || "Select Model"}
          </span>
          <span className="text-sm font-medium truncate max-w-[120px]">
            {selectedModel || "None"}
          </span>
        </div>

        {/* Status indicator */}
        {selectedProviderInfo && (
          <ProviderStatusBadge
            status={getProviderStatus(selectedProviderInfo.isHealthy, selectedProviderInfo.status)}
            size="sm"
          />
        )}

        {/* Dropdown arrow */}
        <svg
          className={cn(
            "w-4 h-4 text-muted-foreground transition-transform",
            isOpen && "rotate-180"
          )}
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
        >
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
        </svg>

        {/* Live sync indicator */}
        {wsConnected && (
          <span className="absolute -top-1 -right-1 flex h-2 w-2">
            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-green-400 opacity-75"></span>
            <span className="relative inline-flex rounded-full h-2 w-2 bg-green-500"></span>
          </span>
        )}
      </button>

      {/* Dropdown Menu */}
      {isOpen && (
        <div className="absolute top-full left-0 mt-1 w-72 bg-popover border border-border rounded-lg shadow-lg z-50 overflow-hidden">
          {/* Header */}
          <div className="px-3 py-2 border-b border-border bg-muted/50">
            <span className="text-xs font-medium text-muted-foreground">
              Select Provider & Model
            </span>
          </div>

          {/* Provider/Model list */}
          <div className="max-h-[400px] overflow-y-auto">
            {providers.map((provider) => (
              <div key={provider.provider} className="border-b border-border last:border-0">
                {/* Provider header */}
                <div className="flex items-center gap-2 px-3 py-2 bg-muted/30">
                  <span className="text-lg">
                    {providerIcons[provider.provider.toLowerCase()] || "🔮"}
                  </span>
                  <span className="font-medium text-sm">{provider.displayName}</span>
                  <ProviderStatusBadge
                    status={getProviderStatus(provider.isHealthy, provider.status)}
                    size="sm"
                  />
                  {provider.latencyMs && (
                    <span className="text-xs text-muted-foreground ml-auto">
                      {Math.round(provider.latencyMs)}ms
                    </span>
                  )}
                </div>

                {/* Models */}
                <div className="py-1">
                  {provider.models.slice(0, 5).map((model) => (
                    <button
                      key={model}
                      onClick={() => handleProviderModelSelect(provider.provider, model)}
                      disabled={!provider.isHealthy}
                      className={cn(
                        "w-full flex items-center justify-between px-4 py-2 text-sm",
                        "hover:bg-accent transition-colors",
                        "disabled:opacity-50 disabled:cursor-not-allowed",
                        selectedProvider === provider.provider && selectedModel === model
                          ? "bg-primary/10 text-primary"
                          : "text-foreground"
                      )}
                    >
                      <span className="truncate">{model}</span>
                      {selectedProvider === provider.provider && selectedModel === model && (
                        <svg
                          className="w-4 h-4 text-primary flex-shrink-0"
                          fill="currentColor"
                          viewBox="0 0 20 20"
                        >
                          <path
                            fillRule="evenodd"
                            d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z"
                            clipRule="evenodd"
                          />
                        </svg>
                      )}
                    </button>
                  ))}
                  {provider.models.length > 5 && (
                    <div className="px-4 py-1 text-xs text-muted-foreground">
                      +{provider.models.length - 5} more models
                    </div>
                  )}
                </div>
              </div>
            ))}
          </div>

          {/* Footer */}
          <div className="px-3 py-2 border-t border-border bg-muted/30 flex items-center justify-between">
            <span className="text-xs text-muted-foreground">
              {providers.length} provider{providers.length !== 1 ? "s" : ""} available
            </span>
            {isLoading && (
              <span className="text-xs text-muted-foreground">Updating...</span>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

export default ModelDropdown;
