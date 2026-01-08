"use client";

import React, { useState, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Checkbox } from "@/components/ui/checkbox";
import { Input } from "@/components/ui/input";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import { enhancedApi } from "@/lib/api-enhanced";
import { toast } from "sonner";
import {
  Loader2,
  Library,
  ChevronDown,
  ChevronUp,
  Search,
  Star,
  TrendingUp,
  Sparkles,
  RefreshCw,
} from "lucide-react";
import type { JailbreakStrategy, LibraryStatsResponse } from "@/lib/types/autodan-turbo-types";

interface StrategySelectorProps {
  selectedStrategies: string[];
  onSelectionChange: (strategies: string[]) => void;
  maxSelections?: number;
  showStats?: boolean;
  compact?: boolean;
}

export function StrategySelector({
  selectedStrategies,
  onSelectionChange,
  maxSelections = 3,
  showStats = true,
  compact = false,
}: StrategySelectorProps) {
  const [strategies, setStrategies] = useState<JailbreakStrategy[]>([]);
  const [libraryStats, setLibraryStats] = useState<LibraryStatsResponse | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [isExpanded, setIsExpanded] = useState(!compact);
  const [searchQuery, setSearchQuery] = useState("");
  const [filterByTopPerformers, setFilterByTopPerformers] = useState(false);

  useEffect(() => {
    loadStrategies();
    if (showStats) {
      loadLibraryStats();
    }
  }, [showStats]);

  const loadStrategies = async () => {
    setIsLoading(true);
    try {
      const response = await enhancedApi.autodanTurbo.strategies.list();
      setStrategies(response.data.strategies || []);
    } catch (error) {
      console.error("Failed to load strategies:", error);
      toast.error("Failed to load strategies");
    } finally {
      setIsLoading(false);
    }
  };

  const loadLibraryStats = async () => {
    try {
      const response = await enhancedApi.autodanTurbo.library.stats();
      setLibraryStats(response.data);
    } catch (error) {
      console.error("Failed to load library stats:", error);
    }
  };

  const handleStrategyToggle = (strategyId: string) => {
    if (selectedStrategies.includes(strategyId)) {
      onSelectionChange(selectedStrategies.filter(id => id !== strategyId));
    } else {
      if (selectedStrategies.length >= maxSelections) {
        toast.warning(`Maximum ${maxSelections} strategies can be selected`);
        return;
      }
      onSelectionChange([...selectedStrategies, strategyId]);
    }
  };

  const handleSelectTopPerformers = () => {
    if (!libraryStats?.top_strategies_by_success_rate) return;
    
    const topIds = libraryStats.top_strategies_by_success_rate
      .slice(0, maxSelections)
      .map(s => s.id);
    onSelectionChange(topIds);
    toast.success(`Selected top ${topIds.length} performing strategies`);
  };

  const handleClearSelection = () => {
    onSelectionChange([]);
  };

  // Filter strategies based on search and filters
  const filteredStrategies = strategies.filter(strategy => {
    const matchesSearch = !searchQuery || 
      strategy.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
      strategy.description?.toLowerCase().includes(searchQuery.toLowerCase()) ||
      strategy.tags?.some(tag => tag.toLowerCase().includes(searchQuery.toLowerCase()));
    
    const matchesTopFilter = !filterByTopPerformers || 
      (strategy.success_rate !== undefined && strategy.success_rate >= 0.5);
    
    return matchesSearch && matchesTopFilter;
  });

  // Sort by success rate
  const sortedStrategies = [...filteredStrategies].sort((a, b) => {
    const rateA = a.success_rate ?? 0;
    const rateB = b.success_rate ?? 0;
    return rateB - rateA;
  });

  if (compact) {
    return (
      <Collapsible open={isExpanded} onOpenChange={setIsExpanded}>
        <div className="rounded-lg border p-3 space-y-2">
          <CollapsibleTrigger asChild>
            <div className="flex items-center justify-between cursor-pointer">
              <div className="flex items-center gap-2">
                <Library className="h-4 w-4 text-muted-foreground" />
                <span className="text-sm font-medium">Strategy Selection</span>
                {selectedStrategies.length > 0 && (
                  <Badge variant="secondary" className="text-xs">
                    {selectedStrategies.length} selected
                  </Badge>
                )}
              </div>
              {isExpanded ? (
                <ChevronUp className="h-4 w-4 text-muted-foreground" />
              ) : (
                <ChevronDown className="h-4 w-4 text-muted-foreground" />
              )}
            </div>
          </CollapsibleTrigger>
          <CollapsibleContent className="space-y-3 pt-2">
            {renderContent()}
          </CollapsibleContent>
        </div>
      </Collapsible>
    );
  }

  return (
    <Card>
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <div>
            <CardTitle className="text-base flex items-center gap-2">
              <Library className="h-4 w-4" />
              Strategy Selection
            </CardTitle>
            <CardDescription className="text-xs">
              Select strategies to enhance jailbreak generation
            </CardDescription>
          </div>
          <Button
            variant="ghost"
            size="sm"
            onClick={loadStrategies}
            disabled={isLoading}
          >
            <RefreshCw className={`h-3 w-3 ${isLoading ? "animate-spin" : ""}`} />
          </Button>
        </div>
      </CardHeader>
      <CardContent className="space-y-3">
        {renderContent()}
      </CardContent>
    </Card>
  );

  function renderContent() {
    return (
      <>
        {/* Stats Summary */}
        {showStats && libraryStats && (
          <div className="flex items-center gap-2 text-xs text-muted-foreground pb-2 border-b">
            <span>{libraryStats.total_strategies} strategies</span>
            <span>•</span>
            <span>Avg score: {libraryStats.average_score.toFixed(1)}</span>
            <span>•</span>
            <span>Success: {(libraryStats.average_success_rate * 100).toFixed(0)}%</span>
          </div>
        )}

        {/* Search and Filters */}
        <div className="space-y-2">
          <div className="relative">
            <Search className="absolute left-2 top-2.5 h-3 w-3 text-muted-foreground" />
            <Input
              placeholder="Search strategies..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="pl-7 h-8 text-xs"
            />
          </div>
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Switch
                id="top-performers"
                checked={filterByTopPerformers}
                onCheckedChange={setFilterByTopPerformers}
                className="scale-75"
              />
              <Label htmlFor="top-performers" className="text-xs cursor-pointer">
                Top performers only
              </Label>
            </div>
            <div className="flex items-center gap-1">
              <Button
                variant="ghost"
                size="sm"
                onClick={handleSelectTopPerformers}
                className="h-6 text-xs px-2"
                disabled={!libraryStats?.top_strategies_by_success_rate?.length}
              >
                <Star className="h-3 w-3 mr-1" />
                Auto-select best
              </Button>
              {selectedStrategies.length > 0 && (
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={handleClearSelection}
                  className="h-6 text-xs px-2"
                >
                  Clear
                </Button>
              )}
            </div>
          </div>
        </div>

        {/* Strategy List */}
        {isLoading ? (
          <div className="flex items-center justify-center py-8">
            <Loader2 className="h-5 w-5 animate-spin text-muted-foreground" />
          </div>
        ) : sortedStrategies.length === 0 ? (
          <div className="text-center py-6 text-muted-foreground text-sm">
            {searchQuery ? "No strategies match your search" : "No strategies available"}
          </div>
        ) : (
          <ScrollArea className={compact ? "h-[200px]" : "h-[250px]"}>
            <div className="space-y-2 pr-3">
              {sortedStrategies.map((strategy) => (
                <StrategyItem
                  key={strategy.id}
                  strategy={strategy}
                  isSelected={selectedStrategies.includes(strategy.id)}
                  onToggle={() => handleStrategyToggle(strategy.id)}
                  disabled={!selectedStrategies.includes(strategy.id) && selectedStrategies.length >= maxSelections}
                />
              ))}
            </div>
          </ScrollArea>
        )}

        {/* Selection Summary */}
        {selectedStrategies.length > 0 && (
          <div className="pt-2 border-t">
            <div className="flex items-center gap-1 flex-wrap">
              <span className="text-xs text-muted-foreground">Selected:</span>
              {selectedStrategies.map(id => {
                const strategy = strategies.find(s => s.id === id);
                return (
                  <Badge
                    key={id}
                    variant="secondary"
                    className="text-xs cursor-pointer hover:bg-destructive/20"
                    onClick={() => handleStrategyToggle(id)}
                  >
                    {strategy?.name || id}
                    <span className="ml-1 opacity-60">×</span>
                  </Badge>
                );
              })}
            </div>
          </div>
        )}
      </>
    );
  }
}

interface StrategyItemProps {
  strategy: JailbreakStrategy;
  isSelected: boolean;
  onToggle: () => void;
  disabled: boolean;
}

function StrategyItem({ strategy, isSelected, onToggle, disabled }: StrategyItemProps) {
  const successRate = strategy.success_rate ?? 0;
  const avgScore = strategy.average_score ?? 0;

  return (
    <div
      className={`
        flex items-start gap-2 p-2 rounded-md border cursor-pointer transition-colors
        ${isSelected ? "border-primary bg-primary/5" : "border-transparent hover:bg-muted/50"}
        ${disabled && !isSelected ? "opacity-50 cursor-not-allowed" : ""}
      `}
      onClick={() => !disabled && onToggle()}
    >
      <Checkbox
        checked={isSelected}
        disabled={disabled && !isSelected}
        className="mt-0.5"
      />
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2">
          <span className="text-sm font-medium truncate">{strategy.name}</span>
          {successRate >= 0.7 && (
            <Sparkles className="h-3 w-3 text-yellow-500 flex-shrink-0" />
          )}
        </div>
        {strategy.description && (
          <p className="text-xs text-muted-foreground line-clamp-1 mt-0.5">
            {strategy.description}
          </p>
        )}
        <div className="flex items-center gap-2 mt-1">
          {strategy.tags?.slice(0, 2).map(tag => (
            <Badge key={tag} variant="outline" className="text-[10px] px-1 py-0">
              {tag}
            </Badge>
          ))}
          <div className="flex items-center gap-1 text-[10px] text-muted-foreground ml-auto">
            <TrendingUp className="h-2.5 w-2.5" />
            <span>{(successRate * 100).toFixed(0)}%</span>
            <span>•</span>
            <span>{avgScore.toFixed(1)}</span>
          </div>
        </div>
      </div>
    </div>
  );
}

export default StrategySelector;