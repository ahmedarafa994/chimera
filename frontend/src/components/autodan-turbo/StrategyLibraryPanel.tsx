"use client";

import React, { useState, useEffect, useCallback } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog";
import { enhancedApi } from "@/lib/api-enhanced";
import { toast } from "sonner";
import {
  Loader2,
  Plus,
  Search,
  Trash2,
  RefreshCw,
  Download,
  Upload,
  Database,
  BookOpen,
  Copy,
  Check,
  ChevronDown,
  ChevronUp,
  Save,
  AlertTriangle,
} from "lucide-react";
import type {
  JailbreakStrategy,
  LibraryStatsResponse,
  StrategySearchResult,
} from "@/lib/types/autodan-turbo-types";

interface StrategyCardProps {
  strategy: JailbreakStrategy;
  onDelete: (id: string) => void;
  isDeleting: boolean;
}

function StrategyCard({ strategy, onDelete, isDeleting }: StrategyCardProps) {
  const [expanded, setExpanded] = useState(false);
  const [copied, setCopied] = useState(false);

  const handleCopy = async () => {
    const examplesText = strategy.examples.length > 0 ? strategy.examples[0] : "No example available";
    const text = `Strategy: ${strategy.name}\nDescription: ${strategy.description}\nExample: ${examplesText}`;
    await navigator.clipboard.writeText(text);
    setCopied(true);
    toast.success("Strategy copied to clipboard");
    setTimeout(() => setCopied(false), 2000);
  };

  const getSourceBadgeVariant = (source: string) => {
    switch (source) {
      case "seed":
        return "default";
      case "warmup":
        return "secondary";
      case "lifelong":
        return "outline";
      default:
        return "default";
    }
  };

  return (
    <Card className="mb-3">
      <CardHeader className="pb-2">
        <div className="flex items-start justify-between">
          <div className="flex-1">
            <CardTitle className="text-base flex items-center gap-2">
              <BookOpen className="h-4 w-4" />
              {strategy.name}
            </CardTitle>
            <div className="flex items-center gap-2 mt-1">
              <Badge variant={getSourceBadgeVariant(strategy.source)}>
                {strategy.source}
              </Badge>
              {strategy.average_score > 0 && (
                <Badge variant="outline" className="text-green-600">
                  Avg: {strategy.average_score.toFixed(1)}
                </Badge>
              )}
              {strategy.usage_count > 0 && (
                <Badge variant="outline">
                  Used {strategy.usage_count}x
                </Badge>
              )}
            </div>
          </div>
          <Button
            variant="ghost"
            size="sm"
            onClick={() => setExpanded(!expanded)}
          >
            {expanded ? (
              <ChevronUp className="h-4 w-4" />
            ) : (
              <ChevronDown className="h-4 w-4" />
            )}
          </Button>
        </div>
      </CardHeader>
      <CardContent className="pb-2">
        <p className="text-sm text-muted-foreground line-clamp-2">
          {strategy.description}
        </p>
        {expanded && (
          <div className="mt-4 space-y-3">
            <div>
              <Label className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
                Full Description
              </Label>
              <p className="text-sm mt-1">{strategy.description}</p>
            </div>
            {strategy.examples && strategy.examples.length > 0 && (
              <div>
                <Label className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
                  Example Prompts
                </Label>
                {strategy.examples.map((example, idx) => (
                  <pre key={idx} className="text-xs mt-1 whitespace-pre-wrap bg-muted p-2 rounded font-mono max-h-[200px] overflow-auto">
                    {example}
                  </pre>
                ))}
              </div>
            )}
            <div className="text-xs text-muted-foreground">
              ID: {strategy.id}
              {strategy.created_at && (
                <> • Created: {new Date(strategy.created_at).toLocaleString()}</>
              )}
            </div>
          </div>
        )}
      </CardContent>
      <CardFooter className="pt-2 gap-2">
        <Button variant="ghost" size="sm" onClick={handleCopy}>
          {copied ? (
            <Check className="h-3 w-3 mr-1" />
          ) : (
            <Copy className="h-3 w-3 mr-1" />
          )}
          Copy
        </Button>
        <Button
          variant="ghost"
          size="sm"
          className="text-destructive hover:text-destructive"
          onClick={() => onDelete(strategy.id)}
          disabled={isDeleting}
        >
          {isDeleting ? (
            <Loader2 className="h-3 w-3 mr-1 animate-spin" />
          ) : (
            <Trash2 className="h-3 w-3 mr-1" />
          )}
          Delete
        </Button>
      </CardFooter>
    </Card>
  );
}

export function StrategyLibraryPanel() {
  const [strategies, setStrategies] = useState<JailbreakStrategy[]>([]);
  const [stats, setStats] = useState<LibraryStatsResponse | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [isSearching, setIsSearching] = useState(false);
  const [deletingId, setDeletingId] = useState<string | null>(null);
  const [searchQuery, setSearchQuery] = useState("");
  const [searchResults, setSearchResults] = useState<StrategySearchResult[] | null>(null);
  
  // Create strategy dialog state
  const [createDialogOpen, setCreateDialogOpen] = useState(false);
  const [newStrategy, setNewStrategy] = useState({
    name: "",
    description: "",
    template: "",
    example: "",
  });
  const [isCreating, setIsCreating] = useState(false);

  // Import/Export state
  const [isExporting, setIsExporting] = useState(false);
  const [isSaving, setIsSaving] = useState(false);
  const [isClearing, setIsClearing] = useState(false);
  const [clearDialogOpen, setClearDialogOpen] = useState(false);

  const loadStrategies = useCallback(async () => {
    setIsLoading(true);
    try {
      const [strategiesRes, statsRes] = await Promise.all([
        enhancedApi.autodanTurbo.strategies.list(0, 100),
        enhancedApi.autodanTurbo.library.stats(),
      ]);
      setStrategies(strategiesRes.data.strategies);
      setStats(statsRes.data);
    } catch (error) {
      console.error("Failed to load strategies:", error);
      toast.error("Failed to load strategy library");
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    loadStrategies();
  }, [loadStrategies]);

  const handleSearch = async () => {
    if (!searchQuery.trim()) {
      setSearchResults(null);
      return;
    }

    setIsSearching(true);
    try {
      const response = await enhancedApi.autodanTurbo.strategies.search({
        query: searchQuery,
        top_k: 10,
      });
      setSearchResults(response.data.results);
      toast.success(`Found ${response.data.results.length} matching strategies`);
    } catch (error) {
      console.error("Search failed:", error);
      toast.error("Search failed");
    } finally {
      setIsSearching(false);
    }
  };

  const handleDelete = async (strategyId: string) => {
    setDeletingId(strategyId);
    try {
      await enhancedApi.autodanTurbo.strategies.delete(strategyId);
      setStrategies(prev => prev.filter(s => s.id !== strategyId));
      toast.success("Strategy deleted");
      // Refresh stats
      const statsRes = await enhancedApi.autodanTurbo.library.stats();
      setStats(statsRes.data);
    } catch (error) {
      console.error("Delete failed:", error);
      toast.error("Failed to delete strategy");
    } finally {
      setDeletingId(null);
    }
  };

  const handleCreate = async () => {
    if (!newStrategy.name.trim() || !newStrategy.description.trim()) {
      toast.error("Name and description are required");
      return;
    }

    setIsCreating(true);
    try {
      const response = await enhancedApi.autodanTurbo.strategies.create({
        name: newStrategy.name,
        description: newStrategy.description,
        template: newStrategy.template || newStrategy.description,
        examples: newStrategy.example ? [newStrategy.example] : undefined,
      });
      setStrategies(prev => [response.data, ...prev]);
      setNewStrategy({ name: "", description: "", template: "", example: "" });
      setCreateDialogOpen(false);
      toast.success("Strategy created successfully");
      // Refresh stats
      const statsRes = await enhancedApi.autodanTurbo.library.stats();
      setStats(statsRes.data);
    } catch (error) {
      console.error("Create failed:", error);
      toast.error("Failed to create strategy");
    } finally {
      setIsCreating(false);
    }
  };

  const handleExport = async () => {
    setIsExporting(true);
    try {
      const response = await enhancedApi.autodanTurbo.library.export();
      const blob = new Blob([JSON.stringify(response.data, null, 2)], {
        type: "application/json",
      });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `strategy-library-${new Date().toISOString().split("T")[0]}.json`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
      toast.success("Library exported successfully");
    } catch (error) {
      console.error("Export failed:", error);
      toast.error("Failed to export library");
    } finally {
      setIsExporting(false);
    }
  };

  const handleSave = async () => {
    setIsSaving(true);
    try {
      await enhancedApi.autodanTurbo.library.save();
      toast.success("Library saved to disk");
    } catch (error) {
      console.error("Save failed:", error);
      toast.error("Failed to save library");
    } finally {
      setIsSaving(false);
    }
  };

  const handleClear = async () => {
    setIsClearing(true);
    try {
      await enhancedApi.autodanTurbo.library.clear();
      setStrategies([]);
      setStats(null);
      setClearDialogOpen(false);
      toast.success("Library cleared");
    } catch (error) {
      console.error("Clear failed:", error);
      toast.error("Failed to clear library");
    } finally {
      setIsClearing(false);
    }
  };

  const displayStrategies = searchResults
    ? searchResults.map(r => r.strategy)
    : strategies;

  return (
    <div className="space-y-6">
      {/* Stats Card */}
      {stats && (
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-lg flex items-center gap-2">
              <Database className="h-5 w-5" />
              Library Statistics
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div className="text-center">
                <div className="text-2xl font-bold">{stats.total_strategies}</div>
                <div className="text-xs text-muted-foreground">Total Strategies</div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold">{stats.discovered_count ?? 0}</div>
                <div className="text-xs text-muted-foreground">Discovered</div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold">{stats.human_designed_count ?? 0}</div>
                <div className="text-xs text-muted-foreground">Human Designed</div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold">{stats.average_success_rate?.toFixed(1) ?? "0.0"}%</div>
                <div className="text-xs text-muted-foreground">Avg Success Rate</div>
              </div>
            </div>
            {stats.avg_score_differential !== undefined && (
              <div className="mt-4 text-center">
                <Badge variant="outline" className="text-green-600">
                  Avg Score Improvement: +{stats.avg_score_differential.toFixed(2)}
                </Badge>
              </div>
            )}
          </CardContent>
        </Card>
      )}

      {/* Actions Bar */}
      <div className="flex flex-wrap gap-2">
        <Dialog open={createDialogOpen} onOpenChange={setCreateDialogOpen}>
          <DialogTrigger asChild>
            <Button>
              <Plus className="h-4 w-4 mr-2" />
              Add Strategy
            </Button>
          </DialogTrigger>
          <DialogContent className="max-w-2xl">
            <DialogHeader>
              <DialogTitle>Create New Strategy</DialogTitle>
              <DialogDescription>
                Add a human-designed jailbreak strategy to the library.
              </DialogDescription>
            </DialogHeader>
            <div className="space-y-4 py-4">
              <div className="space-y-2">
                <Label htmlFor="strategy-name">Strategy Name</Label>
                <Input
                  id="strategy-name"
                  value={newStrategy.name}
                  onChange={(e) => setNewStrategy(prev => ({ ...prev, name: e.target.value }))}
                  placeholder="e.g., Role-Playing Scenario"
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="strategy-description">Description</Label>
                <Textarea
                  id="strategy-description"
                  value={newStrategy.description}
                  onChange={(e) => setNewStrategy(prev => ({ ...prev, description: e.target.value }))}
                  placeholder="A concise description of the strategy..."
                  rows={3}
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="strategy-template">Template (Optional)</Label>
                <Textarea
                  id="strategy-template"
                  value={newStrategy.template}
                  onChange={(e) => setNewStrategy(prev => ({ ...prev, template: e.target.value }))}
                  placeholder="A template for generating prompts using this strategy..."
                  rows={3}
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="strategy-example">Example Prompt (Optional)</Label>
                <Textarea
                  id="strategy-example"
                  value={newStrategy.example}
                  onChange={(e) => setNewStrategy(prev => ({ ...prev, example: e.target.value }))}
                  placeholder="An example jailbreak prompt using this strategy..."
                  rows={5}
                />
              </div>
            </div>
            <DialogFooter>
              <Button variant="outline" onClick={() => setCreateDialogOpen(false)}>
                Cancel
              </Button>
              <Button onClick={handleCreate} disabled={isCreating}>
                {isCreating ? (
                  <>
                    <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                    Creating...
                  </>
                ) : (
                  <>
                    <Plus className="h-4 w-4 mr-2" />
                    Create Strategy
                  </>
                )}
              </Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>

        <Button variant="outline" onClick={loadStrategies} disabled={isLoading}>
          <RefreshCw className={`h-4 w-4 mr-2 ${isLoading ? "animate-spin" : ""}`} />
          Refresh
        </Button>

        <Button variant="outline" onClick={handleSave} disabled={isSaving}>
          {isSaving ? (
            <Loader2 className="h-4 w-4 mr-2 animate-spin" />
          ) : (
            <Save className="h-4 w-4 mr-2" />
          )}
          Save to Disk
        </Button>

        <Button variant="outline" onClick={handleExport} disabled={isExporting}>
          {isExporting ? (
            <Loader2 className="h-4 w-4 mr-2 animate-spin" />
          ) : (
            <Download className="h-4 w-4 mr-2" />
          )}
          Export
        </Button>

        <Dialog open={clearDialogOpen} onOpenChange={setClearDialogOpen}>
          <DialogTrigger asChild>
            <Button variant="outline" className="text-destructive hover:text-destructive">
              <Trash2 className="h-4 w-4 mr-2" />
              Clear All
            </Button>
          </DialogTrigger>
          <DialogContent>
            <DialogHeader>
              <DialogTitle className="flex items-center gap-2">
                <AlertTriangle className="h-5 w-5 text-destructive" />
                Clear Strategy Library
              </DialogTitle>
              <DialogDescription>
                This will permanently delete all strategies from the library. This action cannot be undone.
              </DialogDescription>
            </DialogHeader>
            <DialogFooter>
              <Button variant="outline" onClick={() => setClearDialogOpen(false)}>
                Cancel
              </Button>
              <Button variant="destructive" onClick={handleClear} disabled={isClearing}>
                {isClearing ? (
                  <>
                    <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                    Clearing...
                  </>
                ) : (
                  <>
                    <Trash2 className="h-4 w-4 mr-2" />
                    Clear All Strategies
                  </>
                )}
              </Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>
      </div>

      {/* Search Bar */}
      <div className="flex gap-2">
        <div className="flex-1">
          <Input
            placeholder="Search strategies by response similarity..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && handleSearch()}
          />
        </div>
        <Button onClick={handleSearch} disabled={isSearching}>
          {isSearching ? (
            <Loader2 className="h-4 w-4 animate-spin" />
          ) : (
            <Search className="h-4 w-4" />
          )}
        </Button>
        {searchResults && (
          <Button variant="outline" onClick={() => {
            setSearchResults(null);
            setSearchQuery("");
          }}>
            Clear Search
          </Button>
        )}
      </div>

      {/* Strategies List */}
      <Card>
        <CardHeader>
          <CardTitle className="text-lg">
            {searchResults ? `Search Results (${searchResults.length})` : `Strategies (${strategies.length})`}
          </CardTitle>
          <CardDescription>
            {searchResults
              ? "Strategies matching your search query, ranked by similarity"
              : "All strategies in the library, sorted by creation date"}
          </CardDescription>
        </CardHeader>
        <CardContent>
          {isLoading ? (
            <div className="flex items-center justify-center py-12">
              <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
            </div>
          ) : displayStrategies.length === 0 ? (
            <div className="flex flex-col items-center justify-center py-12 text-muted-foreground">
              <Database className="h-12 w-12 mb-4 opacity-20" />
              <p>No strategies found</p>
              <p className="text-sm">Add strategies manually or run the lifelong learning process</p>
            </div>
          ) : (
            <ScrollArea className="h-[500px] pr-4">
              {displayStrategies.map((strategy) => (
                <StrategyCard
                  key={strategy.id}
                  strategy={strategy}
                  onDelete={handleDelete}
                  isDeleting={deletingId === strategy.id}
                />
              ))}
            </ScrollArea>
          )}
        </CardContent>
      </Card>
    </div>
  );
}