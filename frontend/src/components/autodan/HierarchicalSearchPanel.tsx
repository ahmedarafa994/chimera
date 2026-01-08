"use client";

import React, { useState } from "react";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { Input } from "@/components/ui/input";
import { Slider } from "@/components/ui/slider";
import { enhancedApi, HierarchicalSearchRequest, HierarchicalSearchResponse } from "@/lib/api-enhanced";
import { toast } from "sonner";
import { Loader2, Network, TrendingUp } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";

export function HierarchicalSearchPanel() {
  const [request, setRequest] = useState("");
  const [populationSize, setPopulationSize] = useState(20);
  const [generations, setGenerations] = useState(10);
  const [mutationRate, setMutationRate] = useState(0.3);
  const [crossoverRate, setCrossoverRate] = useState(0.7);
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState<HierarchicalSearchResponse | null>(null);

  const handleSearch = async () => {
    if (!request.trim()) {
      toast.error("Please enter a request");
      return;
    }

    setIsLoading(true);
    setResult(null);

    try {
      const payload: HierarchicalSearchRequest = {
        request: request.trim(),
        population_size: populationSize,
        generations,
        mutation_rate: mutationRate,
        crossover_rate: crossoverRate,
      };

      const response = await enhancedApi.hierarchicalSearch(payload);
      setResult(response as any);

      toast.success("Hierarchical search completed", {
        description: `Best score: ${(response as any).best_score?.toFixed(2) || 'N/A'}`,
      });
    } catch (error: unknown) {
      console.error("Hierarchical search failed:", error);
      const errorMessage = error instanceof Error ? error.message : "Unknown error occurred";
      toast.error("Search failed", {
        description: errorMessage,
      });
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="grid gap-6 md:grid-cols-2">
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Network className="h-5 w-5" />
            Hierarchical Genetic Search
          </CardTitle>
          <CardDescription>
            Bi-level evolution with meta-strategy optimization
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="space-y-2">
            <Label htmlFor="hgs-request">Target Request</Label>
            <Textarea
              id="hgs-request"
              value={request}
              onChange={(e) => setRequest(e.target.value)}
              placeholder="Enter the harmful request to optimize..."
              rows={4}
            />
          </div>

          <div className="space-y-2">
            <Label>Population Size: {populationSize}</Label>
            <Slider
              value={[populationSize]}
              onValueChange={(v) => setPopulationSize(v[0])}
              min={10}
              max={100}
              step={10}
            />
          </div>

          <div className="space-y-2">
            <Label>Generations: {generations}</Label>
            <Slider
              value={[generations]}
              onValueChange={(v) => setGenerations(v[0])}
              min={5}
              max={50}
              step={5}
            />
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div className="space-y-2">
              <Label>Mutation Rate: {mutationRate.toFixed(2)}</Label>
              <Slider
                value={[mutationRate * 100]}
                onValueChange={(v) => setMutationRate(v[0] / 100)}
                min={0}
                max={100}
                step={5}
              />
            </div>
            <div className="space-y-2">
              <Label>Crossover Rate: {crossoverRate.toFixed(2)}</Label>
              <Slider
                value={[crossoverRate * 100]}
                onValueChange={(v) => setCrossoverRate(v[0] / 100)}
                min={0}
                max={100}
                step={5}
              />
            </div>
          </div>
        </CardContent>
        <CardFooter>
          <Button
            onClick={handleSearch}
            disabled={isLoading || !request.trim()}
            className="w-full"
          >
            {isLoading ? (
              <>
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                Evolving...
              </>
            ) : (
              <>
                <Network className="mr-2 h-4 w-4" />
                Start Evolution
              </>
            )}
          </Button>
        </CardFooter>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <TrendingUp className="h-5 w-5" />
            Evolution Results
          </CardTitle>
          <CardDescription>
            Best prompt and generation metrics
          </CardDescription>
        </CardHeader>
        <CardContent>
          {result ? (
            <div className="space-y-4">
              <div className="flex items-center gap-2">
                <Badge variant="default">Score: {result.best_score?.toFixed(2) ?? "N/A"}</Badge>
                <Badge variant="secondary">
                  {result.execution_time_ms ?? 0}ms
                </Badge>
              </div>

              <div className="space-y-2">
                <Label>Best Prompt</Label>
                <ScrollArea className="h-[200px] rounded-md border p-3">
                  <pre className="text-sm whitespace-pre-wrap">
                    {result.best_prompt}
                  </pre>
                </ScrollArea>
              </div>

              <div className="space-y-2">
                <Label>Generation History</Label>
                <ScrollArea className="h-[150px]">
                  <div className="space-y-2">
                    {(result.generation_history ?? []).map((gen, idx) => (
                      <div key={idx} className="flex items-center justify-between text-sm border-b pb-2">
                        <span>Gen {gen.generation}</span>
                        <div className="flex gap-2">
                          <Badge variant="outline">Best: {gen.best_score?.toFixed(2) ?? "N/A"}</Badge>
                          <Badge variant="outline">Avg: {gen.avg_score?.toFixed(2) ?? "N/A"}</Badge>
                        </div>
                      </div>
                    ))}
                  </div>
                </ScrollArea>
              </div>
            </div>
          ) : (
            <div className="flex flex-col items-center justify-center h-[400px] text-muted-foreground">
              <Network className="h-12 w-12 mb-4 opacity-20" />
              <p>Configure and start evolution to see results</p>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
