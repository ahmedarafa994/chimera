"use client";

import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Switch } from "@/components/ui/switch";
import { apiClient } from "@/lib/api-enhanced";
import { Badge } from "@/components/ui/badge";

interface ServiceControl {
  name: string;
  enabled: boolean;
  config: Record<string, unknown>;
}

export default function InteractiveControlPanel() {
  const [circuitBreaker, setCircuitBreaker] = useState<ServiceControl>({
    name: "Circuit Breaker",
    enabled: true,
    config: { threshold: 3, timeout: 60000 }
  });

  const [rateLimit, setRateLimit] = useState<ServiceControl>({
    name: "Rate Limiter",
    enabled: true,
    config: { maxRequests: 100, windowMs: 60000 }
  });

  const [selectedProvider, setSelectedProvider] = useState("google");
  const [testPrompt, setTestPrompt] = useState("");
  const [testResult, setTestResult] = useState<{ error?: string; text?: string; usage?: { total_tokens?: number }; latency?: number } | null>(null);
  const [testing, setTesting] = useState(false);

  const handleTestProvider = async () => {
    setTesting(true);
    try {
      // API v1 endpoint (with /v1/ prefix)
      const response = await apiClient.post("/generate", {
        prompt: testPrompt,
        provider: selectedProvider,
        config: {
          temperature: 0.7,
          max_tokens: 100
        }
      });
      setTestResult(response.data);
    } catch (error: unknown) {
      const errorMessage = error instanceof Error ? error.message : "Unknown error";
      setTestResult({ error: errorMessage });
    } finally {
      setTesting(false);
    }
  };

  const handleToggleService = async (service: string, enabled: boolean) => {
    try {
      // API v1 endpoint (with /v1/ prefix)
      await apiClient.post(`/services/${service}/toggle`, { enabled });
      if (service === "circuit-breaker") {
        setCircuitBreaker({ ...circuitBreaker, enabled });
      } else if (service === "rate-limit") {
        setRateLimit({ ...rateLimit, enabled });
      }
    } catch (error) {
      console.error(`Failed to toggle ${service}:`, error);
    }
  };

  const handleClearCache = async () => {
    try {
      // API v1 endpoint (with /v1/ prefix)
      await apiClient.post("/cache/clear");
      alert("Cache cleared successfully");
    } catch (error) {
      console.error("Failed to clear cache:", error);
    }
  };

  return (
    <div className="container mx-auto p-6 space-y-6">
      <h1 className="text-3xl font-bold">System Control Panel</h1>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <Card>
          <CardHeader>
            <CardTitle>Service Controls</CardTitle>
          </CardHeader>
          <CardContent className="space-y-6">
            <div className="flex items-center justify-between">
              <div>
                <Label htmlFor="circuit-breaker">Circuit Breaker</Label>
                <p className="text-sm text-muted-foreground">
                  Threshold: {circuitBreaker.config.threshold as number} failures
                </p>
              </div>
              <Switch
                id="circuit-breaker"
                checked={circuitBreaker.enabled}
                onCheckedChange={(checked) => handleToggleService("circuit-breaker", checked)}
              />
            </div>

            <div className="flex items-center justify-between">
              <div>
                <Label htmlFor="rate-limit">Rate Limiter</Label>
                <p className="text-sm text-muted-foreground">
                  Max: {rateLimit.config.maxRequests as number} req/min
                </p>
              </div>
              <Switch
                id="rate-limit"
                checked={rateLimit.enabled}
                onCheckedChange={(checked) => handleToggleService("rate-limit", checked)}
              />
            </div>

            <div className="pt-4 border-t">
              <Button onClick={handleClearCache} variant="destructive" className="w-full">
                Clear Redis Cache
              </Button>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Provider Testing</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div>
              <Label htmlFor="provider">Select Provider</Label>
              <Select value={selectedProvider} onValueChange={setSelectedProvider}>
                <SelectTrigger id="provider">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="google">Google/Gemini</SelectItem>
                  <SelectItem value="openai">OpenAI</SelectItem>
                  <SelectItem value="anthropic">Anthropic/Claude</SelectItem>
                  <SelectItem value="deepseek">DeepSeek</SelectItem>
                </SelectContent>
              </Select>
            </div>

            <div>
              <Label htmlFor="test-prompt">Test Prompt</Label>
              <Input
                id="test-prompt"
                placeholder="Enter a test prompt..."
                value={testPrompt}
                onChange={(e) => setTestPrompt(e.target.value)}
              />
            </div>

            <Button
              onClick={handleTestProvider}
              disabled={testing || !testPrompt}
              className="w-full"
            >
              {testing ? "Testing..." : "Test Provider"}
            </Button>

            {testResult && (
              <div className="mt-4 p-4 border rounded-lg bg-muted">
                <h4 className="font-semibold mb-2">Result:</h4>
                {testResult.error ? (
                  <Badge variant="destructive">{testResult.error}</Badge>
                ) : (
                  <div className="space-y-2">
                    <p className="text-sm">{testResult.text}</p>
                    <div className="flex gap-2 text-xs text-muted-foreground">
                      <span>Tokens: {testResult.usage?.total_tokens}</span>
                      <span>Latency: {testResult.latency}ms</span>
                    </div>
                  </div>
                )}
              </div>
            )}
          </CardContent>
        </Card>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Transformation Techniques</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-5 gap-2">
            {[
              "simple", "advanced", "expert", "cognitive_hacking", "hypothetical_scenario",
              "advanced_obfuscation", "typoglycemia", "hierarchical_persona", "dan_persona",
              "contextual_inception", "nested_context", "logical_inference", "conditional_logic",
              "multimodal_jailbreak", "visual_context", "agentic_exploitation", "multi_agent",
              "payload_splitting", "instruction_fragmentation", "quantum_exploit", "deep_inception",
              "code_chameleon", "cipher"
            ].map((technique) => (
              <Badge key={technique} variant="outline" className="justify-center">
                {technique}
              </Badge>
            ))}
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>WebSocket Connection</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <span>Real-time Enhancement Stream</span>
              <Badge variant="outline">ws://localhost:8001/ws/enhance</Badge>
            </div>
            <Button variant="outline" className="w-full">
              Test WebSocket Connection
            </Button>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}