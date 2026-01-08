"use client";

import { useEffect, useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { apiClient } from "@/lib/api-enhanced";

interface ServiceStatus {
  name: string;
  status: "healthy" | "degraded" | "down";
  latency?: number;
  lastCheck: string;
}

interface SystemHealth {
  backend: ServiceStatus;
  redis: ServiceStatus;
  providers: {
    google: ServiceStatus;
    openai: ServiceStatus;
    anthropic: ServiceStatus;
    deepseek: ServiceStatus;
  };
  services: {
    llm: ServiceStatus;
    transformation: ServiceStatus;
    enhancement: ServiceStatus;
    jailbreak: ServiceStatus;
  };
  infrastructure: {
    circuitBreaker: ServiceStatus;
    rateLimit: ServiceStatus;
    websocket: ServiceStatus;
  };
}

export default function SystemArchitectureDashboard() {
  const [health, setHealth] = useState<SystemHealth | null>(null);
  const [loading, setLoading] = useState(true);
  const [autoRefresh, setAutoRefresh] = useState(true);

  const fetchHealth = async () => {
    try {
      // Root-level endpoint (no /v1/ prefix)
      const response = await apiClient.get("/health/full");
      setHealth(response.data);
    } catch (error) {
      console.error("Failed to fetch health:", error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchHealth();
    if (autoRefresh) {
      const interval = setInterval(fetchHealth, 5000);
      return () => clearInterval(interval);
    }
  }, [autoRefresh]);

  const getStatusColor = (status: string) => {
    switch (status) {
      case "healthy":
        return "bg-green-500";
      case "degraded":
        return "bg-yellow-500";
      case "down":
        return "bg-red-500";
      default:
        return "bg-gray-500";
    }
  };

  const ServiceCard = ({ service, name }: { service: ServiceStatus; name: string }) => (
    <div className="p-4 border rounded-lg bg-card">
      <div className="flex items-center justify-between mb-2">
        <h4 className="font-semibold">{name}</h4>
        <Badge className={getStatusColor(service.status)}>{service.status}</Badge>
      </div>
      {service.latency && (
        <p className="text-sm text-muted-foreground">Latency: {service.latency}ms</p>
      )}
      <p className="text-xs text-muted-foreground">Last check: {service.lastCheck}</p>
    </div>
  );

  if (loading) {
    return <div className="flex items-center justify-center h-screen">Loading system status...</div>;
  }

  return (
    <div className="container mx-auto p-6 space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-3xl font-bold">Chimera System Architecture</h1>
        <div className="flex gap-2">
          <Button onClick={fetchHealth} variant="outline">Refresh</Button>
          <Button
            onClick={() => setAutoRefresh(!autoRefresh)}
            variant={autoRefresh ? "default" : "outline"}
          >
            Auto-refresh: {autoRefresh ? "ON" : "OFF"}
          </Button>
        </div>
      </div>

      {/* Backend Services */}
      <Card>
        <CardHeader>
          <CardTitle>Backend Services (Port 8001)</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {health?.backend && <ServiceCard service={health.backend} name="FastAPI Server" />}
            {health?.services.llm && <ServiceCard service={health.services.llm} name="LLM Service" />}
            {health?.services.transformation && (
              <ServiceCard service={health.services.transformation} name="Transformation Engine" />
            )}
            {health?.services.enhancement && (
              <ServiceCard service={health.services.enhancement} name="Prompt Enhancer" />
            )}
            {health?.services.jailbreak && (
              <ServiceCard service={health.services.jailbreak} name="Jailbreak Enhancer" />
            )}
          </div>
        </CardContent>
      </Card>

      {/* LLM Providers */}
      <Card>
        <CardHeader>
          <CardTitle>LLM Providers</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            {health?.providers.google && (
              <ServiceCard service={health.providers.google} name="Google/Gemini" />
            )}
            {health?.providers.openai && (
              <ServiceCard service={health.providers.openai} name="OpenAI" />
            )}
            {health?.providers.anthropic && (
              <ServiceCard service={health.providers.anthropic} name="Anthropic/Claude" />
            )}
            {health?.providers.deepseek && (
              <ServiceCard service={health.providers.deepseek} name="DeepSeek" />
            )}
          </div>
        </CardContent>
      </Card>

      {/* Infrastructure */}
      <Card>
        <CardHeader>
          <CardTitle>Infrastructure & Middleware</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            {health?.redis && <ServiceCard service={health.redis} name="Redis Cache" />}
            {health?.infrastructure.circuitBreaker && (
              <ServiceCard service={health.infrastructure.circuitBreaker} name="Circuit Breaker" />
            )}
            {health?.infrastructure.rateLimit && (
              <ServiceCard service={health.infrastructure.rateLimit} name="Rate Limiter" />
            )}
            {health?.infrastructure.websocket && (
              <ServiceCard service={health.infrastructure.websocket} name="WebSocket" />
            )}
          </div>
        </CardContent>
      </Card>

      {/* Data Flow Visualization */}
      <Card>
        <CardHeader>
          <CardTitle>System Data Flow</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="relative p-8 bg-muted rounded-lg">
            <svg className="w-full h-96" viewBox="0 0 800 400">
              {/* Frontend Layer */}
              <rect x="50" y="50" width="150" height="80" fill="#3b82f6" rx="8" />
              <text x="125" y="95" textAnchor="middle" fill="white" fontSize="14" fontWeight="bold">
                Next.js Frontend
              </text>
              <text x="125" y="115" textAnchor="middle" fill="white" fontSize="12">
                Port 3000
              </text>

              {/* API Gateway */}
              <rect x="325" y="50" width="150" height="80" fill="#8b5cf6" rx="8" />
              <text x="400" y="95" textAnchor="middle" fill="white" fontSize="14" fontWeight="bold">
                FastAPI Backend
              </text>
              <text x="400" y="115" textAnchor="middle" fill="white" fontSize="12">
                Port 8001
              </text>

              {/* LLM Service */}
              <rect x="325" y="180" width="150" height="60" fill="#10b981" rx="8" />
              <text x="400" y="215" textAnchor="middle" fill="white" fontSize="14" fontWeight="bold">
                LLM Service
              </text>

              {/* Transformation Service */}
              <rect x="325" y="260" width="150" height="60" fill="#f59e0b" rx="8" />
              <text x="400" y="295" textAnchor="middle" fill="white" fontSize="14" fontWeight="bold">
                Transformation
              </text>

              {/* Providers */}
              <rect x="600" y="50" width="150" height="50" fill="#ec4899" rx="8" />
              <text x="675" y="80" textAnchor="middle" fill="white" fontSize="12">
                Google/Gemini
              </text>

              <rect x="600" y="120" width="150" height="50" fill="#ec4899" rx="8" />
              <text x="675" y="150" textAnchor="middle" fill="white" fontSize="12">
                OpenAI
              </text>

              <rect x="600" y="190" width="150" height="50" fill="#ec4899" rx="8" />
              <text x="675" y="220" textAnchor="middle" fill="white" fontSize="12">
                Anthropic/Claude
              </text>

              <rect x="600" y="260" width="150" height="50" fill="#ec4899" rx="8" />
              <text x="675" y="290" textAnchor="middle" fill="white" fontSize="12">
                DeepSeek
              </text>

              {/* Redis */}
              <rect x="50" y="260" width="150" height="60" fill="#ef4444" rx="8" />
              <text x="125" y="295" textAnchor="middle" fill="white" fontSize="14" fontWeight="bold">
                Redis Cache
              </text>

              {/* Arrows */}
              <defs>
                <marker id="arrowhead" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto">
                  <polygon points="0 0, 10 3, 0 6" fill="#64748b" />
                </marker>
              </defs>

              {/* Frontend to Backend */}
              <line x1="200" y1="90" x2="325" y2="90" stroke="#64748b" strokeWidth="2" markerEnd="url(#arrowhead)" />

              {/* Backend to LLM Service */}
              <line x1="400" y1="130" x2="400" y2="180" stroke="#64748b" strokeWidth="2" markerEnd="url(#arrowhead)" />

              {/* Backend to Transformation */}
              <line x1="400" y1="130" x2="400" y2="260" stroke="#64748b" strokeWidth="2" markerEnd="url(#arrowhead)" />

              {/* LLM to Providers */}
              <line x1="475" y1="210" x2="600" y2="75" stroke="#64748b" strokeWidth="2" markerEnd="url(#arrowhead)" />
              <line x1="475" y1="210" x2="600" y2="145" stroke="#64748b" strokeWidth="2" markerEnd="url(#arrowhead)" />
              <line x1="475" y1="210" x2="600" y2="215" stroke="#64748b" strokeWidth="2" markerEnd="url(#arrowhead)" />
              <line x1="475" y1="210" x2="600" y2="285" stroke="#64748b" strokeWidth="2" markerEnd="url(#arrowhead)" />

              {/* Backend to Redis */}
              <line x1="325" y1="110" x2="200" y2="280" stroke="#64748b" strokeWidth="2" markerEnd="url(#arrowhead)" />
            </svg>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}