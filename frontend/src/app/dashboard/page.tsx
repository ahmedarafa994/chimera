"use client";

import Link from "next/link";
import { useQuery } from "@tanstack/react-query";
import enhancedApi from "@/lib/api-enhanced";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import {
  Zap,
  Sparkles,
  Server,
  Layers,
  Skull,
  Activity,
  ArrowRight,
  CheckCircle2,
  XCircle,
  Target,
  Brain
} from "lucide-react";

export default function DashboardPage() {
  const { data: healthData, isLoading: healthLoading } = useQuery({
    queryKey: ["health"],
    queryFn: () => enhancedApi.health(),
  });

  const { data: providersData, isLoading: providersLoading } = useQuery({
    queryKey: ["providers"],
    queryFn: () => enhancedApi.providers.list(),
  });

  const { data: techniquesData, isLoading: techniquesLoading } = useQuery({
    queryKey: ["techniques"],
    queryFn: () => enhancedApi.techniques(),
  });

  const health = healthData;
  const providers = providersData?.data?.providers || [];
  const techniques = techniquesData;

  const activeProviders = providers.filter((p) => p.status === "active").length;

  const panels = [
    {
      title: "Execution Panel",
      description: "Transform prompts and execute with LLM providers",
      icon: Zap,
      href: "/dashboard/execution",
      color: "text-orange-400",
      bgColor: "bg-orange-500/10",
      borderColor: "hover:border-orange-500/50",
    },
    {
      title: "Generation Panel",
      description: "Direct LLM interaction without transformation",
      icon: Sparkles,
      href: "/dashboard/generation",
      color: "text-purple-400",
      bgColor: "bg-purple-500/10",
      borderColor: "hover:border-purple-500/50",
    },
    {
      title: "Jailbreak Generator",
      description: "Advanced jailbreak prompt creation",
      icon: Skull,
      href: "/dashboard/jailbreak",
      color: "text-red-400",
      bgColor: "bg-red-500/10",
      borderColor: "hover:border-red-500/50",
    },
    {
      title: "HouYi Optimizer",
      description: "Evolutionary prompt optimization",
      icon: Target,
      href: "/dashboard/houyi",
      color: "text-amber-400",
      bgColor: "bg-amber-500/10",
      borderColor: "hover:border-amber-500/50",
    },
    {
      title: "AutoDAN Generator",
      description: "AI-powered jailbreak generation",
      icon: Brain,
      href: "/dashboard/autodan",
      color: "text-pink-400",
      bgColor: "bg-pink-500/10",
      borderColor: "hover:border-pink-500/50",
    },
    {
      title: "Providers Manager",
      description: "View and manage LLM providers",
      icon: Server,
      href: "/dashboard/providers",
      color: "text-blue-400",
      bgColor: "bg-blue-500/10",
      borderColor: "hover:border-blue-500/50",
    },
    {
      title: "Techniques Explorer",
      description: "Browse 40+ transformation techniques",
      icon: Layers,
      href: "/dashboard/techniques",
      color: "text-emerald-400",
      bgColor: "bg-emerald-500/10",
      borderColor: "hover:border-emerald-500/50",
    },
    {
      title: "Metrics Dashboard",
      description: "Real-time system metrics and analytics",
      icon: Activity,
      href: "/dashboard/metrics",
      color: "text-cyan-400",
      bgColor: "bg-cyan-500/10",
      borderColor: "hover:border-cyan-500/50",
    },
  ];

  return (
    <div className="space-y-8 bg-pattern min-h-screen">
      {/* Header with gradient text */}
      <div className="flex flex-col gap-2">
        <h1 className="text-4xl font-bold tracking-tight gradient-text">
          Dashboard Overview
        </h1>
        <p className="text-muted-foreground text-lg">
          Welcome to <span className="font-semibold text-foreground">Project Chimera</span> — AI Prompt Transformation Engine for Security Research
        </p>
      </div>

      {/* Status Cards with glass effect */}
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
        <Card className="glass glass-hover card-transition">
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">System Status</CardTitle>
            <Activity className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="flex items-center gap-2">
              {healthLoading ? (
                <div className="flex items-center gap-2">
                  <div className="w-5 h-5 rounded-full shimmer" />
                  <span className="text-muted-foreground">Checking...</span>
                </div>
              ) : health?.status === "healthy" || health?.status === "degraded" ? (
                <>
                  <div className="relative">
                    <CheckCircle2 className={`h-5 w-5 ${health?.status === "healthy" ? "text-emerald-400" : "text-amber-400"}`} />
                    <div className={`absolute inset-0 rounded-full ${health?.status === "healthy" ? "glow-green" : ""}`} />
                  </div>
                  <span className={`text-2xl font-bold ${health?.status === "healthy" ? "text-emerald-400" : "text-amber-400"}`}>
                    {health?.status === "healthy" ? "Online" : "Degraded"}
                  </span>
                </>
              ) : (
                <>
                  <div className="relative">
                    <XCircle className="h-5 w-5 text-red-400" />
                    <div className="absolute inset-0 rounded-full glow-red" />
                  </div>
                  <span className="text-2xl font-bold text-red-400">Offline</span>
                </>
              )}
            </div>
          </CardContent>
        </Card>

        <Card className="glass glass-hover card-transition">
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Active Providers</CardTitle>
            <Server className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {providersLoading ? (
                <span className="shimmer inline-block w-16 h-8 rounded" />
              ) : (
                <span className="gradient-text">{activeProviders} / {providers.length}</span>
              )}
            </div>
            <p className="text-xs text-muted-foreground mt-1">LLM integrations</p>
          </CardContent>
        </Card>

        <Card className="glass glass-hover card-transition">
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Techniques</CardTitle>
            <Layers className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {techniquesLoading ? (
                <span className="shimmer inline-block w-12 h-8 rounded" />
              ) : (
                <span className="gradient-text">{techniques?.techniques?.length || 0}</span>
              )}
            </div>
            <p className="text-xs text-muted-foreground mt-1">Available suites</p>
          </CardContent>
        </Card>

        <Card className="glass glass-hover card-transition">
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Default Provider</CardTitle>
            <Zap className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold capitalize">
              {providersLoading ? (
                <span className="shimmer inline-block w-20 h-8 rounded" />
              ) : (
                <span className="gradient-text">{providersData?.data?.default || "N/A"}</span>
              )}
            </div>
            <p className="text-xs text-muted-foreground mt-1">Primary routing</p>
          </CardContent>
        </Card>
      </div>

      {/* Panel Cards with enhanced styling */}
      <div>
        <h2 className="text-xl font-semibold mb-4 flex items-center gap-2">
          <span className="gradient-text">Quick Access</span>
          <ArrowRight className="h-4 w-4 text-muted-foreground" />
        </h2>
        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
          {panels.map((panel) => (
            <Link key={panel.href} href={panel.href as unknown as "/dashboard"}>
              <Card className={`h-full glass glass-hover card-transition cursor-pointer group ${panel.borderColor}`}>
                <CardHeader>
                  <div className="flex items-center justify-between">
                    <div className={`p-3 rounded-xl ${panel.bgColor} transition-transform group-hover:scale-110`}>
                      <panel.icon className={`h-6 w-6 ${panel.color}`} />
                    </div>
                    <ArrowRight className="h-5 w-5 text-muted-foreground group-hover:text-primary group-hover:translate-x-1 transition-all" />
                  </div>
                  <CardTitle className="mt-4 group-hover:text-primary transition-colors">{panel.title}</CardTitle>
                  <CardDescription className="line-clamp-2">{panel.description}</CardDescription>
                </CardHeader>
              </Card>
            </Link>
          ))}
        </div>
      </div>

      {/* Provider Status with improved design */}
      {providers.length > 0 && (
        <div>
          <h2 className="text-xl font-semibold mb-4 gradient-text">Provider Status</h2>
          <div className="grid gap-3 md:grid-cols-4 lg:grid-cols-6">
            {providers.map((provider, index) => (
              <div
                key={`${provider.provider}-${index}`}
                className="flex items-center gap-3 p-4 rounded-xl glass glass-hover card-transition"
              >
                <div className="relative">
                  <div className={`w-2.5 h-2.5 rounded-full ${provider.status === "active" ? "bg-emerald-400 pulse-dot" : "bg-gray-500"
                    }`} />
                  {provider.status === "active" && (
                    <div className="absolute inset-0 w-2.5 h-2.5 rounded-full bg-emerald-400/30 animate-ping" />
                  )}
                </div>
                <span className="text-sm font-medium capitalize truncate">
                  {provider.provider}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
