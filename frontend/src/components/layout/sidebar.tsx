"use client";

import { useState, memo, useCallback } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import {
  ShieldAlert,
  LayoutDashboard,
  Terminal,
  GitBranch,
  Sliders,
  Zap,
  Sparkles,
  Server,
  Layers,
  Skull,
  Activity,
  Wifi,
  Bug,
  Brain,
  Dna,
  Cpu,
  Heart,
  Wand2,
  Code,
  BookOpen,
  Target,
  ChevronDown,
  Shield,
  Repeat,
  Users,
  Key,
  Radar,
} from "lucide-react";
import type { LucideIcon } from "lucide-react";

interface RouteConfig {
  label: string;
  icon: LucideIcon;
  href: string;
  description?: string;
}

interface SidebarProps extends React.HTMLAttributes<HTMLDivElement> {
  /** Optional callback when a navigation item is clicked */
  onNavigate?: (href: string) => void;
  /** Whether to show the logo/header */
  showLogo?: boolean;
}

/**
 * Static route configurations - defined outside component to prevent recreation (PERF-010)
 * These never change, so they should be module-level constants
 */
const MAIN_ROUTES: RouteConfig[] = [
  { label: "Overview", icon: LayoutDashboard, href: "/dashboard" },
  { label: "Execution", icon: Zap, href: "/dashboard/execution", description: "Transform + Execute" },
  { label: "Generation", icon: Sparkles, href: "/dashboard/generation", description: "Direct LLM" },
  { label: "GPTFuzz", icon: Bug, href: "/dashboard/gptfuzz", description: "Evolutionary Fuzzing" },
  { label: "AutoDAN", icon: Brain, href: "/dashboard/autodan", description: "AutoDAN Reasoning" },
  { label: "AutoDAN-Turbo", icon: Zap, href: "/dashboard/autodan-turbo", description: "Lifelong Strategy Learning" },
  { label: "HouYi", icon: Dna, href: "/dashboard/houyi", description: "Prompt Optimization" },
  { label: "Gradient Optimizer", icon: Target, href: "/dashboard/gradient", description: "HotFlip/GCG" },
  { label: "Lifelong Learning", icon: Repeat, href: "/dashboard/lifelong", description: "Continuous Learning" },
  { label: "Aegis Campaign", icon: Radar, href: "/dashboard/aegis-campaign", description: "Real-Time Telemetry" },
  { label: "Evasion Tasks", icon: Shield, href: "/evasion", description: "Prompt Evasion" },
];

const AI_TOOLS_ROUTES: RouteConfig[] = [
  { label: "AI Tools Hub", icon: Wand2, href: "/dashboard/ai-tools", description: "All AI Tools" },
  { label: "Code Generator", icon: Code, href: "/dashboard/ai-tools/code", description: "Multi-Language" },
  { label: "Paper Summarizer", icon: BookOpen, href: "/dashboard/ai-tools/summarize", description: "Academic" },
  { label: "Red Team Suite", icon: Target, href: "/dashboard/ai-tools/red-team", description: "Security Testing" },
];

const RESOURCE_ROUTES: RouteConfig[] = [
  { label: "Models", icon: Cpu, href: "/dashboard/models", description: "Model & Session Mgmt" },
  { label: "AI Providers", icon: Server, href: "/dashboard/providers", description: "Configure & Switch Providers" },
  { label: "Techniques", icon: Layers, href: "/dashboard/techniques" },
  { label: "Metrics", icon: Activity, href: "/dashboard/metrics" },
];

const CONFIG_ROUTES: RouteConfig[] = [
  { label: "API Keys", icon: Key, href: "/dashboard/api-keys", description: "Manage API Keys" },
  { label: "Mutators", icon: GitBranch, href: "/dashboard/config/mutators" },
  { label: "Policies", icon: Sliders, href: "/dashboard/config/policies" },
  { label: "Settings", icon: Wifi, href: "/dashboard/settings", description: "Connection & Config" },
];

const SYSTEM_ROUTES: RouteConfig[] = [
  { label: "Activity Feed", icon: Activity, href: "/dashboard/activity" },
  { label: "Health Status", icon: Heart, href: "/dashboard/health" },
];

const ADMIN_ROUTES: RouteConfig[] = [
  { label: "Admin Dashboard", icon: Users, href: "/admin" },
];

/**
 * Memoized navigation item component (PERF-011)
 * Prevents re-renders when other parts of the sidebar change
 */
const NavItem = memo(function NavItem({
  route,
  isActive,
  onNavigate,
  className,
}: {
  route: RouteConfig;
  isActive: boolean;
  onNavigate?: (href: string) => void;
  className?: string;
}) {
  const Icon = route.icon;
  const handleClick = useCallback(() => {
    onNavigate?.(route.href);
  }, [onNavigate, route.href]);

  return (
    <Link href={route.href as unknown as "/dashboard"} onClick={handleClick}>
      <Button
        variant={isActive ? "secondary" : "ghost"}
        className={cn("w-full justify-start", className)}
      >
        <Icon className="mr-2 h-4 w-4" />
        {route.label}
      </Button>
    </Link>
  );
});

/**
 * Memoized route section component (PERF-012)
 */
const RouteSection = memo(function RouteSection({
  title,
  routes,
  pathname,
  onNavigate,
  itemClassName,
}: {
  title: string;
  routes: RouteConfig[];
  pathname: string | null;
  onNavigate?: (href: string) => void;
  itemClassName?: string;
}) {
  return (
    <div className="px-3 py-2">
      <h3 className="mb-2 px-4 text-xs font-semibold tracking-tight text-muted-foreground uppercase">
        {title}
      </h3>
      <div className="space-y-1">
        {routes.map((route) => (
          <NavItem
            key={route.href}
            route={route}
            isActive={pathname === route.href}
            onNavigate={onNavigate}
            className={itemClassName}
          />
        ))}
      </div>
    </div>
  );
});

/**
 * Optimized Sidebar component with memoization (PERF-013)
 * - Static routes moved outside component
 * - Memoized child components
 * - Stable callback references
 */
export function Sidebar({ className, onNavigate, showLogo = true }: SidebarProps) {
  const pathname = usePathname();
  const [aiToolsOpen, setAiToolsOpen] = useState(pathname?.startsWith("/dashboard/ai-tools") ?? false);

  // Memoize the toggle handler to prevent unnecessary re-renders
  const handleAiToolsToggle = useCallback((open: boolean) => {
    setAiToolsOpen(open);
  }, []);

  return (
    <div className={cn("flex flex-col h-full bg-sidebar glass", className)}>
      <ScrollArea className="flex-1 h-full">
        <div className="space-y-4 py-4 pb-12">
          <div className="px-3 py-2">
            {showLogo && (
              <div className="flex items-center pl-2 mb-8 group">
                <div className="relative">
                  <ShieldAlert className="h-8 w-8 mr-2 text-primary glow-primary" />
                  <div className="absolute inset-0 blur-md bg-primary/20 -z-10" />
                </div>
                <h2 className="text-2xl font-bold tracking-tight gradient-text">Chimera</h2>
              </div>
            )}

            {/* Main Operations - using memoized components */}
            <div className="space-y-1">
              <h3 className="mb-2 px-4 text-xs font-semibold tracking-tight text-muted-foreground uppercase">
                Operations
              </h3>
              {MAIN_ROUTES.map((route) => (
                <NavItem
                  key={route.href}
                  route={route}
                  isActive={pathname === route.href}
                  onNavigate={onNavigate}
                />
              ))}
            </div>
          </div>

          {/* AI Tools - Collapsible with memoized items */}
          <div className="px-3 py-2">
            <Collapsible open={aiToolsOpen} onOpenChange={handleAiToolsToggle}>
              <CollapsibleTrigger asChild>
                <Button
                  variant="ghost"
                  className="w-full justify-between px-4 mb-2"
                >
                  <span className="flex items-center text-xs font-semibold tracking-tight text-muted-foreground uppercase">
                    <Wand2 className="mr-2 h-4 w-4" />
                    AI Tools
                  </span>
                  <ChevronDown className={cn(
                    "h-4 w-4 text-muted-foreground transition-transform duration-200",
                    aiToolsOpen && "rotate-180"
                  )} />
                </Button>
              </CollapsibleTrigger>
              <CollapsibleContent className="space-y-1">
                {AI_TOOLS_ROUTES.map((route) => (
                  <NavItem
                    key={route.href}
                    route={route}
                    isActive={pathname === route.href}
                    onNavigate={onNavigate}
                    className="pl-6"
                  />
                ))}
              </CollapsibleContent>
            </Collapsible>
          </div>

          {/* Resources - using memoized RouteSection */}
          <RouteSection
            title="Resources"
            routes={RESOURCE_ROUTES}
            pathname={pathname}
            onNavigate={onNavigate}
          />

          {/* Configuration - using memoized RouteSection */}
          <RouteSection
            title="Configuration"
            routes={CONFIG_ROUTES}
            pathname={pathname}
            onNavigate={onNavigate}
          />

          {/* System */}
          <div className="px-3 py-2">
            <h3 className="mb-2 px-4 text-xs font-semibold tracking-tight text-muted-foreground uppercase">
              System
            </h3>
            <div className="space-y-1">
              {SYSTEM_ROUTES.map((route) => (
                <NavItem
                  key={route.href}
                  route={route}
                  isActive={pathname === route.href}
                  onNavigate={onNavigate}
                />
              ))}
              <Button variant="ghost" className="w-full justify-start text-muted-foreground hover:text-foreground" asChild>
                <a href="https://github.com/your-org/chimera/issues/new" target="_blank" rel="noopener noreferrer">
                  <Bug className="mr-2 h-4 w-4" />
                  Report Issue
                </a>
              </Button>
              <Button variant="ghost" className="w-full justify-start">
                <Terminal className="mr-2 h-4 w-4" />
                System Logs
              </Button>
            </div>
          </div>

          {/* Admin */}
          <RouteSection
            title="Administration"
            routes={ADMIN_ROUTES}
            pathname={pathname}
            onNavigate={onNavigate}
          />
        </div>
      </ScrollArea>
    </div>
  );
}
