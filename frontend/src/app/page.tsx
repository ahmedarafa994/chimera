"use client";

import { useState, useEffect, useMemo } from "react";
import Link from "next/link";
import {
  ShieldAlert,
  Zap,
  Brain,
  Target,
  ArrowRight,
  Sparkles,
  Shield,
  Lock,
  Cpu,
  Network,
  ChevronDown,
  Github,
  ExternalLink,
  Play,
  CheckCircle,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";

// Animated counter component with optimized rendering
function AnimatedCounter({ value, suffix = "" }: { value: number; suffix?: string }) {
  const [count, setCount] = useState(0);

  useEffect(() => {
    const duration = 2000;
    const steps = 60;
    const increment = value / steps;
    let current = 0;
    const timer = setInterval(() => {
      current += increment;
      if (current >= value) {
        setCount(value);
        clearInterval(timer);
      } else {
        setCount(Math.floor(current));
      }
    }, duration / steps);
    return () => clearInterval(timer);
  }, [value]);

  return (
    <span className="tabular-nums">
      {count}
      {suffix}
    </span>
  );
}

const PARTICLES = [...Array(30)].map((_, i) => ({
  id: i,
  left: `${Math.random() * 100}%`,
  top: `${Math.random() * 100}%`,
  animationDelay: `${Math.random() * 5}s`,
  animationDuration: `${3 + Math.random() * 4}s`,
}));

function ParticleBackground() {
  const particles = useMemo(() => PARTICLES, []);

  return (
    <div className="absolute inset-0 overflow-hidden pointer-events-none" aria-hidden="true">
      {particles.map((particle) => (
        <div
          key={particle.id}
          className="absolute w-1 h-1 bg-primary/40 rounded-full animate-pulse-slow"
          style={{
            left: particle.left,
            top: particle.top,
            animationDelay: particle.animationDelay,
            animationDuration: particle.animationDuration,
          }}
        />
      ))}
    </div>
  );
}

// Feature card component with hover effects
function FeatureCard({
  icon: Icon,
  title,
  description,
  index,
}: {
  icon: React.ComponentType<{ className?: string }>;
  title: string;
  description: string;
  index: number;
}) {
  return (
    <div
      className={cn(
        "group relative p-6 rounded-2xl",
        "bg-white/[0.03] backdrop-blur-md border border-white/[0.08]",
        "hover:bg-white/[0.06] hover:border-primary/40",
        "transition-all duration-300 ease-out hover:-translate-y-1",
        "animate-fade-in-up"
      )}
      style={{ animationDelay: `${index * 100}ms` }}
    >
      {/* Gradient overlay on hover */}
      <div className="absolute inset-0 bg-gradient-to-br from-primary/10 via-transparent to-purple-500/5 opacity-0 group-hover:opacity-100 rounded-2xl transition-opacity duration-300" />

      <div className="relative z-10">
        <div className={cn(
          "w-12 h-12 rounded-xl flex items-center justify-center mb-4",
          "bg-gradient-to-br from-primary/20 to-primary/5",
          "group-hover:scale-110 group-hover:from-primary/30 group-hover:to-primary/10",
          "transition-all duration-300"
        )}>
          <Icon className="w-6 h-6 text-primary" />
        </div>
        <h3 className="text-lg font-semibold mb-2 text-foreground group-hover:text-primary transition-colors duration-200">
          {title}
        </h3>
        <p className="text-muted-foreground text-sm leading-relaxed">
          {description}
        </p>
      </div>
    </div>
  );
}

// Stats card component
function StatCard({ label, value, suffix, index }: { label: string; value: number; suffix: string; index: number }) {
  return (
    <div
      className={cn(
        "text-center p-6 rounded-2xl",
        "bg-gradient-to-br from-white/[0.04] to-white/[0.01]",
        "border border-white/[0.08]",
        "animate-fade-in-up"
      )}
      style={{ animationDelay: `${index * 100}ms` }}
    >
      <div className="text-3xl md:text-4xl font-bold mb-1">
        <span className="bg-gradient-to-r from-primary via-purple-400 to-pink-400 bg-clip-text text-transparent">
          <AnimatedCounter value={value} suffix={suffix} />
        </span>
      </div>
      <div className="text-sm text-muted-foreground font-medium">{label}</div>
    </div>
  );
}

// Main landing page
export default function HomePage() {
  const [mounted, setMounted] = useState(false);
  const [scrollY, setScrollY] = useState(0);

  useEffect(() => {
    setMounted(true);  // eslint-disable-line react-hooks/set-state-in-effect

    const handleScroll = () => {
      setScrollY(window.scrollY);
    };

    window.addEventListener("scroll", handleScroll, { passive: true });
    return () => window.removeEventListener("scroll", handleScroll);
  }, []);

  const features = useMemo(() => [
    {
      icon: Brain,
      title: "AI-Powered Jailbreaks",
      description: "Leverage advanced AI to automatically generate sophisticated bypass prompts with intent analysis and reasoning.",
    },
    {
      icon: Target,
      title: "Evolutionary Optimization",
      description: "HouYi and AutoDAN algorithms evolve prompts across generations for maximum red-team effectiveness.",
    },
    {
      icon: Shield,
      title: "DeepTeam Framework",
      description: "Multi-agent collaborative attack simulation with configurable personas and strategies.",
    },
    {
      icon: Zap,
      title: "40+ Technique Suites",
      description: "From basic obfuscation to quantum-level complexity—comprehensive coverage for all scenarios.",
    },
    {
      icon: Network,
      title: "Multi-Provider Support",
      description: "Unified API for OpenAI, Anthropic, Google, DeepSeek, Routeway, BigModel, and more.",
    },
    {
      icon: Lock,
      title: "Security Research Focus",
      description: "Built for legitimate red-team testing, LLM safety evaluation, and security auditing.",
    },
  ], []);

  const stats = useMemo(() => [
    { label: "Techniques", value: 40, suffix: "+" },
    { label: "Attack Vectors", value: 15, suffix: "+" },
    { label: "Model Providers", value: 10, suffix: "+" },
    { label: "Success Rate", value: 94, suffix: "%" },
  ], []);

  const capabilities = useMemo(() => [
    "AutoDAN-Turbo with lifelong learning",
    "HouYi evolutionary prompt optimization",
    "GPTFuzz mutation-based fuzzing",
    "Intent-aware jailbreak generation",
    "Cross-model attack transfer",
    "Real-time metrics and analytics",
  ], []);

  if (!mounted) {
    return (
      <div className="min-h-screen bg-background flex items-center justify-center">
        <div className="w-12 h-12 border-2 border-primary border-t-transparent rounded-full animate-spin" />
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-background text-foreground overflow-x-hidden">
      {/* Background effects */}
      <div className="fixed inset-0 pointer-events-none overflow-hidden">
        {/* Gradient orbs */}
        <div
          className="absolute top-1/4 -left-32 w-96 h-96 bg-primary/20 rounded-full blur-[100px] animate-pulse-slow"
          style={{ transform: `translateY(${scrollY * 0.1}px)` }}
        />
        <div
          className="absolute bottom-1/4 -right-32 w-96 h-96 bg-purple-500/15 rounded-full blur-[100px] animate-pulse-slow"
          style={{ animationDelay: "2s", transform: `translateY(${-scrollY * 0.1}px)` }}
        />
        <ParticleBackground />
      </div>

      {/* Navigation */}
      <nav className={cn(
        "fixed top-0 left-0 right-0 z-50 transition-all duration-300",
        scrollY > 50 ? "bg-background/80 backdrop-blur-lg border-b border-white/[0.05]" : "bg-transparent"
      )}>
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            <div className="flex items-center gap-2">
              <ShieldAlert className="w-8 h-8 text-primary" />
              <span className="text-xl font-bold bg-gradient-to-r from-primary to-purple-400 bg-clip-text text-transparent">
                Chimera
              </span>
            </div>
            <div className="flex items-center gap-4">
              <Button variant="ghost" size="sm" asChild className="hidden sm:inline-flex">
                <a href="#features">Features</a>
              </Button>
              <Button asChild size="sm" className="gap-1">
                <Link href="/dashboard">
                  Dashboard
                  <ArrowRight className="w-4 h-4" />
                </Link>
              </Button>
            </div>
          </div>
        </div>
      </nav>

      {/* Hero Section */}
      <section className="relative min-h-screen flex flex-col items-center justify-center px-4 pt-24 pb-16">
        {/* Animated logo */}
        <div className="relative mb-8 animate-fade-in">
          <div className="absolute inset-0 blur-3xl bg-primary/40 rounded-full animate-pulse-slow" />
          <ShieldAlert className="w-20 h-20 md:w-28 md:h-28 text-primary relative z-10 drop-shadow-glow" />
        </div>

        {/* Title */}
        <h1 className="text-4xl sm:text-5xl md:text-6xl lg:text-7xl font-bold text-center mb-6 animate-fade-in-up" style={{ animationDelay: "200ms" }}>
          <span className="bg-gradient-to-r from-white via-white to-white/80 bg-clip-text text-transparent">
            Project{" "}
          </span>
          <span className="bg-gradient-to-r from-primary via-purple-400 to-pink-400 bg-clip-text text-transparent">
            Chimera
          </span>
        </h1>

        {/* Subtitle */}
        <p className="text-lg sm:text-xl md:text-2xl text-muted-foreground text-center max-w-3xl mb-8 animate-fade-in-up" style={{ animationDelay: "400ms" }}>
          Advanced AI Prompt Transformation Engine for{" "}
          <span className="text-foreground font-semibold">Security Research</span>
          {" "}and Red-Team Operations
        </p>

        {/* Feature badges */}
        <div className="flex flex-wrap justify-center gap-2 mb-10 animate-fade-in-up" style={{ animationDelay: "600ms" }}>
          {[
            { icon: Cpu, label: "AutoDAN-Turbo" },
            { icon: Target, label: "HouYi Optimizer" },
            { icon: Brain, label: "Intent-Aware" },
            { icon: Shield, label: "DeepTeam" },
          ].map((badge, _i) => (
            <Badge
              key={badge.label}
              variant="outline"
              className="px-3 py-1.5 bg-white/[0.03] border-white/[0.1] hover:border-primary/40 hover:bg-primary/5 transition-colors cursor-default"
            >
              <badge.icon className="w-3.5 h-3.5 mr-1.5 text-primary" />
              {badge.label}
            </Badge>
          ))}
        </div>

        {/* CTA Buttons */}
        <div className="flex flex-col sm:flex-row gap-4 animate-fade-in-up" style={{ animationDelay: "800ms" }}>
          <Button asChild size="lg" className="gap-2 text-base sm:text-lg px-6 sm:px-8 h-12 sm:h-14 shadow-lg shadow-primary/20 hover:shadow-primary/30 transition-shadow">
            <Link href="/dashboard">
              <Play className="w-5 h-5" />
              Launch Dashboard
              <ArrowRight className="w-4 h-4" />
            </Link>
          </Button>
          <Button asChild variant="outline" size="lg" className="gap-2 text-base sm:text-lg px-6 sm:px-8 h-12 sm:h-14 bg-white/[0.02] border-white/[0.1] hover:bg-white/[0.05]">
            <a href="https://github.com" target="_blank" rel="noopener noreferrer">
              <Github className="w-5 h-5" />
              View Source
              <ExternalLink className="w-4 h-4" />
            </a>
          </Button>
        </div>

        {/* Test Button */}
        <div className="mt-6 animate-fade-in-up" style={{ animationDelay: "1000ms" }}>
          <Button
            variant="secondary"
            size="lg"
            onClick={() => {
              console.log("Test button clicked!");
            }}
            className="gap-2 px-6 h-12 bg-white/[0.05] border border-white/[0.1] hover:bg-white/[0.08] transition-all"
          >
            Test Button
          </Button>
        </div>

        {/* Scroll indicator */}
        <div className="absolute bottom-8 animate-bounce" style={{ animationDuration: "2s" }}>
          <ChevronDown className="w-6 h-6 text-muted-foreground/50" />
        </div>
      </section>

      {/* Features Section */}
      <section id="features" className="relative py-24 px-4">
        <div className="max-w-6xl mx-auto">
          <div className="text-center mb-16">
            <h2 className="text-3xl md:text-4xl font-bold mb-4">
              <span className="bg-gradient-to-r from-primary via-purple-400 to-pink-400 bg-clip-text text-transparent">
                Powerful Features
              </span>
            </h2>
            <p className="text-muted-foreground text-lg max-w-2xl mx-auto">
              A comprehensive toolkit for AI security research, featuring cutting-edge
              techniques and battle-tested algorithms.
            </p>
          </div>

          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
            {features.map((feature, i) => (
              <FeatureCard key={feature.title} {...feature} index={i} />
            ))}
          </div>
        </div>
      </section>

      {/* Stats Section */}
      <section className="relative py-24 px-4">
        <div className="max-w-4xl mx-auto">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 md:gap-6">
            {stats.map((stat, i) => (
              <StatCard key={stat.label} {...stat} index={i} />
            ))}
          </div>
        </div>
      </section>

      {/* Capabilities Section */}
      <section className="relative py-24 px-4">
        <div className="max-w-4xl mx-auto">
          <div className="grid md:grid-cols-2 gap-12 items-center">
            <div>
              <h2 className="text-3xl md:text-4xl font-bold mb-6">
                <span className="bg-gradient-to-r from-primary via-purple-400 to-pink-400 bg-clip-text text-transparent">
                  Built for Professionals
                </span>
              </h2>
              <p className="text-muted-foreground text-lg mb-8">
                Everything you need for comprehensive LLM security testing,
                from automated fuzzing to sophisticated jailbreak generation.
              </p>
              <ul className="space-y-3">
                {capabilities.map((cap, i) => (
                  <li key={i} className="flex items-center gap-3 text-foreground/90">
                    <CheckCircle className="w-5 h-5 text-primary flex-shrink-0" />
                    <span>{cap}</span>
                  </li>
                ))}
              </ul>
            </div>
            <div className="relative">
              <div className="absolute inset-0 bg-gradient-to-r from-primary/20 to-purple-500/20 rounded-3xl blur-2xl" />
              <div className="relative p-8 rounded-3xl bg-white/[0.03] border border-white/[0.08] backdrop-blur-sm">
                <div className="aspect-video rounded-xl bg-gradient-to-br from-primary/20 via-purple-500/10 to-transparent flex items-center justify-center">
                  <ShieldAlert className="w-24 h-24 text-primary/50" />
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="relative py-24 px-4">
        <div className="max-w-xl mx-auto text-center">
          <h2 className="text-3xl md:text-4xl font-bold mb-4">
            Ready to start?
          </h2>
          <p className="text-muted-foreground text-lg mb-8">
            Begin your AI security research with the most advanced LLM testing platform.
          </p>
          <Button asChild size="lg" className="gap-2 text-lg px-10 h-14 shadow-lg shadow-primary/20">
            <Link href="/dashboard">
              <Sparkles className="w-5 h-5" />
              Get Started Now
              <ArrowRight className="w-5 h-5" />
            </Link>
          </Button>
        </div>
      </section>

      {/* Footer */}
      <footer className="relative py-8 px-4 border-t border-white/[0.05]">
        <div className="max-w-6xl mx-auto flex flex-col md:flex-row items-center justify-between gap-4">
          <div className="flex items-center gap-2">
            <ShieldAlert className="w-5 h-5 text-primary" />
            <span className="font-semibold">Project Chimera</span>
          </div>
          <p className="text-sm text-muted-foreground text-center">
            For legitimate security research purposes only.
          </p>
          <div className="flex items-center gap-6">
            <a href="#" className="text-sm text-muted-foreground hover:text-foreground transition-colors">
              Docs
            </a>
            <a href="#" className="text-sm text-muted-foreground hover:text-foreground transition-colors">
              Privacy
            </a>
            <a href="#" className="text-sm text-muted-foreground hover:text-foreground transition-colors">
              Terms
            </a>
          </div>
        </div>
      </footer>

      {/* Custom styles for animations */}
      <style jsx global>{`
        @keyframes fade-in {
          from { opacity: 0; }
          to { opacity: 1; }
        }

        @keyframes fade-in-up {
          from {
            opacity: 0;
            transform: translateY(20px);
          }
          to {
            opacity: 1;
            transform: translateY(0);
          }
        }

        @keyframes pulse-slow {
          0%, 100% { opacity: 0.4; }
          50% { opacity: 0.8; }
        }

        .animate-fade-in {
          animation: fade-in 0.6s ease-out forwards;
        }

        .animate-fade-in-up {
          opacity: 0;
          animation: fade-in-up 0.6s ease-out forwards;
        }

        .animate-pulse-slow {
          animation: pulse-slow 4s ease-in-out infinite;
        }

        .drop-shadow-glow {
          filter: drop-shadow(0 0 20px oklch(65% 0.22 260 / 0.5));
        }
      `}</style>
    </div>
  );
}
