"use client";

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import {
  Skull, Code, BookOpen, Target, Sparkles, Brain, Shield, Zap
} from "lucide-react";
import Link from "next/link";

const AI_TOOLS = [
  {
    id: "jailbreak",
    name: "Enhanced Jailbreak",
    description: "Advanced AI-powered prompt transformation with 25+ techniques",
    icon: Skull,
    color: "text-red-500",
    bgColor: "bg-red-500/10",
    borderColor: "border-red-500/20",
    href: "/dashboard/ai-tools/jailbreak",
    badge: "25+ Techniques",
    features: [
      "Content transformation (LeetSpeak, Homoglyphs, Caesar Cipher)",
      "Structural & semantic manipulation",
      "Advanced neural techniques",
      "Research-driven methods",
    ],
  },
  {
    id: "code",
    name: "Code Generator",
    description: "Generate clean, efficient, and secure code using AI",
    icon: Code,
    color: "text-blue-500",
    bgColor: "bg-blue-500/10",
    borderColor: "border-blue-500/20",
    href: "/dashboard/ai-tools/code",
    badge: "Multi-Language",
    features: [
      "Support for 10+ programming languages",
      "Syntax highlighting and formatting",
      "Security-focused code generation",
      "Thinking mode for complex tasks",
    ],
  },
  {
    id: "summarize",
    name: "Paper Summarizer",
    description: "Generate concise summaries of research papers",
    icon: BookOpen,
    color: "text-purple-500",
    bgColor: "bg-purple-500/10",
    borderColor: "border-purple-500/20",
    href: "/dashboard/ai-tools/summarize",
    badge: "Academic",
    features: [
      "Extract key contributions",
      "Identify methodology",
      "Summarize findings",
      "Deep analysis mode",
    ],
  },
  {
    id: "red-team",
    name: "Red Team Suite",
    description: "Generate comprehensive security test prompt suites",
    icon: Target,
    color: "text-orange-500",
    bgColor: "bg-orange-500/10",
    borderColor: "border-orange-500/20",
    href: "/dashboard/ai-tools/red-team",
    badge: "Security",
    features: [
      "5-7 test variants per concept",
      "Multiple adversarial techniques",
      "Detailed metadata",
      "Responsible use guidelines",
    ],
  },
];

export default function AIToolsPage() {
  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold tracking-tight flex items-center gap-2">
            <Sparkles className="h-8 w-8 text-primary" />
            AI Tools
          </h1>
          <p className="text-muted-foreground mt-1">
            Advanced AI-powered generation tools for security research
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Badge variant="outline" className="flex items-center gap-1">
            <Brain className="h-3 w-3" />
            Gemini Powered
          </Badge>
          <Badge variant="outline" className="flex items-center gap-1">
            <Shield className="h-3 w-3" />
            Research Only
          </Badge>
        </div>
      </div>

      {/* Tool Cards Grid */}
      <div className="grid gap-6 md:grid-cols-2">
        {AI_TOOLS.map((tool: any) => (
          <Link key={tool.id} href={tool.href as any}>
            <Card className={`h-full transition-all hover:shadow-lg hover:scale-[1.02] cursor-pointer border-2 ${tool.borderColor} hover:border-opacity-50`}>
              <CardHeader>
                <div className="flex items-start justify-between">
                  <div className={`p-3 rounded-lg ${tool.bgColor}`}>
                    <tool.icon className={`h-6 w-6 ${tool.color}`} />
                  </div>
                  <Badge variant="secondary">{tool.badge}</Badge>
                </div>
                <CardTitle className="mt-4">{tool.name}</CardTitle>
                <CardDescription>{tool.description}</CardDescription>
              </CardHeader>
              <CardContent>
                <ul className="space-y-2">
                  {tool.features.map((feature: any, i: number) => (
                    <li key={i} className="flex items-center gap-2 text-sm text-muted-foreground">
                      <Zap className="h-3 w-3 text-primary" />
                      {feature}
                    </li>
                  ))}
                </ul>
              </CardContent>
            </Card>
          </Link>
        ))}
      </div>

      {/* Quick Stats */}
      <div className="grid gap-4 md:grid-cols-4">
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-muted-foreground">
              Total Techniques
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">25+</div>
            <p className="text-xs text-muted-foreground">Jailbreak methods</p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-muted-foreground">
              Languages
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">10+</div>
            <p className="text-xs text-muted-foreground">Code generation</p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-muted-foreground">
              Providers
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">2</div>
            <p className="text-xs text-muted-foreground">Gemini & DeepSeek</p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-muted-foreground">
              Test Variants
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">5-7</div>
            <p className="text-xs text-muted-foreground">Per red team suite</p>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
