"use client";

import { useState } from "react";
import { useMutation } from "@tanstack/react-query";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Label } from "@/components/ui/label";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { toast } from "sonner";
import { FileText, Copy, Timer, Brain, BookOpen, Sparkles, Download } from "lucide-react";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Switch } from "@/components/ui/switch";
import { GeminiService, SummarizePaperRequest, SummarizePaperResponse } from "@/lib/services/gemini-service";

export function PaperSummarizer() {
  const [content, setContent] = useState("");
  const [isThinkingMode, setIsThinkingMode] = useState(false);
  const [provider, setProvider] = useState("google");
  const [result, setResult] = useState<SummarizePaperResponse | null>(null);

  const summarizeMutation = useMutation({
    mutationFn: async (request: SummarizePaperRequest) => {
      return await GeminiService.summarizePaper(request);
    },
    onSuccess: (response) => {
      setResult(response);
      toast.success("Paper Summarized", {
        description: `Summary generated in ${(response.execution_time_seconds * 1000).toFixed(0)}ms`,
      });
    },
    onError: (error: Error) => {
      console.error("Summarization failed", error);
      toast.error("Summarization Failed", {
        description: error.message || "An error occurred during summarization",
      });
      setResult(null);
    },
  });

  const handleSummarize = () => {
    if (!content.trim()) {
      toast.error("Validation Error", { description: "Please enter paper content to summarize" });
      return;
    }

    const request: SummarizePaperRequest = {
      content,
      is_thinking_mode: isThinkingMode,
      provider,
    };

    summarizeMutation.mutate(request);
  };

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text);
    toast.success("Copied to clipboard");
  };

  const downloadSummary = () => {
    if (!result?.summary) return;
    
    const blob = new Blob([result.summary], { type: "text/markdown" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "paper-summary.md";
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    toast.success("Summary downloaded");
  };

  const wordCount = content.trim().split(/\s+/).filter(Boolean).length;

  return (
    <div className="grid gap-6 lg:grid-cols-2">
      {/* Input Panel */}
      <Card className="h-fit">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <BookOpen className="h-5 w-5 text-purple-500" />
            Paper Summarizer
          </CardTitle>
          <CardDescription>
            Generate concise summaries of research papers and academic content
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          {/* Content Input */}
          <div className="space-y-2">
            <div className="flex justify-between items-center">
              <Label htmlFor="paper-content">Paper Content</Label>
              <span className="text-xs text-muted-foreground">{wordCount} words</span>
            </div>
            <Textarea
              id="paper-content"
              value={content}
              onChange={(e) => setContent(e.target.value)}
              rows={12}
              placeholder="Paste the research paper abstract, introduction, or full text here...&#10;&#10;The AI will extract key contributions, methodology, findings, and conclusions."
              className="font-mono text-sm"
              aria-describedby="paper-content-help"
            />
            <p id="paper-content-help" className="text-xs text-muted-foreground">
              Paste the paper abstract, introduction, or relevant sections for summarization
            </p>
          </div>

          {/* Options */}
          <div className="grid grid-cols-2 gap-4">
            <div className="space-y-2">
              <Label>Provider</Label>
              <Select value={provider} onValueChange={setProvider}>
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="google">Gemini</SelectItem>
                  <SelectItem value="deepseek">DeepSeek</SelectItem>
                </SelectContent>
              </Select>
            </div>
            <div className="flex items-center justify-between p-3 rounded-md border h-fit mt-6">
              <div>
                <Label className="text-sm font-medium">Thinking Mode</Label>
                <p className="text-xs text-muted-foreground">Deep analysis</p>
              </div>
              <Switch
                checked={isThinkingMode}
                onCheckedChange={setIsThinkingMode}
              />
            </div>
          </div>

          {/* Summarize Button */}
          <Button
            onClick={handleSummarize}
            disabled={summarizeMutation.isPending || !content.trim()}
            className="w-full bg-purple-600 hover:bg-purple-700"
            size="lg"
          >
            {summarizeMutation.isPending ? (
              <>
                <div className="mr-2 h-4 w-4 animate-spin rounded-full border-2 border-current border-t-transparent" />
                Summarizing...
              </>
            ) : (
              <>
                <Sparkles className="mr-2 h-4 w-4" />
                Summarize Paper
              </>
            )}
          </Button>

          {/* Tips */}
          <div className="p-3 rounded-md bg-purple-500/10 border border-purple-500/20">
            <p className="text-xs text-purple-600 dark:text-purple-400">
              <strong>Tips:</strong> For best results, include the abstract and introduction. 
              The AI will identify key contributions, methodology, and conclusions.
            </p>
          </div>
        </CardContent>
      </Card>

      {/* Result Panel */}
      <Card className="flex flex-col h-fit min-h-[500px]">
        <CardHeader>
          <CardTitle className="flex items-center justify-between">
            <span className="flex items-center gap-2">
              <FileText className="h-5 w-5" />
              Summary
            </span>
            {result && (
              <div className="flex gap-2">
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => copyToClipboard(result.summary)}
                >
                  <Copy className="mr-2 h-3 w-3" />
                  Copy
                </Button>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={downloadSummary}
                >
                  <Download className="mr-2 h-3 w-3" />
                  Download
                </Button>
              </div>
            )}
          </CardTitle>
          <CardDescription>
            {result ? (
              <div className="flex flex-wrap gap-2 mt-1">
                <Badge variant={result.success ? "default" : "destructive"}>
                  {result.success ? "Success" : "Failed"}
                </Badge>
                <Badge variant="outline" className="flex items-center gap-1">
                  <Timer className="h-3 w-3" /> {(result.execution_time_seconds * 1000).toFixed(0)}ms
                </Badge>
                <Badge variant="outline">
                  {result.provider}
                </Badge>
                {isThinkingMode && (
                  <Badge variant="secondary">
                    <Brain className="h-3 w-3 mr-1" />
                    Deep Analysis
                  </Badge>
                )}
              </div>
            ) : (
              "Paper summary will appear here"
            )}
          </CardDescription>
        </CardHeader>
        <CardContent className="flex-1 flex flex-col gap-4">
          {result ? (
            <>
              <ScrollArea className="flex-1 min-h-[350px] w-full rounded-md border p-4 bg-card">
                <div className="prose prose-sm dark:prose-invert max-w-none">
                  {/* Render markdown-like content */}
                  {result.summary.split('\n').map((line, i) => {
                    if (line.startsWith('# ')) {
                      return <h1 key={i} className="text-xl font-bold mt-4 mb-2">{line.slice(2)}</h1>;
                    }
                    if (line.startsWith('## ')) {
                      return <h2 key={i} className="text-lg font-semibold mt-3 mb-2">{line.slice(3)}</h2>;
                    }
                    if (line.startsWith('### ')) {
                      return <h3 key={i} className="text-base font-medium mt-2 mb-1">{line.slice(4)}</h3>;
                    }
                    if (line.startsWith('- ')) {
                      return <li key={i} className="ml-4">{line.slice(2)}</li>;
                    }
                    if (line.startsWith('**') && line.endsWith('**')) {
                      return <p key={i} className="font-semibold">{line.slice(2, -2)}</p>;
                    }
                    if (line.trim() === '') {
                      return <br key={i} />;
                    }
                    return <p key={i} className="mb-2">{line}</p>;
                  })}
                </div>
              </ScrollArea>

              {/* Model info */}
              <div className="flex items-center gap-2 text-xs text-muted-foreground">
                <span>Model: {result.model_used}</span>
              </div>

              {/* Error display */}
              {result.error && (
                <div className="p-3 rounded-md bg-red-500/10 border border-red-500/20">
                  <p className="text-sm text-red-500">{result.error}</p>
                </div>
              )}
            </>
          ) : (
            <div className="flex items-center justify-center h-full text-muted-foreground flex-col gap-2">
              <BookOpen className="h-12 w-12 opacity-20" />
              <p>Ready to summarize</p>
              <p className="text-xs">Paste paper content to get started</p>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}

export default PaperSummarizer;