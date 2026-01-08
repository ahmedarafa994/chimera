"use client";

import { useQuery } from "@tanstack/react-query";
import enhancedApi from "@/lib/api-enhanced";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Badge } from "@/components/ui/badge";
import { Shield, Box, Activity, Server, AlertCircle } from "lucide-react";
import { ScrollArea } from "@/components/ui/scroll-area";

export function ProvidersPanel() {
  const { data, isLoading, error } = useQuery({
    queryKey: ["providers"],
    queryFn: () => enhancedApi.providers.list(),
  });

  if (isLoading) {
    return <div className="p-8 text-center text-muted-foreground">Loading providers...</div>;
  }

  if (error || !data) {
    return (
      <div className="p-8 text-center text-destructive flex flex-col items-center gap-2">
        <AlertCircle className="h-8 w-8" />
        <p>Failed to load providers list</p>
      </div>
    );
  }

  // Filter out null/undefined providers to prevent runtime errors
  const providers = (data.data.providers || []).filter(
    (p): p is NonNullable<typeof p> => p != null && typeof p.provider === 'string'
  );

  return (
    <div className="space-y-6">
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Total Providers</CardTitle>
            <Server className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{providers.length}</div>
            <p className="text-xs text-muted-foreground">Active integrations</p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Default Provider</CardTitle>
            <Shield className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold capitalize">{data.data.default}</div>
            <p className="text-xs text-muted-foreground">Primary routing</p>
          </CardContent>
        </Card>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Connected LLM Providers</CardTitle>
          <CardDescription>
            Status and capabilities of all registered AI endpoints via AIClient-2-API
          </CardDescription>
        </CardHeader>
        <CardContent>
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Provider</TableHead>
                <TableHead>Status</TableHead>
                <TableHead>Current Model</TableHead>
                <TableHead>Available Models</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {providers.map((provider, index) => (
                <TableRow key={`${provider.provider}-${index}`}>
                  <TableCell className="font-medium flex items-center gap-2">
                    <Box className="h-4 w-4 text-muted-foreground" />
                    <span className="capitalize">
                      {provider.provider === "google" || provider.provider === "gemini"
                        ? "Gemini AI"
                        : provider.provider === "gemini-cli"
                        ? "Gemini CLI"
                        : provider.provider === "antigravity"
                        ? "Antigravity (Hybrid)"
                        : provider.provider === "anthropic"
                        ? "Anthropic Claude"
                        : provider.provider}
                    </span>
                  </TableCell>
                  <TableCell>
                    <Badge variant={provider.status === "active" ? "default" : "secondary"}>
                      {provider.status}
                    </Badge>
                  </TableCell>
                  <TableCell className="font-mono text-xs">{provider.model}</TableCell>
                  <TableCell>
                    <ScrollArea className="h-[60px] w-[300px]">
                      <div className="flex flex-wrap gap-1">
                        {(provider.available_models ?? []).map((model, index) => (
                          <Badge key={`${provider.provider}-${model}-${index}`} variant="outline" className="text-[10px]">
                            {model}
                          </Badge>
                        ))}
                      </div>
                    </ScrollArea>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </CardContent>
      </Card>
    </div>
  );
}