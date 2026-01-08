"use client";

import { useState } from "react";
import { UnifiedProviderSelector } from "@/components/providers";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { ScrollArea } from "@/components/ui/scroll-area";
import { useUnifiedProviders, useAllUnifiedModels } from "@/hooks";
import { toast } from "sonner";
import {
  Loader2,
  Server,
  Cpu,
  Activity as ActivityIcon,
  CheckCircle,
  XCircle
} from "lucide-react";

export default function ModelsPage() {
  const [activeTab, setActiveTab] = useState("selector");

  // Use unified provider hooks with TanStack Query
  const { data: providersData, isLoading: providersLoading } = useUnifiedProviders();
  const { data: allModelsData, isLoading: modelsLoading } = useAllUnifiedModels();

  const isLoading = providersLoading || modelsLoading;

  const handleSelectionChange = (provider: string, model: string, scope: string) => {
    toast.success("Model selection updated", {
      description: `Now using ${model} from ${provider} (${scope})`,
    });
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-[400px]">
        <Loader2 className="h-8 w-8 animate-spin mr-2" />
        <span>Loading model data...</span>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="flex flex-col gap-2">
        <h1 className="text-3xl font-bold tracking-tight">Model Management</h1>
        <p className="text-muted-foreground">
          Configure AI providers, models, and session settings for your requests.
        </p>
      </div>


      <Tabs value={activeTab} onValueChange={setActiveTab}>
        <TabsList>
          <TabsTrigger value="selector">Model Selector</TabsTrigger>
          <TabsTrigger value="providers">Providers</TabsTrigger>
          <TabsTrigger value="stats">Statistics</TabsTrigger>
        </TabsList>

        <TabsContent value="selector" className="mt-6">
          <UnifiedProviderSelector
            onSelectionChange={handleSelectionChange}
            showSessionInfo={true}
            showConnectionStatus={true}
          />
        </TabsContent>

        <TabsContent value="providers" className="mt-6">
          <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
            {(providersData?.providers ?? []).map((provider, index) => (
              <Card key={provider.id || `provider-${index}`}>
                <CardHeader>
                  <div className="flex items-center justify-between">
                    <CardTitle className="flex items-center gap-2">
                      <Server className="h-5 w-5" />
                      {provider.name}
                    </CardTitle>
                    <Badge variant={provider.is_available ? "default" : "secondary"}>
                      {provider.is_available ? "available" : "unavailable"}
                    </Badge>
                  </div>
                  <CardDescription>
                    {provider.supported_models?.length ?? 0} models available
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <ScrollArea className="h-[200px]">
                      <div className="space-y-2">
                        {(provider.supported_models ?? []).map((model, modelIndex) => (
                        <div
                          key={model || `model-${modelIndex}`}
                          className="flex items-center justify-between p-2 rounded border"
                        >
                          <div className="flex items-center gap-2">
                            <Cpu className="h-4 w-4 text-muted-foreground" />
                            <span className="text-sm">{model}</span>
                          </div>
                        </div>
                      ))}
                    </div>
                  </ScrollArea>
                </CardContent>
              </Card>
            ))}
          </div>
        </TabsContent>

        <TabsContent value="stats" className="mt-6">
          <div className="grid gap-6 md:grid-cols-3">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <ActivityIcon className="h-5 w-5" />
                  Total Models
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-4xl font-bold">
                  {allModelsData?.total ?? 0}
                </div>
                <p className="text-sm text-muted-foreground mt-2">
                  Across {providersData?.providers?.length ?? 0} providers
                </p>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Server className="h-5 w-5" />
                  Available Providers
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-4xl font-bold">
                  {(providersData?.providers ?? []).filter(p => p.is_available).length}
                </div>
                <p className="text-sm text-muted-foreground mt-2">
                  of {providersData?.providers?.length ?? 0} total
                </p>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <CheckCircle className="h-5 w-5" />
                  Service Health
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="flex items-center gap-2">
                  {(providersData?.providers ?? []).some(p => p.is_available) ? (
                    <>
                      <CheckCircle className="h-8 w-8 text-green-500" />
                      <span className="text-2xl font-bold">Healthy</span>
                    </>
                  ) : (
                    <>
                      <XCircle className="h-8 w-8 text-red-500" />
                      <span className="text-2xl font-bold">Unavailable</span>
                    </>
                  )}
                </div>
                <p className="text-sm text-muted-foreground mt-2">
                  Provider system status
                </p>
              </CardContent>
            </Card>
          </div>
        </TabsContent>
      </Tabs>
    </div>
  );
}
