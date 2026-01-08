"use client";

import React from "react";
import { useQuery } from "@tanstack/react-query";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Button } from "@/components/ui/button";
import { useModelSelection } from "@/contexts/ModelSelectionContext";
import { Loader2, Check, Server, Cpu } from "lucide-react";
import { toast } from "sonner";
import enhancedApi from "@/lib/api-enhanced";

interface ProviderData {
  provider: string;
  status?: string;
  model?: string;
  available_models: string[];
}

export function ModelSelectorCompact() {
  const { selection, setSelection, isLoading: contextLoading } = useModelSelection();
  const [isSaving, setIsSaving] = React.useState(false);
  const [localProvider, setLocalProvider] = React.useState(selection.provider || "");
  const [localModel, setLocalModel] = React.useState(selection.model || "");

  const { data: providersData, isLoading: providersLoading } = useQuery({
    queryKey: ["providers"],
    queryFn: async () => {
      const response = await enhancedApi.providers.list();
      return response.data;
    },
  });

  // Sync local state with context
  React.useEffect(() => {
    if (selection.provider) setLocalProvider(selection.provider);
    if (selection.model) setLocalModel(selection.model);
  }, [selection]);

  const providers = (providersData?.providers || []) as ProviderData[];
  const selectedProviderData = providers.find((p: ProviderData) => p.provider === localProvider);
  const availableModels = selectedProviderData?.available_models || [];

  const handleProviderChange = (newProvider: string) => {
    setLocalProvider(newProvider);
    const providerData = providers.find((p: ProviderData) => p.provider === newProvider);
    if (providerData?.available_models && providerData.available_models.length > 0) {
      setLocalModel(providerData.available_models[0]);
    }
  };

  const handleSave = async () => {
    if (!localProvider || !localModel) {
      toast.error("Please select both provider and model");
      return;
    }

    setIsSaving(true);
    try {
      await setSelection(localProvider, localModel);
      toast.success("Model selection saved", {
        description: `Using ${localModel} from ${localProvider}`,
      });
    } catch {
      toast.error("Failed to save selection");
    } finally {
      setIsSaving(false);
    }
  };

  if (contextLoading || providersLoading) {
    return (
      <div className="flex items-center gap-2 text-sm text-muted-foreground">
        <Loader2 className="h-4 w-4 animate-spin" />
        <span>Loading models...</span>
      </div>
    );
  }

  return (
    <div className="flex items-center gap-3">
      <div className="flex items-center gap-2">
        <Server className="h-4 w-4 text-muted-foreground" />
        <Select value={localProvider} onValueChange={handleProviderChange}>
          <SelectTrigger className="w-[140px] h-9">
            <SelectValue placeholder="Provider" />
          </SelectTrigger>
          <SelectContent>
            {providers.map((provider: ProviderData) => (
              <SelectItem key={provider.provider} value={provider.provider}>
                {provider.provider}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>

      <div className="flex items-center gap-2">
        <Cpu className="h-4 w-4 text-muted-foreground" />
        <Select value={localModel} onValueChange={setLocalModel}>
          <SelectTrigger className="w-[200px] h-9">
            <SelectValue placeholder="Model" />
          </SelectTrigger>
          <SelectContent>
            {availableModels.map((model: string) => (
              <SelectItem key={model} value={model}>
                {model}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>

      <Button size="sm" onClick={handleSave} disabled={isSaving}>
        {isSaving ? <Loader2 className="h-4 w-4 animate-spin" /> : <Check className="h-4 w-4" />}
      </Button>
    </div>
  );
}
