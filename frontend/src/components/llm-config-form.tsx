"use client";

import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import * as z from "zod";
import { Button } from "@/components/ui/button";
import {
  Form,
  FormControl,
  FormDescription,
  FormField,
  FormItem,
  FormLabel,
  FormMessage,
} from "@/components/ui/form";
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Provider } from "@/types/schemas";
import { toast } from "sonner";
import { useState, useEffect } from "react";
import { saveApiConfig, getApiConfig } from "@/lib/api-config";

// We'll update the schema to be looser for the general form,
// but we'll handle the specific DeepSeek logic in onSubmit
const llmConfigSchema = z.object({
  provider: z.nativeEnum(Provider),
  apiKey: z.string().optional().or(z.literal("")),
  model: z.string().min(1, "Model name is required"),
  endpoint: z.string().url().optional().or(z.literal("")),
  // DeepSeek specific field
  deepseekApiKey: z.string().optional(),
});

type LLMConfigValues = z.infer<typeof llmConfigSchema>;

export function LLMConfigForm() {
  const [isClient, setIsClient] = useState(false);
  const [showDeepSeekInput, setShowDeepSeekInput] = useState(false);

  useEffect(() => {
    setIsClient(true);
    // Initialize form with values from api-config
    const currentConfig = getApiConfig();
    if (currentConfig.aiProvider === "deepseek") {
      setShowDeepSeekInput(true);
    }
  }, []);

  const form = useForm<LLMConfigValues>({
    resolver: zodResolver(llmConfigSchema),
    defaultValues: {
      provider: Provider.GOOGLE, // Default to Google/Gemini as per request
      apiKey: "",
      model: "gemini-3-pro-preview",
      endpoint: "",
      deepseekApiKey: "",
    },
  });

  // Watch for provider changes to toggle DeepSeek input
  const selectedProvider = form.watch("provider");
  useEffect(() => {
    if (selectedProvider === Provider.DEEPSEEK) {
      setShowDeepSeekInput(true);
      form.setValue("model", "deepseek-chat");
    } else {
      setShowDeepSeekInput(false);
      // Reset to default model if switching away (optional UX improvement)
      if (selectedProvider === Provider.GOOGLE) {
        form.setValue("model", "gemini-3-pro-preview");
      }
    }
  }, [selectedProvider, form]);

  function onSubmit(values: LLMConfigValues) {
    // 1. Save general non-sensitive config to localStorage (existing logic)
    const nonSensitiveConfig = {
      provider: values.provider,
      model: values.model,
      endpoint: values.endpoint,
    };

    localStorage.setItem("chimera_llm_config", JSON.stringify(nonSensitiveConfig));

    // 2. Save to api-config (for the new provider-agnostic system)
    // Map the selected provider to the "aiProvider" type expected by api-config
    let aiProvider: "gemini" | "deepseek" = "gemini";
    if (values.provider === Provider.DEEPSEEK) {
      aiProvider = "deepseek";
    }

    // We only save what we can. Ideally API keys are in env vars,
    // but if the user provides one here, we might want to temporarily use it
    // (though the requirement says secure input, usually implies not persisting to local storage if possible,
    // or warning about it. The existing code warns about apiKey).

    // Update the global API config
    saveApiConfig({
      aiProvider: aiProvider,
      // If the user entered a DeepSeek key, we currently don't have a secure way to persist it
      // other than environment variables as per the existing pattern.
      // However, for the purpose of the session/context, we might need a way to pass it.
      // For now, we will follow the existing pattern: WARN if they try to set it here,
      // but strictly speaking, api-config.ts only loads keys from ENV.
      // If we want to support dynamic keys from UI, we would need to update api-config.ts to read from storage or memory.
      // The requirement said "secure input field for deepseekApiKey".
      // Let's assume for this refactor we strictly follow the "Env Var Only" rule for persistence,
      // but maybe allow in-memory override if we modify api-config.
    });

    // Show warning if API key was provided (General or DeepSeek)
    if (values.apiKey || values.deepseekApiKey) {
      toast.warning("Security Notice", {
        description: "API keys should be configured via environment variables (NEXT_PUBLIC_*_API_KEY) for security. Keys entered here are not saved to persistent storage.",
      });
    } else {
      toast.success("Configuration updated", {
        description: `Provider set to ${values.provider}`,
      });
    }
  }

  // Load from local storage on mount
  useEffect(() => {
    const saved = localStorage.getItem("chimera_llm_config");
    const apiConfig = getApiConfig();

    if (saved) {
      try {
        const parsed = JSON.parse(saved);
        form.reset({
          provider: parsed.provider || Provider.GOOGLE,
          model: parsed.model || "gemini-3-pro-preview",
          endpoint: parsed.endpoint || "",
          apiKey: "",
          deepseekApiKey: "",
        });
      } catch (e) {
        console.error("Failed to parse saved config", e);
      }
    } else {
      // Sync with api-config defaults if no local storage
      if (apiConfig.aiProvider === "deepseek") {
        form.setValue("provider", Provider.DEEPSEEK);
        form.setValue("model", "deepseek-chat");
      }
    }
  }, [form]);

  if (!isClient) return null;

  return (
    <Card className="w-full max-w-2xl mx-auto">
      <CardHeader>
        <CardTitle>LLM Configuration</CardTitle>
        <CardDescription>Configure your LLM provider settings for fuzzing sessions.</CardDescription>
      </CardHeader>
      <CardContent>
        <Form {...form}>
          <form onSubmit={form.handleSubmit(onSubmit)} className="space-y-6">
            <FormField
              control={form.control}
              name="provider"
              render={({ field }) => (
                <FormItem>
                  <FormLabel>Provider</FormLabel>
                  <Select onValueChange={field.onChange} defaultValue={field.value}>
                    <FormControl>
                      <SelectTrigger>
                        <SelectValue placeholder="Select a provider" />
                      </SelectTrigger>
                    </FormControl>
                    <SelectContent>
                      <SelectItem value={Provider.GOOGLE}>Google (Gemini)</SelectItem>
                      <SelectItem value={Provider.DEEPSEEK}>DeepSeek</SelectItem>
                      <SelectItem value={Provider.OPENAI}>OpenAI</SelectItem>
                      <SelectItem value={Provider.ANTHROPIC}>Anthropic</SelectItem>
                      <SelectItem value={Provider.XAI}>xAI</SelectItem>
                      <SelectItem value={Provider.CUSTOM}>Custom</SelectItem>
                    </SelectContent>
                  </Select>
                  <FormDescription>
                    Select the AI provider you want to use.
                  </FormDescription>
                  <FormMessage />
                </FormItem>
              )}
            />

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <FormField
                control={form.control}
                name="model"
                render={({ field }) => (
                  <FormItem>
                    <FormLabel>Model Name</FormLabel>
                    <FormControl>
                      <Input placeholder="e.g. gemini-3-pro-preview, deepseek-chat" {...field} />
                    </FormControl>
                    <FormMessage />
                  </FormItem>
                )}
              />

              {/* Conditional DeepSeek API Key Input */}
              {showDeepSeekInput && (
                <FormField
                  control={form.control}
                  name="deepseekApiKey"
                  render={({ field }) => (
                    <FormItem>
                      <FormLabel>DeepSeek API Key</FormLabel>
                      <FormControl>
                        <Input type="password" placeholder="sk-..." {...field} />
                      </FormControl>
                      <FormDescription>
                        Set NEXT_PUBLIC_DEEPSEEK_API_KEY in .env
                      </FormDescription>
                      <FormMessage />
                    </FormItem>
                  )}
                />
              )}

              {/* Standard API Key Input (for other providers) */}
              {!showDeepSeekInput && (
                <FormField
                  control={form.control}
                  name="apiKey"
                  render={({ field }) => (
                    <FormItem>
                      <FormLabel>API Key</FormLabel>
                      <FormControl>
                        <Input type="password" placeholder="sk-..." {...field} />
                      </FormControl>
                      <FormDescription>
                        Set via env vars for security.
                      </FormDescription>
                      <FormMessage />
                    </FormItem>
                  )}
                />
              )}
            </div>

            <FormField
              control={form.control}
              name="endpoint"
              render={({ field }) => (
                <FormItem>
                  <FormLabel>Custom Endpoint (Optional)</FormLabel>
                  <FormControl>
                    <Input placeholder="https://api.example.com/v1" {...field} />
                  </FormControl>
                  <FormDescription>
                    Override default API endpoint for proxies.
                  </FormDescription>
                  <FormMessage />
                </FormItem>
              )}
            />

            <Button type="submit">Save Configuration</Button>
          </form>
        </Form>
      </CardContent>
    </Card>
  );
}
