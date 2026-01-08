import { LLMConfigForm } from "@/components/llm-config-form";
import { ConnectionConfig } from "@/components/connection-config";
import { Separator } from "@/components/ui/separator";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";

export default function SettingsPage() {
  return (
    <div className="space-y-6">
      <div>
        <h3 className="text-lg font-medium">Platform Settings</h3>
        <p className="text-sm text-muted-foreground">
          Manage your API connections and LLM provider configurations.
        </p>
      </div>
      <Separator />

      <Tabs defaultValue="connection" className="w-full">
        <TabsList className="grid w-full grid-cols-2 max-w-md">
          <TabsTrigger value="connection">API Connection</TabsTrigger>
          <TabsTrigger value="llm">LLM Config</TabsTrigger>
        </TabsList>

        <TabsContent value="connection" className="mt-6">
          <ConnectionConfig />
        </TabsContent>

        <TabsContent value="llm" className="mt-6">
          <LLMConfigForm />
        </TabsContent>
      </Tabs>
    </div>
  );
}