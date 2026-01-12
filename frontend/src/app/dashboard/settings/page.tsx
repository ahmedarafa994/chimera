import Link from "next/link";
import { LLMConfigForm } from "@/components/llm-config-form";
import { ConnectionConfig } from "@/components/connection-config";
import { Separator } from "@/components/ui/separator";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Shield, Key, ArrowRight } from "lucide-react";

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

      {/* API Keys Quick Access Card */}
      <Card className="border-primary/20 bg-primary/5">
        <CardHeader className="pb-3">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Shield className="h-5 w-5 text-primary" />
              <CardTitle className="text-base">API Key Management</CardTitle>
            </div>
            <Link href="/dashboard/settings/api-keys">
              <Button variant="outline" size="sm">
                <Key className="h-4 w-4 mr-2" />
                Manage Keys
                <ArrowRight className="h-4 w-4 ml-2" />
              </Button>
            </Link>
          </div>
          <CardDescription>
            Securely manage your LLM provider API keys with encrypted storage, automatic failover, and usage tracking.
          </CardDescription>
        </CardHeader>
      </Card>

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