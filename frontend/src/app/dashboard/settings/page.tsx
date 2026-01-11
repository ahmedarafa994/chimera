"use client";

import { LLMConfigForm } from "@/components/llm-config-form";
import { ConnectionConfig } from "@/components/connection-config";
import { Separator } from "@/components/ui/separator";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { AdminOnly } from "@/components/auth/RoleGuard";
import { Shield } from "lucide-react";
import { Badge } from "@/components/ui/badge";

export default function SettingsPage() {
  return (
    <AdminOnly>
      <div className="space-y-6">
        <div className="flex items-start justify-between">
          <div>
            <div className="flex items-center gap-3">
              <h3 className="text-lg font-medium">Platform Settings</h3>
              <Badge variant="outline" className="bg-rose-500/10 text-rose-400 border-rose-500/20 gap-1">
                <Shield className="h-3 w-3" />
                Admin Only
              </Badge>
            </div>
            <p className="text-sm text-muted-foreground">
              Manage your API connections and LLM provider configurations.
            </p>
          </div>
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
    </AdminOnly>
  );
}
