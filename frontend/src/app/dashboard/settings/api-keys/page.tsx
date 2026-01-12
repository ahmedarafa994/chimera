"use client";

import { ApiKeyManager } from "@/components/api-keys";
import { useApiKeys } from "@/hooks";
import { Separator } from "@/components/ui/separator";

export default function ApiKeysPage() {
  const {
    keys,
    providers,
    isLoading,
    error,
    refresh,
    createKey,
    updateKey,
    deleteKey,
    testKey,
    testNewKey,
    activateKey,
    deactivateKey,
    revokeKey,
  } = useApiKeys({ autoRefresh: true, refreshInterval: 60000 });

  return (
    <div className="space-y-6">
      <div>
        <h3 className="text-lg font-medium">API Key Management</h3>
        <p className="text-sm text-muted-foreground">
          Securely manage your LLM provider API keys with encrypted storage and automatic failover.
        </p>
      </div>
      <Separator />

      <ApiKeyManager
        keys={keys}
        providers={providers}
        isLoading={isLoading}
        error={error}
        onCreateKey={createKey}
        onUpdateKey={updateKey}
        onDeleteKey={deleteKey}
        onTestKey={testKey}
        onTestNewKey={testNewKey}
        onActivateKey={activateKey}
        onDeactivateKey={deactivateKey}
        onRevokeKey={revokeKey}
        onRefresh={refresh}
      />
    </div>
  );
}
