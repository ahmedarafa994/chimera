import { GenerationPanel } from "@/components/generation-panel";

export default function GenerationPage() {
  return (
    <div className="space-y-6">
      <div className="flex flex-col gap-2">
        <h1 className="text-3xl font-bold tracking-tight">Generation Panel</h1>
        <p className="text-muted-foreground">
          Direct LLM interaction without transformation. Generate content from multiple providers.
        </p>
      </div>
      <div className="border-t pt-6">
        <GenerationPanel />
      </div>
    </div>
  );
}