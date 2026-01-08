import { ExecutionPanel } from "@/components/execution-panel";

export default function ExecutionPage() {
  return (
    <div className="space-y-6">
      <div className="flex flex-col gap-2">
        <h1 className="text-3xl font-bold tracking-tight">Execution Panel</h1>
        <p className="text-muted-foreground">
          Transform prompts and execute them with LLM providers in a single operation.
        </p>
      </div>
      <div className="border-t pt-6">
        <ExecutionPanel />
      </div>
    </div>
  );
}