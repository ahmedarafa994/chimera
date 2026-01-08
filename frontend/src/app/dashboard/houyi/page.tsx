import { HouYiInterface } from "@/components/houyi/HouYiInterface";

export default function HouYiPage() {
  return (
    <div className="space-y-6">
      <div className="flex flex-col gap-2">
        <h1 className="text-3xl font-bold tracking-tight">HouYi Optimizer</h1>
        <p className="text-muted-foreground">
          Evolutionary prompt optimization using genetic algorithms to find the most effective jailbreak prompts.
        </p>
      </div>
      <HouYiInterface />
    </div>
  );
}