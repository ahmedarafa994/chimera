import { AutoDANTurboInterface } from "@/components/autodan-turbo";

export default function AutoDANTurboPage() {
  return (
    <div className="space-y-6">
      <div className="flex flex-col gap-2">
        <h1 className="text-3xl font-bold tracking-tight">AutoDAN-Turbo</h1>
        <p className="text-muted-foreground">
          Lifelong learning agent for automatic jailbreak strategy discovery and evolution.
          Based on the ICLR 2025 paper achieving 88.5% ASR on GPT-4-Turbo.
        </p>
      </div>
      <AutoDANTurboInterface />
    </div>
  );
}