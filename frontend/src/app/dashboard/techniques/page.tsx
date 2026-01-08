import { TechniquesExplorer } from "@/components/techniques-explorer";

export default function TechniquesPage() {
  return (
    <div className="space-y-6">
      <div className="flex flex-col gap-2">
        <h1 className="text-3xl font-bold tracking-tight">Techniques Explorer</h1>
        <p className="text-muted-foreground">
          Browse all 40+ transformation techniques. View transformers, framers, and obfuscators.
        </p>
      </div>
      <div className="border-t pt-6">
        <TechniquesExplorer />
      </div>
    </div>
  );
}