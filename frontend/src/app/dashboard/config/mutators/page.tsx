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
} from "@/components/ui/form";
import { Switch } from "@/components/ui/switch";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { toast } from "sonner";
import { Separator } from "@/components/ui/separator";

const mutatorsSchema = z.object({
  crossover: z.boolean(),
  expand: z.boolean(),
  rephrase: z.boolean(),
  shorten: z.boolean(),
  generateSimilar: z.boolean(),

  // Advanced settings (mocked)
  temperature: z.number().min(0).max(2),
  maxTokens: z.number().min(1),
});

type MutatorValues = z.infer<typeof mutatorsSchema>;

export default function MutatorsConfigPage() {
  const form = useForm<MutatorValues>({
    resolver: zodResolver(mutatorsSchema),
    defaultValues: {
      crossover: true,
      expand: true,
      rephrase: true,
      shorten: true,
      generateSimilar: true,
      temperature: 1.0,
      maxTokens: 256,
    },
  });

  function onSubmit(_values: MutatorValues) {
    // Save configuration (values are logged in development)
    toast.success("Mutator Configuration Saved", {
      description: "Active mutators have been updated.",
    });
  }

  return (
    <div className="space-y-6">
      <div>
        <h3 className="text-lg font-medium">Mutator Strategies</h3>
        <p className="text-sm text-muted-foreground">
          Enable or disable specific mutation operators used during the fuzzing process.
        </p>
      </div>
      <Separator />
      <Form {...form}>
        <form onSubmit={form.handleSubmit(onSubmit)} className="space-y-8">
          <Card>
            <CardHeader>
              <CardTitle>Active Mutators</CardTitle>
              <CardDescription>Select which strategies to employ.</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <FormField
                control={form.control}
                name="crossover"
                render={({ field }) => (
                  <FormItem className="flex flex-row items-center justify-between rounded-lg border p-4">
                    <div className="space-y-0.5">
                      <FormLabel className="text-base">Crossover</FormLabel>
                      <FormDescription>
                        Combines two existing seeds to create a new one.
                      </FormDescription>
                    </div>
                    <FormControl>
                      <Switch checked={field.value} onCheckedChange={field.onChange} />
                    </FormControl>
                  </FormItem>
                )}
              />
              <FormField
                control={form.control}
                name="expand"
                render={({ field }) => (
                  <FormItem className="flex flex-row items-center justify-between rounded-lg border p-4">
                    <div className="space-y-0.5">
                      <FormLabel className="text-base">Expand</FormLabel>
                      <FormDescription>
                        Adds detail and length to the prompt.
                      </FormDescription>
                    </div>
                    <FormControl>
                      <Switch checked={field.value} onCheckedChange={field.onChange} />
                    </FormControl>
                  </FormItem>
                )}
              />
              <FormField
                control={form.control}
                name="rephrase"
                render={({ field }) => (
                  <FormItem className="flex flex-row items-center justify-between rounded-lg border p-4">
                    <div className="space-y-0.5">
                      <FormLabel className="text-base">Rephrase</FormLabel>
                      <FormDescription>
                        Rewrites the prompt with different wording but same meaning.
                      </FormDescription>
                    </div>
                    <FormControl>
                      <Switch checked={field.value} onCheckedChange={field.onChange} />
                    </FormControl>
                  </FormItem>
                )}
              />
              {/* Add others similarly if needed */}
            </CardContent>
          </Card>
          <Button type="submit">Save Changes</Button>
        </form>
      </Form>
    </div>
  );
}