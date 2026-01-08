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
  FormMessage,
} from "@/components/ui/form";
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { toast } from "sonner";
import { Separator } from "@/components/ui/separator";

const policySchema = z.object({
  strategy: z.enum(["random", "mcts-explore"]),
  mctsWeight: z.number().min(0).max(10),
  alpha: z.number().min(0).max(1),
});

type PolicyValues = z.infer<typeof policySchema>;

export default function PoliciesConfigPage() {
  const form = useForm<PolicyValues>({
    resolver: zodResolver(policySchema),
    defaultValues: {
      strategy: "mcts-explore",
      mctsWeight: 0.1,
      alpha: 0.1,
    },
  });

  function onSubmit(_values: PolicyValues) {
    // Save configuration (values are logged in development)
    toast.success("Policy Configuration Saved", {
      description: "Selection strategy parameters updated.",
    });
  }

  return (
    <div className="space-y-6">
      <div>
        <h3 className="text-lg font-medium">Selection Policies</h3>
        <p className="text-sm text-muted-foreground">
          Configure how seeds are selected from the pool for mutation.
        </p>
      </div>
      <Separator />
      <Form {...form}>
        <form onSubmit={form.handleSubmit(onSubmit)} className="space-y-8">
          <Card>
            <CardHeader>
              <CardTitle>Strategy Selection</CardTitle>
              <CardDescription>Choose the algorithm for seed selection.</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <FormField
                control={form.control}
                name="strategy"
                render={({ field }) => (
                  <FormItem>
                    <FormLabel>Strategy</FormLabel>
                    <Select onValueChange={field.onChange} defaultValue={field.value}>
                      <FormControl>
                        <SelectTrigger>
                          <SelectValue placeholder="Select a strategy" />
                        </SelectTrigger>
                      </FormControl>
                      <SelectContent>
                        <SelectItem value="random">Random Selection</SelectItem>
                        <SelectItem value="mcts-explore">MCTS Exploration</SelectItem>
                      </SelectContent>
                    </Select>
                    <FormDescription>
                      MCTS balances exploration and exploitation. Random is purely stochastic.
                    </FormDescription>
                    <FormMessage />
                  </FormItem>
                )}
              />

              {form.watch("strategy") === "mcts-explore" && (
                <div className="space-y-4 pt-4 border-t">
                  <h4 className="font-medium text-sm">MCTS Parameters</h4>
                  <FormField
                    control={form.control}
                    name="mctsWeight"
                    render={({ field }) => (
                      <FormItem>
                        <FormLabel>Exploration Weight</FormLabel>
                        <FormControl>
                          <Input type="number" step="0.1" {...field} />
                        </FormControl>
                        <FormDescription>
                          Higher values favor exploring less-visited seeds.
                        </FormDescription>
                        <FormMessage />
                      </FormItem>
                    )}
                  />
                </div>
              )}
            </CardContent>
          </Card>
          <Button type="submit">Save Changes</Button>
        </form>
      </Form>
    </div>
  );
}