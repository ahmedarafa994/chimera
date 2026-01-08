"use client";

import { useQuery } from "@tanstack/react-query";
import enhancedApi from "@/lib/api-enhanced";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Badge } from "@/components/ui/badge";
import { Layers, AlertCircle } from "lucide-react";

export function TechniquesExplorer() {
  const { data, isLoading, error } = useQuery({
    queryKey: ["techniques"],
    queryFn: () => enhancedApi.techniques(),
  });

  if (isLoading) {
    return <div className="p-8 text-center text-muted-foreground">Loading techniques...</div>;
  }

  if (error || !data) {
    return (
      <div className="p-8 text-center text-destructive flex flex-col items-center gap-2">
        <AlertCircle className="h-8 w-8" />
        <p>Failed to load techniques</p>
      </div>
    );
  }

  const techniques = data.techniques;

  return (
    <div className="space-y-6">
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Total Suites</CardTitle>
            <Layers className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{techniques.length}</div>
            <p className="text-xs text-muted-foreground">Available presets</p>
          </CardContent>
        </Card>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Transformation Techniques</CardTitle>
          <CardDescription>
            Comprehensive library of prompt engineering, framing, and obfuscation strategies
          </CardDescription>
        </CardHeader>
        <CardContent>
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Suite Name</TableHead>
                <TableHead>Type</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {techniques.map((tech) => (
                <TableRow key={tech}>
                  <TableCell className="font-medium capitalize font-mono">
                    {tech.replace(/_/g, " ")}
                  </TableCell>
                  <TableCell>
                    <Badge variant="outline">Transformation Suite</Badge>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </CardContent>
      </Card>
    </div>
  );
}