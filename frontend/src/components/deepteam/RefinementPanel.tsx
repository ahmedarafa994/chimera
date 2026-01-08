// Simplified RefinementPanel component for build fix
'use client';

import React from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";

interface RefinementPanelProps {
  className?: string;
  sessionId?: string;
}

export function RefinementPanel({ className, sessionId }: RefinementPanelProps) {
  return (
    <Card className={className}>
      <CardHeader>
        <CardTitle>Refinement Panel</CardTitle>
        <CardDescription>
          {sessionId ? `Strategy refinement for session: ${sessionId}` : 'Strategy refinement and optimization'}
        </CardDescription>
      </CardHeader>
      <CardContent>
        <div className="flex items-center justify-center h-32 text-muted-foreground">
          <p>{sessionId ? 'Refinement controls will be displayed here' : 'No session selected'}</p>
        </div>
      </CardContent>
    </Card>
  );
}

export default RefinementPanel;