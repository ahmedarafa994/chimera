// Simplified EvaluationPanel component for build fix
'use client';

import React from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";

interface EvaluationPanelProps {
  className?: string;
  sessionId?: string;
}

export function EvaluationPanel({ className, sessionId }: EvaluationPanelProps) {
  return (
    <Card className={className}>
      <CardHeader>
        <CardTitle>Evaluation Panel</CardTitle>
        <CardDescription>
          {sessionId ? `Evaluations for session: ${sessionId}` : 'Performance evaluation and metrics'}
        </CardDescription>
      </CardHeader>
      <CardContent>
        <div className="flex items-center justify-center h-32 text-muted-foreground">
          <p>{sessionId ? 'Evaluation data will be displayed here' : 'No session selected'}</p>
        </div>
      </CardContent>
    </Card>
  );
}

export default EvaluationPanel;