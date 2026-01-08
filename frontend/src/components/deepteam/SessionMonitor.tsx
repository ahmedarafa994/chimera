// Simplified SessionMonitor component for build fix
'use client';

import React from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Agent, Session } from '@/types/deepteam';

interface SessionMonitorProps {
  className?: string;
  session?: Session;
  agents?: Agent[];
}

export function SessionMonitor({ className, session, agents }: SessionMonitorProps) {
  return (
    <Card className={className}>
      <CardHeader>
        <CardTitle>Session Monitor</CardTitle>
        <CardDescription>
          {session ? `Monitoring session: ${session.sessionId}` : 'No active session'}
        </CardDescription>
      </CardHeader>
      <CardContent>
        {session ? (
          <div className="space-y-4">
            <div>
              <p className="text-sm text-muted-foreground">Status: <span className="font-medium">{session.status}</span></p>
            </div>
            {agents && agents.length > 0 && (
              <div>
                <p className="text-sm font-medium mb-2">Agents ({agents.length}):</p>
                <div className="space-y-1">
                  {agents.map((agent) => (
                    <div key={agent.id} className="flex justify-between text-sm">
                      <span>{agent.type}</span>
                      <span className="text-muted-foreground">{agent.status}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        ) : (
          <div className="flex items-center justify-center h-32 text-muted-foreground">
            <p>Session monitoring will be available when a session is active</p>
          </div>
        )}
      </CardContent>
    </Card>
  );
}

export default SessionMonitor;