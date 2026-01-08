// Simplified AgentGraph component for build fix
'use client';

import React from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Agent } from '@/types/deepteam';

interface AgentGraphProps {
  agents?: Agent[];
  className?: string;
}

export function AgentGraph({ agents, className }: AgentGraphProps) {
  return (
    <Card className={className}>
      <CardHeader>
        <CardTitle>Agent Graph</CardTitle>
        <CardDescription>
          Agent interaction visualization
        </CardDescription>
      </CardHeader>
      <CardContent>
        {agents && agents.length > 0 ? (
          <div className="space-y-2">
            <p className="text-sm text-muted-foreground mb-3">Active Agents: {agents.length}</p>
            <div className="grid gap-2">
              {agents.map((agent) => (
                <div key={agent.id} className="flex items-center justify-between p-2 border rounded">
                  <span className="text-sm font-medium">{agent.type}</span>
                  <span className={`text-xs px-2 py-1 rounded ${
                    agent.status === 'working' ? 'bg-green-100 text-green-700' :
                    agent.status === 'idle' ? 'bg-yellow-100 text-yellow-700' :
                    agent.status === 'error' ? 'bg-red-100 text-red-700' :
                    'bg-gray-100 text-gray-700'
                  }`}>
                    {agent.status}
                  </span>
                </div>
              ))}
            </div>
          </div>
        ) : (
          <div className="flex items-center justify-center h-64 text-muted-foreground">
            <p>No agents available</p>
          </div>
        )}
      </CardContent>
    </Card>
  );
}

export default AgentGraph;