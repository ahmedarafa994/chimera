// Simplified ControlPanel component for build fix
'use client';

import React from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Play, Pause, StopCircle } from 'lucide-react';
import { Session } from '@/types/deepteam';

interface ControlPanelProps {
  className?: string;
  session?: Session;
  onStop?: () => void;
  onPause?: () => void;
  onResume?: () => void;
}

export function ControlPanel({ className, session, onStop, onPause, onResume }: ControlPanelProps) {
  const isRunning = session?.status === 'running';
  const isPaused = session?.status === 'paused';

  return (
    <Card className={className}>
      <CardHeader>
        <CardTitle>Control Panel</CardTitle>
        <CardDescription>
          {session ? `Session: ${session.sessionId}` : 'DeepTeam execution controls'}
        </CardDescription>
      </CardHeader>
      <CardContent>
        <div className="flex gap-2">
          {isPaused && (
            <Button size="sm" variant="outline" onClick={onResume}>
              <Play className="h-4 w-4 mr-2" />
              Resume
            </Button>
          )}
          {isRunning && !isPaused && (
            <Button size="sm" variant="outline" onClick={onPause}>
              <Pause className="h-4 w-4 mr-2" />
              Pause
            </Button>
          )}
          {session && (
            <Button size="sm" variant="outline" onClick={onStop}>
              <StopCircle className="h-4 w-4 mr-2" />
              Stop
            </Button>
          )}
        </div>
      </CardContent>
    </Card>
  );
}

export default ControlPanel;