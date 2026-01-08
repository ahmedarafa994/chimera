"use client";

/**
 * Activity Feed Dashboard Page
 * 
 * Real-time activity monitoring page that displays live attack/generation events
 * with WebSocket updates. Supports filtering, search, and statistics.
 */

import React from "react";
import { ActivityFeed } from "@/components/activity-feed";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { 
  Activity as ActivityIcon,
  RefreshCw,
  Download,
  Settings2,
  Bell,
  BellOff
} from "lucide-react";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";

export default function ActivityPage() {
  const [notificationsEnabled, setNotificationsEnabled] = React.useState(true);
  const [isExporting, setIsExporting] = React.useState(false);

  const handleExport = async () => {
    setIsExporting(true);
    // Simulate export delay
    await new Promise(resolve => setTimeout(resolve, 1000));
    
    // In a real implementation, this would export the activity log
    const exportData = {
      exportedAt: new Date().toISOString(),
      message: "Activity log export - implement actual export logic"
    };
    
    const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `activity-log-${new Date().toISOString().split("T")[0]}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    
    setIsExporting(false);
  };

  return (
    <TooltipProvider>
      <div className="flex flex-col gap-6 p-6">
        {/* Page Header */}
        <div className="flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
          <div className="space-y-1">
            <div className="flex items-center gap-2">
              <ActivityIcon className="h-6 w-6 text-primary" />
              <h1 className="text-2xl font-bold tracking-tight">Activity Feed</h1>
              <Badge variant="secondary" className="ml-2">
                Live
              </Badge>
            </div>
            <p className="text-muted-foreground">
              Monitor real-time attack and generation events across all operations
            </p>
          </div>
          
          <div className="flex items-center gap-2">
            <Tooltip>
              <TooltipTrigger asChild>
                <Button
                  variant="outline"
                  size="icon"
                  onClick={() => setNotificationsEnabled(!notificationsEnabled)}
                >
                  {notificationsEnabled ? (
                    <Bell className="h-4 w-4" />
                  ) : (
                    <BellOff className="h-4 w-4 text-muted-foreground" />
                  )}
                </Button>
              </TooltipTrigger>
              <TooltipContent>
                {notificationsEnabled ? "Disable notifications" : "Enable notifications"}
              </TooltipContent>
            </Tooltip>
            
            <Tooltip>
              <TooltipTrigger asChild>
                <Button
                  variant="outline"
                  size="icon"
                  onClick={handleExport}
                  disabled={isExporting}
                >
                  {isExporting ? (
                    <RefreshCw className="h-4 w-4 animate-spin" />
                  ) : (
                    <Download className="h-4 w-4" />
                  )}
                </Button>
              </TooltipTrigger>
              <TooltipContent>Export activity log</TooltipContent>
            </Tooltip>
            
            <Tooltip>
              <TooltipTrigger asChild>
                <Button variant="outline" size="icon">
                  <Settings2 className="h-4 w-4" />
                </Button>
              </TooltipTrigger>
              <TooltipContent>Activity settings</TooltipContent>
            </Tooltip>
          </div>
        </div>

        {/* Main Activity Feed */}
        <div className="grid gap-6 lg:grid-cols-3">
          {/* Activity Feed - Takes 2 columns on large screens */}
          <div className="lg:col-span-2">
            <ActivityFeed
              maxEvents={100}
              showFilters={true}
              showStats={true}
              autoScroll={true}
              demoMode={true} // Enable demo mode for testing
              className="h-[calc(100vh-220px)]"
            />
          </div>

          {/* Sidebar with additional info */}
          <div className="space-y-6">
            {/* Quick Stats Card */}
            <Card>
              <CardHeader className="pb-3">
                <CardTitle className="text-sm font-medium">Quick Actions</CardTitle>
                <CardDescription>Common activity operations</CardDescription>
              </CardHeader>
              <CardContent className="space-y-2">
                <Button variant="outline" className="w-full justify-start" size="sm">
                  <ActivityIcon className="mr-2 h-4 w-4" />
                  View All Jailbreaks
                </Button>
                <Button variant="outline" className="w-full justify-start" size="sm">
                  <ActivityIcon className="mr-2 h-4 w-4" />
                  View AutoDAN Sessions
                </Button>
                <Button variant="outline" className="w-full justify-start" size="sm">
                  <ActivityIcon className="mr-2 h-4 w-4" />
                  View GPTFuzz Attacks
                </Button>
                <Button variant="outline" className="w-full justify-start" size="sm">
                  <ActivityIcon className="mr-2 h-4 w-4" />
                  View Transformations
                </Button>
              </CardContent>
            </Card>

            {/* Event Types Legend */}
            <Card>
              <CardHeader className="pb-3">
                <CardTitle className="text-sm font-medium">Event Types</CardTitle>
                <CardDescription>Activity event categories</CardDescription>
              </CardHeader>
              <CardContent className="space-y-3">
                <div className="flex items-center gap-2">
                  <div className="h-2 w-2 rounded-full bg-blue-500" />
                  <span className="text-sm">Jailbreak Operations</span>
                </div>
                <div className="flex items-center gap-2">
                  <div className="h-2 w-2 rounded-full bg-purple-500" />
                  <span className="text-sm">AutoDAN Sessions</span>
                </div>
                <div className="flex items-center gap-2">
                  <div className="h-2 w-2 rounded-full bg-orange-500" />
                  <span className="text-sm">GPTFuzz Attacks</span>
                </div>
                <div className="flex items-center gap-2">
                  <div className="h-2 w-2 rounded-full bg-cyan-500" />
                  <span className="text-sm">Transformations</span>
                </div>
                <div className="flex items-center gap-2">
                  <div className="h-2 w-2 rounded-full bg-green-500" />
                  <span className="text-sm">Model Operations</span>
                </div>
                <div className="flex items-center gap-2">
                  <div className="h-2 w-2 rounded-full bg-gray-500" />
                  <span className="text-sm">System Events</span>
                </div>
              </CardContent>
            </Card>

            {/* Status Legend */}
            <Card>
              <CardHeader className="pb-3">
                <CardTitle className="text-sm font-medium">Status Indicators</CardTitle>
                <CardDescription>Event status meanings</CardDescription>
              </CardHeader>
              <CardContent className="space-y-3">
                <div className="flex items-center gap-2">
                  <Badge variant="outline" className="bg-green-500/10 text-green-500 border-green-500/20">
                    Success
                  </Badge>
                  <span className="text-sm text-muted-foreground">Operation completed</span>
                </div>
                <div className="flex items-center gap-2">
                  <Badge variant="outline" className="bg-red-500/10 text-red-500 border-red-500/20">
                    Failed
                  </Badge>
                  <span className="text-sm text-muted-foreground">Operation failed</span>
                </div>
                <div className="flex items-center gap-2">
                  <Badge variant="outline" className="bg-blue-500/10 text-blue-500 border-blue-500/20">
                    Running
                  </Badge>
                  <span className="text-sm text-muted-foreground">In progress</span>
                </div>
                <div className="flex items-center gap-2">
                  <Badge variant="outline" className="bg-yellow-500/10 text-yellow-500 border-yellow-500/20">
                    Warning
                  </Badge>
                  <span className="text-sm text-muted-foreground">Needs attention</span>
                </div>
                <div className="flex items-center gap-2">
                  <Badge variant="outline" className="bg-gray-500/10 text-gray-500 border-gray-500/20">
                    Pending
                  </Badge>
                  <span className="text-sm text-muted-foreground">Waiting to start</span>
                </div>
              </CardContent>
            </Card>

            {/* Tips Card */}
            <Card>
              <CardHeader className="pb-3">
                <CardTitle className="text-sm font-medium">Tips</CardTitle>
              </CardHeader>
              <CardContent className="text-sm text-muted-foreground space-y-2">
                <p>• Use filters to focus on specific event types</p>
                <p>• Click on events to view detailed information</p>
                <p>• Enable auto-scroll to follow new events</p>
                <p>• Export logs for offline analysis</p>
              </CardContent>
            </Card>
          </div>
        </div>
      </div>
    </TooltipProvider>
  );
}