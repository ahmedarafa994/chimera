/**
 * Deep Team + AutoDAN Main Dashboard
 *
 * This is the main dashboard page for managing multi-agent collaborative
 * red-teaming sessions with real-time monitoring and controls.
 */

'use client'

import { useEffect, useState } from 'react'
import { useRouter } from 'next/navigation'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert'
import {
  Activity,
  AlertTriangle,
  CheckCircle2,
  PlayCircle,
  TrendingUp,
  Users,
  Zap,
} from 'lucide-react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { toast } from 'sonner'

// Import custom components
import { AgentCard } from '@/components/deepteam/AgentCard'
import { AgentGraph } from '@/components/deepteam/AgentGraph'
import { EvolutionChart } from '@/components/deepteam/EvolutionChart'
import { SessionMonitor } from '@/components/deepteam/SessionMonitor'
import { ControlPanel } from '@/components/deepteam/ControlPanel'
import { EvaluationPanel } from '@/components/deepteam/EvaluationPanel'
import { RefinementPanel } from '@/components/deepteam/RefinementPanel'
import { ConfigurationDialog } from '@/components/deepteam/ConfigurationDialog'

// Import API client and types
import { deepTeamClient } from '@/lib/api/deepteam-client'
import {
  Session,
  SessionStatus,
} from '@/types/deepteam'
import { useWebSocket } from '@/lib/hooks/useWebSocket'

export default function DeepTeamDashboard() {
  const router = useRouter()
  const queryClient = useQueryClient()
  const [selectedSession, setSelectedSession] = useState<Session | null>(null)
  const [isConfigDialogOpen, setIsConfigDialogOpen] = useState(false)

  // Fetch active sessions
  const { data: sessions, isLoading: sessionsLoading } = useQuery({
    queryKey: ['sessions'],
    queryFn: deepTeamClient.listSessions,
    refetchInterval: 5000, // Refresh every 5 seconds
  })

  // Fetch agents for selected session
  const { data: agents } = useQuery({
    queryKey: ['agents', selectedSession?.sessionId],
    queryFn: async () => {
      if (!selectedSession) return [];
      const response = await deepTeamClient.listAgents(selectedSession.sessionId);
      return response.data || [];
    },
    enabled: !!selectedSession,
    refetchInterval: 2000, // Refresh every 2 seconds
  })

  // WebSocket connection for real-time updates
  const { isConnected, lastMessage } = useWebSocket(
    selectedSession ? `ws://localhost:8001/ws/${selectedSession.sessionId}` : 'ws://localhost:8001/ws/dummy',
    {
      reconnectAttempts: 5,
      reconnectInterval: 3000,
      heartbeatInterval: 30000,
    }
  )

  // Start session mutation
  const startSessionMutation = useMutation({
    mutationFn: deepTeamClient.createSession,
    onSuccess: (response) => {
      queryClient.invalidateQueries({ queryKey: ['sessions'] })
      setSelectedSession(response.data || null)
      toast.success('Session started successfully!')
    },
    onError: (error: Error) => {
      toast.error(`Failed to start session: ${error.message}`)
    },
  })

  // Stop session mutation
  const stopSessionMutation = useMutation({
    mutationFn: deepTeamClient.stopSession,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['sessions'] })
      toast.success('Session stopped')
    },
    onError: (error: Error) => {
      toast.error(`Failed to stop session: ${error.message}`)
    },
  })

  // Select most recent active session on load
  useEffect(() => {
    if (sessions?.data && sessions.data.length > 0 && !selectedSession) {
      const activeSession = sessions.data.find(
        (s) => s.status === SessionStatus.RUNNING || s.status === SessionStatus.INITIALIZING
      )
      setSelectedSession(activeSession || sessions.data[0])
    }
  }, [sessions, selectedSession])

  // Calculate dashboard statistics
  const statistics = {
    totalSessions: sessions?.data?.length || 0,
    activeSessions: sessions?.data?.filter((s) => s.status === SessionStatus.RUNNING).length || 0,
    successRate:
      selectedSession?.statistics.successRate !== undefined
        ? (selectedSession.statistics.successRate * 100).toFixed(1)
        : '0.0',
    bestFitness: selectedSession?.bestCandidate?.fitness.toFixed(3) || '0.000',
  }

  return (
    <div className="flex min-h-screen flex-col">
      {/* Header */}
      <header className="sticky top-0 z-50 border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
        <div className="container flex h-16 items-center justify-between">
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-2">
              <Zap className="h-6 w-6 text-primary" />
              <h1 className="text-2xl font-bold">Deep Team + AutoDAN</h1>
            </div>
            {isConnected && (
              <div className="flex items-center gap-2 text-sm text-green-600">
                <Activity className="h-4 w-4 animate-pulse" />
                <span>Live</span>
              </div>
            )}
          </div>

          <div className="flex items-center gap-2">
            <Button
              variant="outline"
              onClick={() => router.push('/dashboard/deepteam/authorization' as unknown as '/dashboard')}
            >
              <Users className="mr-2 h-4 w-4" />
              Authorization
            </Button>
            <Button onClick={() => setIsConfigDialogOpen(true)}>
              <PlayCircle className="mr-2 h-4 w-4" />
              New Session
            </Button>
          </div>
        </div>
      </header>

      {/* Safety Warning */}
      <Alert className="m-4 border-red-500 bg-red-50">
        <AlertTriangle className="h-4 w-4 text-red-600" />
        <AlertTitle className="text-red-600">Critical Safety Notice</AlertTitle>
        <AlertDescription className="text-red-600">
          This system is for AUTHORIZED security research ONLY. Ensure you have proper
          authorization, ethical approval, and valid research objectives before proceeding.
        </AlertDescription>
      </Alert>

      {/* Main Content */}
      <main className="container flex-1 py-6">
        {/* Statistics Cards */}
        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4 mb-6">
          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Total Sessions</CardTitle>
              <Activity className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{statistics.totalSessions}</div>
              <p className="text-xs text-muted-foreground">All-time sessions</p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Active Sessions</CardTitle>
              <PlayCircle className="h-4 w-4 text-green-600" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{statistics.activeSessions}</div>
              <p className="text-xs text-muted-foreground">Currently running</p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Success Rate</CardTitle>
              <TrendingUp className="h-4 w-4 text-blue-600" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{statistics.successRate}%</div>
              <p className="text-xs text-muted-foreground">Current session</p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Best Fitness</CardTitle>
              <CheckCircle2 className="h-4 w-4 text-purple-600" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{statistics.bestFitness}</div>
              <p className="text-xs text-muted-foreground">Highest score achieved</p>
            </CardContent>
          </Card>
        </div>

        {/* Main Dashboard Tabs */}
        <Tabs defaultValue="overview" className="space-y-4">
          <TabsList>
            <TabsTrigger value="overview">Overview</TabsTrigger>
            <TabsTrigger value="agents">Agents</TabsTrigger>
            <TabsTrigger value="evolution">Evolution</TabsTrigger>
            <TabsTrigger value="evaluation">Evaluation</TabsTrigger>
            <TabsTrigger value="refinement">Refinement</TabsTrigger>
          </TabsList>

          {/* Overview Tab */}
          <TabsContent value="overview" className="space-y-4">
            <div className="grid gap-4 md:grid-cols-2">
              {/* Session Monitor */}
              <Card className="md:col-span-2">
                <CardHeader>
                  <CardTitle>Session Monitor</CardTitle>
                  <CardDescription>
                    Real-time monitoring of the active collaborative red-teaming session
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  {selectedSession ? (
                    <SessionMonitor session={selectedSession} agents={agents || []} />
                  ) : (
                    <div className="flex items-center justify-center py-12 text-muted-foreground">
                      No active session. Start a new session to begin.
                    </div>
                  )}
                </CardContent>
              </Card>

              {/* Control Panel */}
              {selectedSession && (
                <Card>
                  <CardHeader>
                    <CardTitle>Control Panel</CardTitle>
                    <CardDescription>Manage session execution</CardDescription>
                  </CardHeader>
                  <CardContent>
                    <ControlPanel
                      session={selectedSession}
                      onStop={() => stopSessionMutation.mutate(selectedSession.sessionId)}
                      onPause={() => {
                        /* Implement pause */
                      }}
                      onResume={() => {
                        /* Implement resume */
                      }}
                    />
                  </CardContent>
                </Card>
              )}

              {/* Agent Graph */}
              {selectedSession && agents && (
                <Card>
                  <CardHeader>
                    <CardTitle>Agent Network</CardTitle>
                    <CardDescription>Multi-agent collaboration visualization</CardDescription>
                  </CardHeader>
                  <CardContent>
                    <AgentGraph agents={agents} />
                  </CardContent>
                </Card>
              )}
            </div>
          </TabsContent>

          {/* Agents Tab */}
          <TabsContent value="agents" className="space-y-4">
            <div className="grid gap-4 md:grid-cols-3">
              {agents?.map((agent) => (
                <AgentCard key={agent.id} agent={agent} />
              ))}

              {!agents || agents.length === 0 ? (
                <div className="col-span-3 flex items-center justify-center py-12 text-muted-foreground">
                  No agents active. Start a session to see agent status.
                </div>
              ) : null}
            </div>
          </TabsContent>

          {/* Evolution Tab */}
          <TabsContent value="evolution" className="space-y-4">
            <Card>
              <CardHeader>
                <CardTitle>Evolution Progress</CardTitle>
                <CardDescription>
                  Fitness evolution across generations with genetic algorithm optimization
                </CardDescription>
              </CardHeader>
              <CardContent>
                {selectedSession ? (
                  <EvolutionChart session={selectedSession} />
                ) : (
                  <div className="flex items-center justify-center py-12 text-muted-foreground">
                    No session data available
                  </div>
                )}
              </CardContent>
            </Card>
          </TabsContent>

          {/* Evaluation Tab */}
          <TabsContent value="evaluation" className="space-y-4">
            <Card>
              <CardHeader>
                <CardTitle>Evaluation Results</CardTitle>
                <CardDescription>
                  Multi-criteria evaluation from the Evaluator Agent
                </CardDescription>
              </CardHeader>
              <CardContent>
                {selectedSession ? (
                  <EvaluationPanel sessionId={selectedSession.sessionId} />
                ) : (
                  <div className="flex items-center justify-center py-12 text-muted-foreground">
                    No evaluation data available
                  </div>
                )}
              </CardContent>
            </Card>
          </TabsContent>

          {/* Refinement Tab */}
          <TabsContent value="refinement" className="space-y-4">
            <Card>
              <CardHeader>
                <CardTitle>Refinement Suggestions</CardTitle>
                <CardDescription>
                  Adaptive optimization recommendations from the Refiner Agent
                </CardDescription>
              </CardHeader>
              <CardContent>
                {selectedSession ? (
                  <RefinementPanel sessionId={selectedSession.sessionId} />
                ) : (
                  <div className="flex items-center justify-center py-12 text-muted-foreground">
                    No refinement data available
                  </div>
                )}
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </main>

      {/* Configuration Dialog */}
      <ConfigurationDialog
        open={isConfigDialogOpen}
        onOpenChange={setIsConfigDialogOpen}
        onSubmit={(config) => {
          startSessionMutation.mutate(config)
          setIsConfigDialogOpen(false)
        }}
      />
    </div>
  )
}
