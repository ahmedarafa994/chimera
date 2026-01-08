/**
 * AgentCard Component
 *
 * Displays status and statistics for individual agents in the multi-agent system
 */

import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Progress } from '@/components/ui/progress'
import {
  Activity,
  Bot,
  CheckCircle2,
  Clock,
  Loader2,
  Target,
  TrendingUp,
  XCircle,
} from 'lucide-react'
import { Agent, AgentStatus, AgentType } from '@/types/deepteam'
import { cn } from '@/lib/utils'

interface AgentCardProps {
  agent: Agent
  className?: string
}

// Agent type configuration
const agentConfig = {
  [AgentType.ATTACKER]: {
    icon: Target,
    color: 'text-purple-600',
    bgColor: 'bg-purple-50',
    borderColor: 'border-purple-200',
    label: 'Attacker',
    description: 'AutoDAN genetic algorithm',
  },
  [AgentType.EVALUATOR]: {
    icon: Activity,
    color: 'text-cyan-600',
    bgColor: 'bg-cyan-50',
    borderColor: 'border-cyan-200',
    label: 'Evaluator',
    description: 'Multi-criteria judge',
  },
  [AgentType.REFINER]: {
    icon: TrendingUp,
    color: 'text-orange-600',
    bgColor: 'bg-orange-50',
    borderColor: 'border-orange-200',
    label: 'Refiner',
    description: 'Adaptive optimizer',
  },
}

// Status configuration
const statusConfig = {
  [AgentStatus.IDLE]: {
    icon: Clock,
    color: 'text-gray-500',
    badgeVariant: 'secondary' as const,
    label: 'Idle',
  },
  [AgentStatus.INITIALIZING]: {
    icon: Loader2,
    color: 'text-blue-500',
    badgeVariant: 'default' as const,
    label: 'Initializing',
    animated: true,
  },
  [AgentStatus.WORKING]: {
    icon: Activity,
    color: 'text-green-500',
    badgeVariant: 'default' as const,
    label: 'Working',
    animated: true,
  },
  [AgentStatus.WAITING]: {
    icon: Clock,
    color: 'text-yellow-500',
    badgeVariant: 'outline' as const,
    label: 'Waiting',
  },
  [AgentStatus.COMPLETED]: {
    icon: CheckCircle2,
    color: 'text-green-600',
    badgeVariant: 'default' as const,
    label: 'Completed',
  },
  [AgentStatus.ERROR]: {
    icon: XCircle,
    color: 'text-red-500',
    badgeVariant: 'destructive' as const,
    label: 'Error',
  },
}

export function AgentCard({ agent, className }: AgentCardProps) {
  const config = agentConfig[agent.type]
  const status = statusConfig[agent.status]

  const AgentIcon = config.icon
  const StatusIcon = status.icon

  return (
    <Card className={cn('overflow-hidden', config.borderColor, className)}>
      <CardHeader className={cn('pb-3', config.bgColor)}>
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <div className={cn('rounded-full p-2 bg-white shadow-sm')}>
              <AgentIcon className={cn('h-5 w-5', config.color)} />
            </div>
            <div>
              <CardTitle className="text-base">{config.label}</CardTitle>
              <p className="text-xs text-muted-foreground">{config.description}</p>
            </div>
          </div>

          <Badge variant={status.badgeVariant} className="flex items-center gap-1">
            <StatusIcon
              className={cn('h-3 w-3', status.color, (status as any).animated && 'animate-pulse')}
            />
            {status.label}
          </Badge>
        </div>
      </CardHeader>

      <CardContent className="pt-4">
        {/* Current Task */}
        {agent.currentTask && (
          <div className="mb-4">
            <p className="text-sm font-medium mb-1">Current Task</p>
            <p className="text-xs text-muted-foreground">{agent.currentTask}</p>

            {agent.progress !== undefined && (
              <div className="mt-2">
                <div className="flex items-center justify-between mb-1">
                  <span className="text-xs text-muted-foreground">Progress</span>
                  <span className="text-xs font-medium">{agent.progress}%</span>
                </div>
                <Progress value={agent.progress} className="h-2" />
              </div>
            )}
          </div>
        )}

        {/* Statistics */}
        {agent.statistics && (
          <div className="space-y-2">
            <div className="flex items-center justify-between text-sm">
              <span className="text-muted-foreground">Tasks Completed</span>
              <span className="font-medium">{agent.statistics.tasksCompleted}</span>
            </div>

            <div className="flex items-center justify-between text-sm">
              <span className="text-muted-foreground">Success Rate</span>
              <span
                className={cn(
                  'font-medium',
                  agent.statistics.successRate > 0.7
                    ? 'text-green-600'
                    : agent.statistics.successRate > 0.4
                      ? 'text-yellow-600'
                      : 'text-red-600'
                )}
              >
                {(agent.statistics.successRate * 100).toFixed(1)}%
              </span>
            </div>

            <div className="flex items-center justify-between text-sm">
              <span className="text-muted-foreground">Avg Processing Time</span>
              <span className="font-medium">
                {agent.statistics.averageProcessingTime.toFixed(2)}s
              </span>
            </div>

            {agent.statistics.errors > 0 && (
              <div className="flex items-center justify-between text-sm">
                <span className="text-muted-foreground">Errors</span>
                <span className="font-medium text-red-600">{agent.statistics.errors}</span>
              </div>
            )}
          </div>
        )}

        {/* Last Update */}
        <div className="mt-4 pt-4 border-t">
          <p className="text-xs text-muted-foreground">
            Last updated: {new Date(agent.lastUpdate).toLocaleTimeString()}
          </p>
        </div>
      </CardContent>
    </Card>
  )
}
