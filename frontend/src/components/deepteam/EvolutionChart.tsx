/**
 * EvolutionChart Component
 *
 * Visualizes the evolution of fitness scores across generations
 * using the genetic algorithm
 */

'use client'

import { useMemo, Suspense } from 'react'
import { RechartsComponents } from '@/lib/components/lazy-components'
import { Session } from '@/types/deepteam'
import { Card } from '@/components/ui/card'
import { Tabs, TabsList, TabsTrigger, TabsContent } from '@/components/ui/tabs'
import { Skeleton } from '@/components/ui/skeleton'

const {
  LineChart,
  Line,
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} = RechartsComponents

// Chart loading skeleton
const ChartSkeleton = () => (
  <div className="w-full h-[400px] flex items-center justify-center">
    <Skeleton className="w-full h-full" />
  </div>
)

// Custom tooltip types (moved outside component to prevent re-creation during render)
interface TooltipPayloadItem {
  value: number
  name: string
  color: string
}

interface CustomTooltipProps {
  active?: boolean
  payload?: TooltipPayloadItem[]
  label?: string | number
}

// Custom tooltip component (declared outside main component)
const CustomTooltip = ({ active, payload, label }: CustomTooltipProps) => {
  if (active && payload && payload.length) {
    return (
      <Card className="p-3 shadow-lg">
        <p className="font-semibold mb-2">Generation {label}</p>
        <div className="space-y-1">
          <p className="text-sm flex items-center justify-between gap-4">
            <span className="text-purple-600">Best Fitness:</span>
            <span className="font-medium">{payload[0].value.toFixed(3)}</span>
          </p>
          <p className="text-sm flex items-center justify-between gap-4">
            <span className="text-blue-600">Avg Fitness:</span>
            <span className="font-medium">{payload[1].value.toFixed(3)}</span>
          </p>
          {payload[2] && (
            <p className="text-sm flex items-center justify-between gap-4">
              <span className="text-gray-600">Difference:</span>
              <span className="font-medium">{payload[2].value.toFixed(3)}</span>
            </p>
          )}
        </div>
      </Card>
    )
  }
  return null
}

interface EvolutionChartProps {
  session: Session
}

export function EvolutionChart({ session }: EvolutionChartProps) {
  // Prepare chart data
  const chartData = useMemo(() => {
    const { averageFitness, bestFitnessPerGeneration } = session.statistics

    return averageFitness.map((avg, index) => ({
      generation: index + 1,
      averageFitness: avg,
      bestFitness: bestFitnessPerGeneration[index] || 0,
      difference: (bestFitnessPerGeneration[index] || 0) - avg,
    }))
  }, [session.statistics])

  // Calculate statistics
  const stats = useMemo(() => {
    if (chartData.length === 0) {
      return {
        improvementRate: 0,
        convergenceScore: 0,
        diversityScore: 0,
      }
    }

    // Improvement rate (first to last best fitness)
    const firstBest = chartData[0].bestFitness
    const lastBest = chartData[chartData.length - 1].bestFitness
    const improvementRate = firstBest > 0 ? ((lastBest - firstBest) / firstBest) * 100 : 0

    // Convergence score (how much best and average fitness converge)
    const lastDiff = chartData[chartData.length - 1].difference
    const convergenceScore = lastDiff < 0.1 ? 100 : Math.max(0, 100 - lastDiff * 100)

    // Diversity score (variance in average fitness)
    const avgFitnesses = chartData.map((d) => d.averageFitness)
    const mean = avgFitnesses.reduce((a, b) => a + b, 0) / avgFitnesses.length
    const variance =
      avgFitnesses.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / avgFitnesses.length
    const diversityScore = Math.min(100, variance * 100)

    return {
      improvementRate,
      convergenceScore,
      diversityScore,
    }
  }, [chartData])

  if (chartData.length === 0) {
    return (
      <div className="flex items-center justify-center py-12 text-muted-foreground">
        No evolution data available yet. Evolution will appear as the session progresses.
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {/* Statistics Cards */}
      <div className="grid grid-cols-3 gap-4">
        <Card className="p-4">
          <p className="text-sm text-muted-foreground mb-1">Improvement Rate</p>
          <p className="text-2xl font-bold text-green-600">
            {stats.improvementRate > 0 ? '+' : ''}
            {stats.improvementRate.toFixed(1)}%
          </p>
          <p className="text-xs text-muted-foreground mt-1">From first to last generation</p>
        </Card>

        <Card className="p-4">
          <p className="text-sm text-muted-foreground mb-1">Convergence Score</p>
          <p className="text-2xl font-bold text-blue-600">{stats.convergenceScore.toFixed(0)}%</p>
          <p className="text-xs text-muted-foreground mt-1">
            Best and average fitness alignment
          </p>
        </Card>

        <Card className="p-4">
          <p className="text-sm text-muted-foreground mb-1">Population Diversity</p>
          <p className="text-2xl font-bold text-purple-600">{stats.diversityScore.toFixed(0)}%</p>
          <p className="text-xs text-muted-foreground mt-1">Genetic variation in population</p>
        </Card>
      </div>

      {/* Charts */}
      <Tabs defaultValue="line">
        <TabsList>
          <TabsTrigger value="line">Line Chart</TabsTrigger>
          <TabsTrigger value="area">Area Chart</TabsTrigger>
        </TabsList>

        <TabsContent value="line" className="mt-4">
          <Suspense fallback={<ChartSkeleton />}>
            <ResponsiveContainer width="100%" height={400}>
              <LineChart data={chartData} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
                <XAxis
                  dataKey="generation"
                  label={{ value: 'Generation', position: 'insideBottom', offset: -5 }}
                />
                <YAxis label={{ value: 'Fitness', angle: -90, position: 'insideLeft' }} />
                <Tooltip content={<CustomTooltip />} />
                <Legend />
                <Line
                  type="monotone"
                  dataKey="bestFitness"
                  stroke="#9333ea"
                  strokeWidth={2}
                  name="Best Fitness"
                  dot={{ fill: '#9333ea', r: 3 }}
                  activeDot={{ r: 6 }}
                />
                <Line
                  type="monotone"
                  dataKey="averageFitness"
                  stroke="#3b82f6"
                  strokeWidth={2}
                  name="Average Fitness"
                  dot={{ fill: '#3b82f6', r: 3 }}
                  activeDot={{ r: 6 }}
                />
              </LineChart>
            </ResponsiveContainer>
          </Suspense>
        </TabsContent>

        <TabsContent value="area" className="mt-4">
          <Suspense fallback={<ChartSkeleton />}>
            <ResponsiveContainer width="100%" height={400}>
              <AreaChart data={chartData} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
                <defs>
                  <linearGradient id="colorBest" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#9333ea" stopOpacity={0.8} />
                    <stop offset="95%" stopColor="#9333ea" stopOpacity={0} />
                  </linearGradient>
                  <linearGradient id="colorAvg" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.8} />
                    <stop offset="95%" stopColor="#3b82f6" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
                <XAxis
                  dataKey="generation"
                  label={{ value: 'Generation', position: 'insideBottom', offset: -5 }}
                />
                <YAxis label={{ value: 'Fitness', angle: -90, position: 'insideLeft' }} />
                <Tooltip content={<CustomTooltip />} />
                <Legend />
                <Area
                  type="monotone"
                  dataKey="bestFitness"
                  stroke="#9333ea"
                  fillOpacity={1}
                  fill="url(#colorBest)"
                  name="Best Fitness"
                />
                <Area
                  type="monotone"
                  dataKey="averageFitness"
                  stroke="#3b82f6"
                  fillOpacity={1}
                  fill="url(#colorAvg)"
                  name="Average Fitness"
                />
              </AreaChart>
            </ResponsiveContainer>
          </Suspense>
        </TabsContent>
      </Tabs>

      {/* Current Generation Info */}
      <Card className="p-4 bg-muted/50">
        <div className="flex items-center justify-between">
          <div>
            <p className="text-sm font-medium">Current Generation</p>
            <p className="text-2xl font-bold">{session.statistics.totalGenerations}</p>
          </div>
          <div>
            <p className="text-sm font-medium">Total Evaluations</p>
            <p className="text-2xl font-bold">{session.statistics.totalEvaluations}</p>
          </div>
          <div>
            <p className="text-sm font-medium">Successful Attacks</p>
            <p className="text-2xl font-bold text-green-600">
              {session.statistics.successfulAttacks}
            </p>
          </div>
          <div>
            <p className="text-sm font-medium">Success Rate</p>
            <p className="text-2xl font-bold text-blue-600">
              {(session.statistics.successRate * 100).toFixed(1)}%
            </p>
          </div>
        </div>
      </Card>
    </div>
  )
}
