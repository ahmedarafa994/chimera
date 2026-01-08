"use client";

import { useEffect, useState, Suspense } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { apiClient } from "@/lib/api-enhanced";
import { RechartsComponents } from "@/lib/components/lazy-components";

const {
  LineChart,
  Line,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} = RechartsComponents;

// Chart loading skeleton
const ChartSkeleton = () => (
  <div className="w-full h-[300px] flex items-center justify-center">
    <Skeleton className="w-full h-full" />
  </div>
);

interface MetricsData {
  timestamp: string;
  requests: number;
  latency: number;
  errors: number;
  activeConnections: number;
}

interface ProviderMetrics {
  name: string;
  requests: number;
  avgLatency: number;
  successRate: number;
  errors: number;
}

export default function SystemMetricsDashboard() {
  const [metrics, setMetrics] = useState<MetricsData[]>([]);
  const [providerMetrics, setProviderMetrics] = useState<ProviderMetrics[]>([]);
  const [loading, setLoading] = useState(true);

  const fetchMetrics = async () => {
    try {
      const metricsRes = await apiClient.get("/integration/stats");
      // Also fetching providers for potential future use
      await apiClient.get("/providers");

      setMetrics(metricsRes.data.history || []);
      setProviderMetrics(metricsRes.data.providers || []);
    } catch (error) {
      console.error("Failed to fetch metrics:", error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchMetrics();
    const interval = setInterval(fetchMetrics, 10000);
    return () => clearInterval(interval);
  }, []);

  if (loading) {
    return <div className="flex items-center justify-center h-screen">Loading metrics...</div>;
  }

  return (
    <div className="container mx-auto p-6 space-y-6">
      <h1 className="text-3xl font-bold">System Metrics & Performance</h1>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <Card>
          <CardHeader>
            <CardTitle>Total Requests</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-3xl font-bold">
              {metrics.reduce((sum, m) => sum + m.requests, 0)}
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Avg Latency</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-3xl font-bold">
              {metrics.length > 0 ? (metrics.reduce((sum, m) => sum + m.latency, 0) / metrics.length).toFixed(0) : 0}ms
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Error Rate</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-3xl font-bold">
              {metrics.length > 0 && metrics.reduce((sum, m) => sum + m.requests, 0) > 0
                ? ((metrics.reduce((sum, m) => sum + m.errors, 0) / metrics.reduce((sum, m) => sum + m.requests, 0)) * 100).toFixed(2)
                : 0}%
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Active Connections</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-3xl font-bold">
              {metrics[metrics.length - 1]?.activeConnections || 0}
            </p>
          </CardContent>
        </Card>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Request Volume Over Time</CardTitle>
        </CardHeader>
        <CardContent>
          <Suspense fallback={<ChartSkeleton />}>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={metrics}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="timestamp" />
                <YAxis />
                <Tooltip />
                <Legend />
                <Line type="monotone" dataKey="requests" stroke="#3b82f6" name="Requests" />
                <Line type="monotone" dataKey="errors" stroke="#ef4444" name="Errors" />
              </LineChart>
            </ResponsiveContainer>
          </Suspense>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Latency Trends</CardTitle>
        </CardHeader>
        <CardContent>
          <Suspense fallback={<ChartSkeleton />}>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={metrics}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="timestamp" />
                <YAxis />
                <Tooltip />
                <Legend />
                <Line type="monotone" dataKey="latency" stroke="#10b981" name="Latency (ms)" />
              </LineChart>
            </ResponsiveContainer>
          </Suspense>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Provider Performance</CardTitle>
        </CardHeader>
        <CardContent>
          <Suspense fallback={<ChartSkeleton />}>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={providerMetrics}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" />
                <YAxis />
                <Tooltip />
                <Legend />
                <Bar dataKey="requests" fill="#3b82f6" name="Requests" />
                <Bar dataKey="avgLatency" fill="#10b981" name="Avg Latency (ms)" />
                <Bar dataKey="errors" fill="#ef4444" name="Errors" />
              </BarChart>
            </ResponsiveContainer>
          </Suspense>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Provider Success Rates</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            {providerMetrics.map((provider) => (
              <div key={provider.name} className="flex items-center justify-between">
                <span className="font-semibold">{provider.name}</span>
                <div className="flex items-center gap-4">
                  <div className="w-64 bg-gray-200 rounded-full h-4">
                    <div
                      className="bg-green-500 h-4 rounded-full"
                      style={{ width: `${provider.successRate}%` }}
                    />
                  </div>
                  <span className="text-sm font-medium">{provider.successRate.toFixed(1)}%</span>
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
