'use client';

import { useEffect, useState } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { useWebVitals, VitalsMetric, VITALS_THRESHOLDS } from '@/lib/performance/web-vitals';

const METRIC_DESCRIPTIONS = {
  LCP: 'Largest Contentful Paint - How quickly the main content loads',
  FID: 'First Input Delay - How quickly the page responds to user input',
  CLS: 'Cumulative Layout Shift - How stable the page layout is',
  FCP: 'First Contentful Paint - How quickly content first appears',
  TTFB: 'Time to First Byte - How quickly the server responds',
};

const METRIC_UNITS = {
  LCP: 'ms',
  FID: 'ms',
  CLS: '',
  FCP: 'ms',
  TTFB: 'ms',
};

function getRatingColor(rating: VitalsMetric['rating']) {
  switch (rating) {
    case 'good': return 'bg-green-500';
    case 'needs-improvement': return 'bg-yellow-500';
    case 'poor': return 'bg-red-500';
    default: return 'bg-gray-500';
  }
}

function getRatingBadgeVariant(rating: VitalsMetric['rating']) {
  switch (rating) {
    case 'good': return 'default';
    case 'needs-improvement': return 'secondary';
    case 'poor': return 'destructive';
    default: return 'outline';
  }
}

function getProgressValue(metric: VitalsMetric) {
  const thresholds = VITALS_THRESHOLDS[metric.name];
  const { value } = metric;

  // For CLS, lower is better, so we invert the scale
  if (metric.name === 'CLS') {
    if (value <= thresholds.good) return 100;
    if (value <= thresholds.poor) return 50;
    return 25;
  }

  // For other metrics, we calculate percentage based on thresholds
  if (value <= thresholds.good) return 100;
  if (value <= thresholds.poor) return 75;
  return Math.max(25, 100 - ((value - thresholds.poor) / thresholds.poor * 50));
}

export default function CoreWebVitalsMonitor() {
  const { manager, getCurrentMetrics } = useWebVitals();
  const [metrics, setMetrics] = useState<VitalsMetric[]>([]);
  const [lastUpdate, setLastUpdate] = useState<Date>(new Date());

  useEffect(() => {
    // Initialize with current metrics
    setMetrics(getCurrentMetrics());

    // Listen for new metrics
    const handleMetricUpdate = (metric: VitalsMetric) => {
      setMetrics(prev => {
        const updated = prev.filter(m => m.name !== metric.name);
        return [...updated, metric];
      });
      setLastUpdate(new Date());
    };

    manager.onMetric(handleMetricUpdate);

    // Refresh current metrics periodically
    const interval = setInterval(() => {
      setMetrics(getCurrentMetrics());
    }, 2000);

    return () => {
      clearInterval(interval);
    };
  }, [manager, getCurrentMetrics]);

  const coreVitalsMetrics = metrics.filter(m => ['LCP', 'FID', 'CLS'].includes(m.name));
  const otherMetrics = metrics.filter(m => ['FCP', 'TTFB'].includes(m.name));

  const overallScore = coreVitalsMetrics.length > 0
    ? Math.round(coreVitalsMetrics.reduce((sum, metric) => {
        if (metric.rating === 'good') return sum + 100;
        if (metric.rating === 'needs-improvement') return sum + 75;
        return sum + 25;
      }, 0) / coreVitalsMetrics.length)
    : 0;

  const getScoreColor = (score: number) => {
    if (score >= 90) return 'text-green-600';
    if (score >= 75) return 'text-yellow-600';
    return 'text-red-600';
  };

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center justify-between">
            Core Web Vitals Monitor
            <Badge variant="outline" className="ml-2">
              Score: <span className={getScoreColor(overallScore)}>{overallScore}</span>
            </Badge>
          </CardTitle>
          <CardDescription>
            Real-time performance metrics tracking
            {lastUpdate && (
              <span className="block text-xs mt-1 text-muted-foreground">
                Last updated: {lastUpdate.toLocaleTimeString()}
              </span>
            )}
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid gap-4 md:grid-cols-3">
            {['LCP', 'FID', 'CLS'].map((metricName) => {
              const metric = metrics.find(m => m.name === metricName);
              const thresholds = VITALS_THRESHOLDS[metricName as keyof typeof VITALS_THRESHOLDS];

              return (
                <Card key={metricName}>
                  <CardContent className="pt-4">
                    <div className="flex items-center justify-between">
                      <div className="space-y-1">
                        <p className="text-sm font-medium">{metricName}</p>
                        <div className="flex items-center space-x-2">
                          {metric ? (
                            <>
                              <span className="text-2xl font-bold">
                                {metricName === 'CLS'
                                  ? metric.value.toFixed(3)
                                  : Math.round(metric.value)
                                }
                              </span>
                              <span className="text-xs text-muted-foreground">
                                {METRIC_UNITS[metricName as keyof typeof METRIC_UNITS]}
                              </span>
                              <Badge
                                variant={getRatingBadgeVariant(metric.rating)}
                                className="text-xs"
                              >
                                {metric.rating.replace('-', ' ')}
                              </Badge>
                            </>
                          ) : (
                            <span className="text-lg text-muted-foreground">--</span>
                          )}
                        </div>
                      </div>
                    </div>

                    {metric && (
                      <div className="mt-3 space-y-2">
                        <Progress
                          value={getProgressValue(metric)}
                          className="h-2"
                        />
                        <div className="flex justify-between text-xs text-muted-foreground">
                          <span>Good: ≤{thresholds.good}{METRIC_UNITS[metricName as keyof typeof METRIC_UNITS]}</span>
                          <span>Poor: &gt;{thresholds.poor}{METRIC_UNITS[metricName as keyof typeof METRIC_UNITS]}</span>
                        </div>
                      </div>
                    )}

                    <p className="text-xs text-muted-foreground mt-2">
                      {METRIC_DESCRIPTIONS[metricName as keyof typeof METRIC_DESCRIPTIONS]}
                    </p>
                  </CardContent>
                </Card>
              );
            })}
          </div>

          {otherMetrics.length > 0 && (
            <div className="mt-6">
              <h3 className="text-lg font-semibold mb-3">Additional Metrics</h3>
              <div className="grid gap-4 md:grid-cols-2">
                {otherMetrics.map((metric) => {
                  const thresholds = VITALS_THRESHOLDS[metric.name];

                  return (
                    <div key={metric.name} className="flex items-center justify-between p-3 border rounded-lg">
                      <div>
                        <p className="font-medium">{metric.name}</p>
                        <p className="text-xs text-muted-foreground">
                          {METRIC_DESCRIPTIONS[metric.name]}
                        </p>
                      </div>
                      <div className="text-right">
                        <div className="flex items-center space-x-2">
                          <span className="text-lg font-bold">
                            {Math.round(metric.value)}
                          </span>
                          <span className="text-xs text-muted-foreground">
                            {METRIC_UNITS[metric.name]}
                          </span>
                        </div>
                        <Badge
                          variant={getRatingBadgeVariant(metric.rating)}
                          className="text-xs mt-1"
                        >
                          {metric.rating.replace('-', ' ')}
                        </Badge>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          {metrics.length === 0 && (
            <div className="text-center py-8 text-muted-foreground">
              <p>Loading performance metrics...</p>
              <p className="text-xs mt-1">Metrics will appear as the page loads and you interact with it</p>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}