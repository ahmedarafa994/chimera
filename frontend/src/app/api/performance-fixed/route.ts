// Enhanced Web Vitals Implementation for Chimera Frontend
// This replaces the problematic /api/performance/route.ts

import { NextRequest, NextResponse } from 'next/server';

// Single POST endpoint for all performance metrics
export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { sessionId, metricType, metric } = body;

    // Validate required fields
    if (!sessionId || !metricType || !metric) {
      return NextResponse.json(
        { error: 'Missing required fields: sessionId, metricType, metric' },
        { status: 400 }
      );
    }

    // Log metric in development
    if (process.env.NODE_ENV === 'development') {
      console.log(`[${metricType.toUpperCase()}] ${metric.name}: ${metric.value}${metric.unit || 'ms'} (${metric.rating})`);

      // Warn about poor performance
      if (metric.rating === 'poor') {
        console.warn(`⚠️ Poor ${metric.name}: ${metric.value}${metric.unit || 'ms'} - Threshold exceeded!`);
      }
    }

    // Prepare analytics payload
    const analyticsData = {
      sessionId,
      metricType,
      metric: {
        ...metric,
        timestamp: new Date().toISOString(),
        url: metric.url || request.headers.get('referer'),
        userAgent: request.headers.get('user-agent'),
        ip: request.headers.get('x-forwarded-for') ||
            request.headers.get('x-real-ip') ||
            'unknown',
        connectionType: metric.connectionType || 'unknown',
        deviceMemory: metric.deviceMemory,
        hardwareConcurrency: metric.hardwareConcurrency,
        viewport: metric.viewport
      },
      environment: process.env.NODE_ENV
    };

    // Send to analytics service
    await sendToAnalyticsService(analyticsData);

    // Store critical metrics in database if needed
    if (['core_web_vitals', 'user_journey', 'error'].includes(metricType)) {
      await storeMetricInDatabase(analyticsData);
    }

    return NextResponse.json({
      success: true,
      timestamp: new Date().toISOString(),
      metricId: generateMetricId(sessionId, metricType, metric.name)
    });

  } catch (error) {
    console.error('Error processing performance metric:', error);
    return NextResponse.json(
      {
        error: 'Failed to process metric',
        timestamp: new Date().toISOString()
      },
      { status: 500 }
    );
  }
}

// GET endpoint for retrieving performance data (for debugging)
export async function GET(request: NextRequest) {
  const { searchParams } = new URL(request.url);
  const sessionId = searchParams.get('sessionId');
  const metricType = searchParams.get('metricType');

  if (!sessionId) {
    return NextResponse.json(
      { error: 'sessionId is required' },
      { status: 400 }
    );
  }

  try {
    // In production, this would fetch from your database
    // For now, return stored metrics from in-memory cache or localStorage equivalent
    const metrics = await getStoredMetrics(sessionId, metricType);

    return NextResponse.json({
      success: true,
      sessionId,
      metricType,
      metrics,
      timestamp: new Date().toISOString()
    });

  } catch (error) {
    console.error('Error retrieving performance metrics:', error);
    return NextResponse.json(
      { error: 'Failed to retrieve metrics' },
      { status: 500 }
    );
  }
}

// Helper function to send metrics to backend analytics service
async function sendToAnalyticsService(data: any): Promise<void> {
  const analyticsEndpoint = process.env.BACKEND_URL
    ? `${process.env.BACKEND_URL}/api/v1/analytics/frontend`
    : process.env.ANALYTICS_ENDPOINT || 'http://localhost:8001/api/v1/analytics/frontend';

  try {
    const response = await fetch(analyticsEndpoint, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${process.env.ANALYTICS_API_KEY || process.env.CHIMERA_API_KEY}`,
        'X-Source': 'chimera-frontend',
        'X-Environment': process.env.NODE_ENV || 'development'
      },
      body: JSON.stringify({
        eventType: 'frontend_performance',
        source: 'frontend',
        data,
        timestamp: new Date().toISOString()
      })
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`Analytics API error (${response.status}): ${errorText}`);
    }

    // Log successful transmission in development
    if (process.env.NODE_ENV === 'development') {
      console.log(`📊 Sent ${data.metricType} metric to analytics service`);
    }

  } catch (error) {
    console.error('Failed to send metrics to analytics service:', error);

    // Fallback: Store in local fallback system
    await storeFallbackMetric(data);
  }
}

// Helper function to store critical metrics in database
async function storeMetricInDatabase(data: any): Promise<void> {
  // This would integrate with your database
  // For now, we'll log critical metrics for manual analysis

  if (data.metricType === 'core_web_vitals' && data.metric.rating === 'poor') {
    console.error(`🚨 CRITICAL PERFORMANCE ISSUE:`, {
      metric: data.metric.name,
      value: data.metric.value,
      threshold: getThresholdForMetric(data.metric.name),
      url: data.metric.url,
      userAgent: data.metric.userAgent?.slice(0, 100) + '...',
      timestamp: data.timestamp
    });
  }

  if (data.metricType === 'user_journey') {
    console.log(`🗺️ User Journey Completed:`, {
      route: data.metric.routeName,
      duration: data.metric.totalDuration,
      interactions: data.metric.interactionCount,
      apiCalls: data.metric.apiCallCount,
      errors: data.metric.errorCount
    });
  }

  // TODO: Implement actual database storage
  // await db.metrics.create({ data });
}

// Helper function to store metrics in fallback system when analytics service fails
async function storeFallbackMetric(data: any): Promise<void> {
  try {
    // In a real implementation, this might write to a local queue or file
    console.warn('📝 Storing metric in fallback system:', {
      type: data.metricType,
      metric: data.metric.name,
      value: data.metric.value,
      timestamp: data.timestamp
    });

    // Could implement local storage, file system, or memory queue
    // For now, just ensure the metric isn't lost completely

  } catch (fallbackError) {
    console.error('Even fallback metric storage failed:', fallbackError);
  }
}

// Helper function to generate unique metric IDs
function generateMetricId(sessionId: string, metricType: string, metricName: string): string {
  return `${sessionId}_${metricType}_${metricName}_${Date.now()}`;
}

// Helper function to get performance thresholds
function getThresholdForMetric(metricName: string): string {
  const thresholds: Record<string, string> = {
    'LCP': '2.5s (good), 4.0s (poor)',
    'FID': '100ms (good), 300ms (poor)',
    'CLS': '0.1 (good), 0.25 (poor)',
    'FCP': '1.8s (good), 3.0s (poor)',
    'TTI': '3.8s (good), 7.3s (poor)',
    'TBT': '200ms (good), 600ms (poor)',
    'INP': '200ms (good), 500ms (poor)'
  };

  return thresholds[metricName] || 'Unknown threshold';
}

// Helper function to retrieve stored metrics (for GET endpoint)
async function getStoredMetrics(sessionId: string, metricType?: string | null): Promise<any[]> {
  // This would fetch from your actual storage
  // For now, return empty array or mock data

  if (process.env.NODE_ENV === 'development') {
    return [
      {
        name: 'LCP',
        value: 2100,
        rating: 'good',
        timestamp: new Date().toISOString()
      },
      // Add more mock metrics as needed
    ];
  }

  return [];
}

// Export types for frontend usage
export interface PerformanceMetricRequest {
  sessionId: string;
  metricType: 'core_web_vitals' | 'component_performance' | 'api_timing' | 'user_journey' | 'resource_timing' | 'error';
  metric: {
    name: string;
    value: number;
    rating: 'good' | 'needs-improvement' | 'poor';
    unit?: string;
    url?: string;
    [key: string]: any;
  };
}

export interface PerformanceMetricResponse {
  success: boolean;
  timestamp: string;
  metricId?: string;
  error?: string;
}