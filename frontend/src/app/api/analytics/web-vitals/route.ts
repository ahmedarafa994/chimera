import { NextRequest, NextResponse } from 'next/server';

/**
 * Web Vitals Analytics Endpoint
 * Receives Core Web Vitals metrics from the frontend
 * In production, this would forward to an analytics service
 */
export async function POST(request: NextRequest) {
  try {
    const metric = await request.json();

    // In development, just log and return success
    if (process.env.NODE_ENV === 'development') {
      // Silently accept - avoid console spam
      return NextResponse.json({ success: true });
    }

    // In production, you could forward to an analytics service
    // Example: await fetch(process.env.ANALYTICS_ENDPOINT, { method: 'POST', body: JSON.stringify(metric) });

    console.log('[Web Vitals]', metric.name, metric.value);

    return NextResponse.json({ success: true });
  } catch (_error) {
    // Silently fail - analytics should not break the app
    return NextResponse.json({ success: false }, { status: 200 });
  }
}

// Health check endpoint
export async function GET() {
  return NextResponse.json({
    status: 'ok',
    message: 'Web Vitals analytics endpoint is active'
  });
}
