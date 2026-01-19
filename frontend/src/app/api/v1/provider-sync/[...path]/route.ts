/**
 * Provider Sync API Proxy Route
 *
 * Forwards requests from frontend to backend provider sync endpoints
 * This handles all /api/v1/provider-sync/* routes and forwards them to
 * the backend at http://localhost:8001/api/v1/provider-sync/*
 */

import { NextRequest, NextResponse } from 'next/server';

const BACKEND_URL = process.env.BACKEND_URL || 'http://localhost:8001';

export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ path: string[] }> }
) {
  const resolvedParams = await params;
  return forwardRequest(request, resolvedParams.path, 'GET');
}

export async function POST(
  request: NextRequest,
  { params }: { params: Promise<{ path: string[] }> }
) {
  const resolvedParams = await params;
  return forwardRequest(request, resolvedParams.path, 'POST');
}

export async function PUT(
  request: NextRequest,
  { params }: { params: Promise<{ path: string[] }> }
) {
  const resolvedParams = await params;
  return forwardRequest(request, resolvedParams.path, 'PUT');
}

export async function PATCH(
  request: NextRequest,
  { params }: { params: Promise<{ path: string[] }> }
) {
  const resolvedParams = await params;
  return forwardRequest(request, resolvedParams.path, 'PATCH');
}

export async function DELETE(
  request: NextRequest,
  { params }: { params: Promise<{ path: string[] }> }
) {
  const resolvedParams = await params;
  return forwardRequest(request, resolvedParams.path, 'DELETE');
}

async function forwardRequest(
  request: NextRequest,
  pathSegments: string[],
  method: string
) {
  try {
    // Construct the backend URL
    const path = pathSegments.join('/');
    const url = new URL(`${BACKEND_URL}/api/v1/provider-sync/${path}`);

    // Copy query parameters
    request.nextUrl.searchParams.forEach((value, key) => {
      url.searchParams.set(key, value);
    });

    // Prepare headers
    const headers: Record<string, string> = {
      'Content-Type': 'application/json',
    };

    // Copy relevant headers from the original request
    const headersToForward = [
      'authorization',
      'x-api-key',
      'x-request-id',
      'x-session-id',
      'user-agent',
    ];

    headersToForward.forEach(headerName => {
      const value = request.headers.get(headerName);
      if (value) {
        headers[headerName] = value;
      }
    });

    // Prepare request options
    const requestOptions: RequestInit = {
      method,
      headers,
    };

    // Add body for POST, PUT, PATCH requests
    if (['POST', 'PUT', 'PATCH'].includes(method)) {
      try {
        const body = await request.text();
        if (body) {
          requestOptions.body = body;
        }
      } catch (error) {
        console.warn('[ProviderSyncProxy] Failed to read request body:', error);
      }
    }

    // Forward the request to backend
    console.log(`[ProviderSyncProxy] Forwarding ${method} request to:`, url.toString());

    const response = await fetch(url.toString(), requestOptions);

    // Handle non-JSON responses (like WebSocket upgrades, which won't happen here)
    const contentType = response.headers.get('content-type');
    let responseBody;

    if (contentType && contentType.includes('application/json')) {
      responseBody = await response.text();
    } else {
      responseBody = await response.text();
    }

    // Create response with same status and headers
    const nextResponse = new NextResponse(responseBody, {
      status: response.status,
      statusText: response.statusText,
    });

    // Copy response headers
    response.headers.forEach((value, key) => {
      // Skip headers that Next.js handles automatically
      if (!['content-encoding', 'content-length', 'transfer-encoding'].includes(key.toLowerCase())) {
        nextResponse.headers.set(key, value);
      }
    });

    return nextResponse;

  } catch (error) {
    console.error('[ProviderSyncProxy] Request forwarding failed:', error);

    return NextResponse.json(
      {
        error: 'Provider sync service unavailable',
        message: error instanceof Error ? error.message : 'Unknown error',
        timestamp: new Date().toISOString(),
      },
      { status: 503 }
    );
  }
}