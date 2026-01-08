import { NextRequest, NextResponse } from "next/server";
import { getChimeraBackendInfo } from "@/lib/server/chimera-backend";
import { Agent, fetch as undiciFetch, RequestInit as UndiciRequestInit } from "undici";

export const runtime = "nodejs";

// Extended timeout for long-running operations (10 minutes)
export const maxDuration = 600;

// Timeout for the fetch call to backend (10 minutes)
const BACKEND_FETCH_TIMEOUT_MS = 600_000;

// Create undici dispatcher with extended timeouts to prevent socket timeouts
// during long-running AutoDAN/GPTFuzz operations
// The default Node.js socket timeout is 5 minutes (300,000ms) which causes
// "fetch failed" errors for operations that take longer
const dispatcher = new Agent({
  keepAliveTimeout: BACKEND_FETCH_TIMEOUT_MS,
  keepAliveMaxTimeout: BACKEND_FETCH_TIMEOUT_MS,
  // Body timeout - how long to wait for the response body
  bodyTimeout: BACKEND_FETCH_TIMEOUT_MS,
  // Headers timeout - how long to wait for response headers
  headersTimeout: BACKEND_FETCH_TIMEOUT_MS,
  // Connect timeout
  connectTimeout: 30_000,
  // Keep connections alive
  pipelining: 1,
  connections: 10,
});

function filterRequestHeaders(headers: Headers): Headers {
  const filtered = new Headers(headers);
  filtered.delete("host");
  filtered.delete("connection");
  return filtered;
}

function filterResponseHeaders(headers: Headers): Headers {
  const filtered = new Headers(headers);
  filtered.delete("content-encoding");
  filtered.delete("transfer-encoding");
  filtered.delete("connection");
  return filtered;
}

async function proxyToBackend(
  request: NextRequest,
  params: { path: string[] }
): Promise<NextResponse> {
  const backend = await getChimeraBackendInfo();
  if (!backend) {
    return NextResponse.json(
      {
        detail:
          "Chimera backend is not reachable. Start the backend (backend-api) and ensure it is listening on 8001-8010.",
      },
      { status: 503 }
    );
  }

  const pathStr = params.path.join("/");
  
  // Determine the correct backend path:
  // - If path starts with "v1/", forward to /api/v1/...
  // - If path is "health" or starts with "health/", forward to /health/...
  // - Otherwise, forward to /api/v1/... (most API calls need this prefix)
  let backendPath: string;
  
  if (pathStr.startsWith("v1/")) {
    // Already has v1 prefix, forward to /api/v1/...
    backendPath = `/api/${pathStr}`;
  } else if (pathStr === "health" || pathStr.startsWith("health/")) {
    // Health endpoints are at root level
    backendPath = `/${pathStr}`;
  } else if (pathStr === "integration" || pathStr.startsWith("integration/")) {
    // Integration endpoints are at root level
    backendPath = `/${pathStr}`;
  } else {
    // All other API calls go to /api/v1/...
    backendPath = `/api/v1/${pathStr}`;
  }

  const upstreamUrl = new URL(`${backend.origin}${backendPath}`);
  upstreamUrl.search = request.nextUrl.search;

  // Create an AbortController for timeout handling
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), BACKEND_FETCH_TIMEOUT_MS);

  const init: RequestInit = {
    method: request.method,
    headers: filterRequestHeaders(request.headers),
    redirect: "manual",
    cache: "no-store",
    signal: controller.signal,
  };

  if (request.method !== "GET" && request.method !== "HEAD") {
    init.body = request.body;
    // Required by Node.js fetch when streaming a request body.
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    (init as any).duplex = "half";
  }

  try {
    // Use undici fetch with custom dispatcher for extended timeouts
    // This prevents the default 5-minute socket timeout from killing long-running requests
    const upstreamResponse = await undiciFetch(upstreamUrl.toString(), {
      ...init,
      dispatcher,
    } as UndiciRequestInit);
    clearTimeout(timeoutId);

    const responseHeaders = filterResponseHeaders(new Headers(upstreamResponse.headers as HeadersInit));

    // Convert undici response body to a format NextResponse can handle
    const responseBody = upstreamResponse.body;

    return new NextResponse(responseBody as ReadableStream<Uint8Array> | null, {
      status: upstreamResponse.status,
      headers: responseHeaders,
    });
  } catch (error) {
    clearTimeout(timeoutId);

    // Handle abort/timeout errors
    if (error instanceof Error && error.name === "AbortError") {
      return NextResponse.json(
        {
          detail: "Request timed out. The operation is taking longer than expected. Please try again with simpler parameters or check the backend logs.",
          error_code: "GATEWAY_TIMEOUT",
        },
        { status: 504 }
      );
    }

    // Handle undici-specific timeout errors
    if (error instanceof Error && (
      error.message.includes("body timeout") ||
      error.message.includes("headers timeout") ||
      error.message.includes("connect timeout")
    )) {
      console.error("[API Root Proxy] Timeout error:", error.message);
      return NextResponse.json(
        {
          detail: `Backend request timed out: ${error.message}. The operation is taking longer than expected.`,
          error_code: "GATEWAY_TIMEOUT",
        },
        { status: 504 }
      );
    }

    // Handle other fetch errors
    console.error("[API Root Proxy] Fetch error:", error);
    return NextResponse.json(
      {
        detail: error instanceof Error ? error.message : "Failed to connect to backend",
        error_code: "BAD_GATEWAY",
      },
      { status: 502 }
    );
  }
}

export async function GET(
  request: NextRequest,
  context: { params: Promise<{ path: string[] }> }
) {
  return proxyToBackend(request, await context.params);
}

export async function POST(
  request: NextRequest,
  context: { params: Promise<{ path: string[] }> }
) {
  return proxyToBackend(request, await context.params);
}

export async function PUT(
  request: NextRequest,
  context: { params: Promise<{ path: string[] }> }
) {
  return proxyToBackend(request, await context.params);
}

export async function PATCH(
  request: NextRequest,
  context: { params: Promise<{ path: string[] }> }
) {
  return proxyToBackend(request, await context.params);
}

export async function DELETE(
  request: NextRequest,
  context: { params: Promise<{ path: string[] }> }
) {
  return proxyToBackend(request, await context.params);
}

export async function OPTIONS(
  request: NextRequest,
  context: { params: Promise<{ path: string[] }> }
) {
  return proxyToBackend(request, await context.params);
}