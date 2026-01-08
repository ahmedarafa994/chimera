import "server-only";

export interface ChimeraBackendInfo {
  origin: string;
  apiBaseUrl: string;
  wsBaseUrl: string;
}

type CacheEntry = {
  info: ChimeraBackendInfo;
  expiresAtMs: number;
};

let cached: CacheEntry | null = null;

function parseBackendOriginFromEnv(): string | null {
  const raw =
    process.env.NEXT_PUBLIC_CHIMERA_API_URL ||
    process.env.NEXT_PUBLIC_API_URL ||
    process.env.CHIMERA_API_URL ||
    null;

  if (!raw) return null;

  try {
    const url = new URL(raw);
    return url.origin;
  } catch {
    return null;
  }
}

function toWsBaseUrl(origin: string): string {
  if (origin.startsWith("https://")) return `wss://${origin.slice("https://".length)}`;
  if (origin.startsWith("http://")) return `ws://${origin.slice("http://".length)}`;
  return origin;
}

async function isChimeraBackendAlive(origin: string, timeoutMs: number): Promise<boolean> {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), timeoutMs);

  try {
    const response = await fetch(`${origin}/health/live`, {
      method: "GET",
      cache: "no-store",
      signal: controller.signal,
    });
    if (!response.ok) return false;
    try {
      const data = (await response.json()) as { name?: string; details?: unknown };
      return data?.name === "liveness";
    } catch {
      // If the endpoint isn't JSON for some reason, treat non-empty 2xx as alive.
      return true;
    }
  } catch {
    return false;
  } finally {
    clearTimeout(timeout);
  }
}

function candidatePorts(): number[] {
  const ports: number[] = [];
  // Include port 8001 first as it's the configured default
  ports.push(8001);
  ports.push(8000); // Fallback to 8000
  for (let port = 8002; port <= 8010; port++) ports.push(port);
  return ports;
}

async function discoverBackendOrigin(): Promise<string | null> {
  const envOrigin = parseBackendOriginFromEnv();
  if (envOrigin && (await isChimeraBackendAlive(envOrigin, 600))) return envOrigin;

  const ports = candidatePorts();
  for (const port of ports) {
    const origin = `http://localhost:${port}`;
    if (await isChimeraBackendAlive(origin, 350)) return origin;
  }

  return null;
}

export async function getChimeraBackendInfo(): Promise<ChimeraBackendInfo | null> {
  const now = Date.now();
  if (cached && cached.expiresAtMs > now) return cached.info;

  const origin = await discoverBackendOrigin();
  if (!origin) return null;

  const info: ChimeraBackendInfo = {
    origin,
    apiBaseUrl: `${origin}/api/v1`,
    wsBaseUrl: toWsBaseUrl(origin),
  };

  cached = {
    info,
    expiresAtMs: now + 30_000,
  };

  return info;
}
