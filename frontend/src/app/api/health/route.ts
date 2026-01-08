import { NextResponse } from "next/server";
import { getChimeraBackendInfo } from "@/lib/server/chimera-backend";

export const runtime = "nodejs";

export async function GET() {
  const backend = await getChimeraBackendInfo();
  if (!backend) {
    return NextResponse.json(
      { status: "unreachable", detail: "Chimera backend is not reachable." },
      { status: 503 }
    );
  }

  try {
    const response = await fetch(`${backend.origin}/health/live`, { cache: "no-store" });
    const body = await response.text();
    return new NextResponse(body, {
      status: response.status,
      headers: response.headers,
    });
  } catch (e) {
    return NextResponse.json(
      { status: "unreachable", detail: e instanceof Error ? e.message : String(e) },
      { status: 503 }
    );
  }
}
