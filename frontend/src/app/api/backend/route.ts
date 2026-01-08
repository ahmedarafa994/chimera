import { NextResponse } from "next/server";
import { getChimeraBackendInfo } from "@/lib/server/chimera-backend";

export const runtime = "nodejs";

export async function GET() {
  const info = await getChimeraBackendInfo();
  if (!info) {
    return NextResponse.json(
      {
        detail:
          "Chimera backend is not reachable. Start the backend (backend-api) and ensure it is listening on 8001-8010.",
      },
      { status: 503 }
    );
  }
  return NextResponse.json(info, { status: 200 });
}

