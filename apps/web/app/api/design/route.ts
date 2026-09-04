import { NextResponse } from "next/server";

export const runtime = "nodejs";
export const maxDuration = 60;

const MAX_UPLOAD_BYTES = 10 * 1024 * 1024;
const ALLOWED_IMAGE_TYPES = new Set(["image/jpeg", "image/png", "image/webp"]);

function backendConfig(): { base: URL; serviceKey: string } | null {
  const rawBase = process.env.FURNITURE_API_URL?.trim();
  if (!rawBase) return null;

  let base: URL;
  try {
    base = new URL(rawBase);
  } catch {
    return null;
  }
  if (!new Set(["http:", "https:"]).has(base.protocol)) return null;
  if (process.env.NODE_ENV === "production" && base.protocol !== "https:") return null;

  const serviceKey = process.env.FURNITURE_SERVICE_KEY?.trim() ?? "";
  return { base, serviceKey };
}

async function safeError(response: Response): Promise<string> {
  const fallback = `FurnitureAI backend returned HTTP ${response.status}`;
  try {
    const payload = (await response.json()) as { detail?: unknown };
    if (typeof payload.detail === "string") return payload.detail.slice(0, 800);
  } catch {
    return fallback;
  }
  return fallback;
}

function endpoint(base: URL, path: string): URL {
  return new URL(path, base.origin);
}

export async function POST(request: Request) {
  const config = backendConfig();
  if (!config) {
    return NextResponse.json(
      { detail: "FurnitureAI backend is not configured for this deployment." },
      { status: 503 },
    );
  }

  const incoming = await request.formData();
  const image = incoming.get("image");
  if (!(image instanceof File)) {
    return NextResponse.json({ detail: "A floor-plan image is required." }, { status: 422 });
  }
  if (!ALLOWED_IMAGE_TYPES.has(image.type)) {
    return NextResponse.json({ detail: "Use a JPEG, PNG, or WebP image." }, { status: 422 });
  }
  if (image.size < 1 || image.size > MAX_UPLOAD_BYTES) {
    return NextResponse.json({ detail: "Image size is outside the allowed range." }, { status: 413 });
  }

  const preferences = String(incoming.get("preferences") ?? "").trim().slice(0, 3000);
  const useOpenAI = incoming.get("use_openai") === "true";
  const pixelsPerCm = String(incoming.get("pixels_per_cm") ?? "").trim();

  const analyzeBody = new FormData();
  analyzeBody.append("image", image, image.name || "floor-plan");
  analyzeBody.append("use_openai", String(useOpenAI));
  analyzeBody.append("preferences", preferences);
  if (pixelsPerCm) analyzeBody.append("pixels_per_cm", pixelsPerCm);

  const headers: HeadersInit = {};
  if (config.serviceKey) headers["X-API-Key"] = config.serviceKey;

  let analysisResponse: Response;
  try {
    analysisResponse = await fetch(endpoint(config.base, "/api/v1/analyze"), {
      method: "POST",
      body: analyzeBody,
      headers,
      cache: "no-store",
      signal: AbortSignal.timeout(55_000),
    });
  } catch {
    return NextResponse.json({ detail: "FurnitureAI analysis service is unavailable." }, { status: 503 });
  }
  if (!analysisResponse.ok) {
    return NextResponse.json(
      { detail: await safeError(analysisResponse) },
      { status: analysisResponse.status },
    );
  }

  const analysis = (await analysisResponse.json()) as { floor_plan?: unknown };
  if (!analysis.floor_plan || typeof analysis.floor_plan !== "object") {
    return NextResponse.json({ detail: "Backend analysis contract is invalid." }, { status: 502 });
  }

  let portfolioResponse: Response;
  try {
    portfolioResponse = await fetch(endpoint(config.base, "/api/v2/design/portfolio"), {
      method: "POST",
      headers: { ...headers, "Content-Type": "application/json" },
      body: JSON.stringify({
        floor_plan: analysis.floor_plan,
        room_types: {},
        minimum_clearance: 0,
        policies: ["balanced", "wall_first", "fit_first"],
      }),
      cache: "no-store",
      signal: AbortSignal.timeout(30_000),
    });
  } catch {
    return NextResponse.json({ detail: "FurnitureAI portfolio service is unavailable." }, { status: 503 });
  }
  if (!portfolioResponse.ok) {
    return NextResponse.json(
      { detail: await safeError(portfolioResponse) },
      { status: portfolioResponse.status },
    );
  }

  return NextResponse.json(
    {
      analysis,
      portfolio: await portfolioResponse.json(),
    },
    { headers: { "Cache-Control": "no-store" } },
  );
}
