"use client";

import { FormEvent, useMemo, useState } from "react";

type Point = { x: number; y: number };
type Placement = {
  id: string;
  category: string;
  center: Point;
  width: number;
  depth: number;
  rotation_degrees: number;
  source_product_id?: string | null;
};
type Room = {
  id: string;
  room_type: string;
  polygon: Point[];
  area: number;
  furniture: Placement[];
};
type FloorPlan = {
  source_width: number;
  source_height: number;
  rooms: Room[];
  warnings: string[];
  analysis_method: string;
};
type ValidationIssue = {
  code: string;
  message: string;
  room_id: string;
  item_ids: string[];
};
type Candidate = {
  id: string;
  rank: number;
  policy: "balanced" | "wall_first" | "fit_first";
  execution_ready: boolean;
  design: { floor_plan: FloorPlan; placed_items: number; warnings: string[] };
  validation: { valid: boolean; checked_rooms: number; checked_items: number; issues: ValidationIssue[] };
};
type Portfolio = {
  selected_candidate_id: string;
  candidates: Candidate[];
  ranking_basis: string;
  decision_graph: { nodes: unknown[]; edges: unknown[] };
};
type ApiResult = {
  analysis: { floor_plan: FloorPlan; placed_items: number; warnings: string[] };
  portfolio: Portfolio;
};

type Locale = "ar" | "en";

const copy = {
  ar: {
    eyebrow: "ذكاء مكاني قابل للتحقق",
    title: "FurnitureAI Studio",
    subtitle: "حلّل المخطط، ولّد عدة توزيعات، وارفض أي تصميم لا يجتاز القيود الهندسية.",
    upload: "صورة المخطط",
    preferences: "تفضيلات التصميم",
    preferencesHint: "مثال: مودرن هادئ، ألوان فاتحة، ميزانية عملية",
    scale: "بكسل لكل سم (اختياري)",
    openai: "تحسين دلالي عبر OpenAI",
    run: "تحليل وبناء التصميم",
    running: "جاري التحليل والتحقق…",
    selected: "التصميم المختار",
    ready: "قابل للتنفيذ هندسيًا",
    blocked: "مرفوض هندسيًا",
    placed: "قطع موضوعة",
    rooms: "غرف",
    issues: "مخالفات",
    candidates: "المرشحون",
    graph: "عقد القرار",
    warnings: "تنبيهات التحليل",
    noResult: "ارفع مخططًا حقيقيًا لبدء التحليل.",
  },
  en: {
    eyebrow: "Auditable spatial intelligence",
    title: "FurnitureAI Studio",
    subtitle: "Analyze a plan, generate multiple layouts, and reject designs that fail geometric constraints.",
    upload: "Floor-plan image",
    preferences: "Design preferences",
    preferencesHint: "Example: calm modern, light colors, practical budget",
    scale: "Pixels per cm (optional)",
    openai: "Semantic refinement with OpenAI",
    run: "Analyze and build design",
    running: "Analyzing and validating…",
    selected: "Selected design",
    ready: "Geometry validated",
    blocked: "Geometry rejected",
    placed: "Placed items",
    rooms: "Rooms",
    issues: "Issues",
    candidates: "Candidates",
    graph: "Decision nodes",
    warnings: "Analysis warnings",
    noResult: "Upload a real floor plan to start analysis.",
  },
} as const;

function PlanPreview({ candidate }: { candidate: Candidate }) {
  const plan = candidate.design.floor_plan;
  const width = Math.max(plan.source_width, 1);
  const height = Math.max(plan.source_height, 1);
  return (
    <div className="plan-shell" aria-label="Validated floor plan preview">
      <svg viewBox={`0 0 ${width} ${height}`} role="img">
        {plan.rooms.map((room) => (
          <g key={room.id}>
            <polygon
              className="room-shape"
              points={room.polygon.map((point) => `${point.x},${point.y}`).join(" ")}
            />
            {room.furniture.map((item) => (
              <rect
                key={item.id}
                className="furniture-shape"
                x={item.center.x - item.width / 2}
                y={item.center.y - item.depth / 2}
                width={item.width}
                height={item.depth}
                rx={Math.min(item.width, item.depth) * 0.08}
                transform={`rotate(${item.rotation_degrees} ${item.center.x} ${item.center.y})`}
              />
            ))}
          </g>
        ))}
      </svg>
    </div>
  );
}

export default function Home() {
  const [locale, setLocale] = useState<Locale>("ar");
  const [file, setFile] = useState<File | null>(null);
  const [preferences, setPreferences] = useState("");
  const [pixelsPerCm, setPixelsPerCm] = useState("");
  const [useOpenAI, setUseOpenAI] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState<ApiResult | null>(null);
  const t = copy[locale];

  const selected = useMemo(() => {
    if (!result) return null;
    return (
      result.portfolio.candidates.find(
        (candidate) => candidate.id === result.portfolio.selected_candidate_id,
      ) ?? result.portfolio.candidates[0] ?? null
    );
  }, [result]);

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!file || busy) return;
    setBusy(true);
    setError("");
    setResult(null);
    const body = new FormData();
    body.append("image", file);
    body.append("preferences", preferences);
    body.append("use_openai", String(useOpenAI));
    if (pixelsPerCm.trim()) body.append("pixels_per_cm", pixelsPerCm.trim());

    try {
      const response = await fetch("/api/design", { method: "POST", body });
      const payload = (await response.json()) as ApiResult & { detail?: string };
      if (!response.ok) throw new Error(payload.detail || `HTTP ${response.status}`);
      setResult(payload);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Request failed");
    } finally {
      setBusy(false);
    }
  }

  return (
    <main dir={locale === "ar" ? "rtl" : "ltr"}>
      <header className="topbar">
        <div className="brand-mark">F</div>
        <div className="brand-copy">
          <strong>FurnitureAI</strong>
          <span>Spatial Design OS</span>
        </div>
        <button className="language" type="button" onClick={() => setLocale(locale === "ar" ? "en" : "ar")}>
          {locale === "ar" ? "EN" : "عربي"}
        </button>
      </header>

      <section className="hero">
        <div>
          <p className="eyebrow">{t.eyebrow}</p>
          <h1>{t.title}</h1>
          <p className="subtitle">{t.subtitle}</p>
        </div>
        <div className="signal-grid" aria-hidden="true">
          <span>VISION</span><span>GEOMETRY</span><span>DIA</span><span>GRAPH</span>
        </div>
      </section>

      <section className="workspace">
        <form className="control-panel" onSubmit={submit}>
          <label className="file-drop">
            <span>{t.upload}</span>
            <strong>{file ? file.name : "PNG · JPEG · WEBP"}</strong>
            <input
              type="file"
              accept="image/png,image/jpeg,image/webp"
              required
              onChange={(event) => setFile(event.target.files?.[0] ?? null)}
            />
          </label>

          <label>
            <span>{t.preferences}</span>
            <textarea
              value={preferences}
              maxLength={3000}
              placeholder={t.preferencesHint}
              onChange={(event) => setPreferences(event.target.value)}
            />
          </label>

          <label>
            <span>{t.scale}</span>
            <input
              inputMode="decimal"
              value={pixelsPerCm}
              onChange={(event) => setPixelsPerCm(event.target.value)}
              placeholder="e.g. 4.25"
            />
          </label>

          <label className="check-row">
            <input
              type="checkbox"
              checked={useOpenAI}
              onChange={(event) => setUseOpenAI(event.target.checked)}
            />
            <span>{t.openai}</span>
          </label>

          <button className="primary" type="submit" disabled={!file || busy}>
            {busy ? t.running : t.run}
          </button>
          {error ? <p className="error" role="alert">{error}</p> : null}
        </form>

        <section className="result-panel">
          {!selected || !result ? (
            <div className="empty-state"><span>◎</span><p>{t.noResult}</p></div>
          ) : (
            <>
              <div className="result-head">
                <div>
                  <p className="eyebrow">{t.selected}</p>
                  <h2>{selected.policy.replace("_", " ")}</h2>
                </div>
                <span className={selected.execution_ready ? "status good" : "status bad"}>
                  {selected.execution_ready ? t.ready : t.blocked}
                </span>
              </div>

              <PlanPreview candidate={selected} />

              <div className="metrics">
                <article><strong>{selected.design.placed_items}</strong><span>{t.placed}</span></article>
                <article><strong>{selected.design.floor_plan.rooms.length}</strong><span>{t.rooms}</span></article>
                <article><strong>{selected.validation.issues.length}</strong><span>{t.issues}</span></article>
                <article><strong>{result.portfolio.decision_graph.nodes.length}</strong><span>{t.graph}</span></article>
              </div>

              <div className="candidate-list">
                <h3>{t.candidates}</h3>
                {result.portfolio.candidates.map((candidate) => (
                  <article key={candidate.id} className={candidate.id === selected.id ? "candidate active" : "candidate"}>
                    <b>#{candidate.rank}</b>
                    <span>{candidate.policy.replace("_", " ")}</span>
                    <span>{candidate.design.placed_items} items</span>
                    <i>{candidate.execution_ready ? "PASS" : "FAIL"}</i>
                  </article>
                ))}
              </div>

              {selected.validation.issues.length ? (
                <div className="issues">
                  {selected.validation.issues.map((issue, index) => (
                    <p key={`${issue.code}-${index}`}><b>{issue.code}</b> · {issue.message}</p>
                  ))}
                </div>
              ) : null}

              {result.analysis.warnings.length ? (
                <details>
                  <summary>{t.warnings}</summary>
                  {result.analysis.warnings.map((warning, index) => <p key={index}>{warning}</p>)}
                </details>
              ) : null}
            </>
          )}
        </section>
      </section>
    </main>
  );
}
