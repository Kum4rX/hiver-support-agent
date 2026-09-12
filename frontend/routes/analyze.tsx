import { createFileRoute } from "@tanstack/react-router";
import { useCallback, useEffect, useState } from "react";
import {
  Loader2,
  Sparkles,
  Copy,
  RefreshCw,
  ShieldAlert,
  Check,
  X,
  Ban,
  Layers,
  Braces,
  Gauge,
  Cpu,
  Database,
  MessageSquareText,
  ShieldCheck,
  Wand2,
  ArrowRight,
  Lock,
} from "lucide-react";
import { toast } from "sonner";
import { Panel, Pill, Meter, RiskBadge, DecisionBadge, riskTone } from "@/components/primitives";
import { cn } from "@/lib/utils";
import {
  analyzeCustomerMessage,
  EXAMPLE_QUERIES,
  type AnalysisResult,
  type PipelineStage,
} from "@/services/api";

export const Route = createFileRoute("/analyze")({
  validateSearch: (search: Record<string, unknown>): { q?: string } => {
    const q = search["q"];
    return typeof q === "string" ? { q } : {};
  },
  head: () => ({
    meta: [
      { title: "Analyze Ticket — AI Support Agent Console" },
      {
        name: "description",
        content:
          "Run a support ticket through intent classification, risk scoring, FAISS retrieval, response generation and guardrails, stage by stage.",
      },
      { property: "og:title", content: "Analyze Ticket — AI Support Agent Console" },
      {
        property: "og:description",
        content: "Observe every pipeline stage for a single customer message.",
      },
    ],
  }),
  component: AnalyzePage,
});

const MAX_CHARS = 600;

const stageIcons: Record<string, typeof Cpu> = {
  preprocess: Braces,
  intent: Cpu,
  risk: Gauge,
  retrieval: Database,
  generation: MessageSquareText,
  guardrails: ShieldCheck,
};

function AnalyzePage() {
  const { q } = Route.useSearch();
  const [text, setText] = useState(q ?? "");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<AnalysisResult | null>(null);

  const run = useCallback(async (value: string) => {
    if (!value.trim()) return;
    setLoading(true);
    setResult(null);
    setError(null);
    try {
      setResult(await analyzeCustomerMessage(value.trim()));
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : "Backend unavailable — start FastAPI at http://localhost:8000.";
      setError(msg);
      toast.error(msg);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    if (q) void run(q);
  }, [q, run]);

  return (
    <div className="mx-auto max-w-7xl space-y-6">
      <Panel
        title="Customer message"
        description="Paste an inbound ticket or pick one of the evaluation scenarios"
      >
        <div className="flex flex-wrap gap-2 pb-4">
          {EXAMPLE_QUERIES.map((ex) => (
            <button
              key={ex.id}
              type="button"
              onClick={() => {
                setText(ex.query);
                void run(ex.query);
              }}
              className={cn(
                "rounded-full border px-3 py-1.5 text-xs font-medium transition-colors",
                ex.tone === "critical"
                  ? "border-danger/30 bg-danger/5 text-danger hover:bg-danger/10"
                  : ex.tone === "warning"
                    ? "border-warning/30 bg-warning/5 text-warning hover:bg-warning/10"
                    : ex.tone === "info"
                      ? "border-brand/25 bg-brand/5 text-brand hover:bg-brand/10"
                      : "border-border bg-card text-muted-foreground hover:bg-accent hover:text-foreground",
              )}
            >
              {ex.label}
            </button>
          ))}
        </div>

        <textarea
          value={text}
          maxLength={MAX_CHARS}
          onChange={(e) => setText(e.target.value)}
          rows={5}
          placeholder="e.g. @AppleSupport my iPhone battery drains within a couple of hours since the last update…"
          className="w-full resize-y rounded-lg border border-border bg-background p-3.5 text-sm text-foreground outline-none transition placeholder:text-muted-foreground focus:border-ring focus:ring-2 focus:ring-ring/25"
        />

        <div className="mt-3 flex flex-wrap items-center justify-between gap-3">
          <span className="text-xs tabular-nums text-muted-foreground">
            {text.length} / {MAX_CHARS} characters
          </span>
          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={() => {
                setText("");
                setResult(null);
              }}
              className="rounded-md border border-border bg-card px-3.5 py-2 text-sm font-medium text-foreground transition hover:bg-accent"
            >
              Clear
            </button>
            <button
              type="button"
              disabled={loading || !text.trim()}
              onClick={() => void run(text)}
              className="inline-flex items-center gap-2 rounded-md bg-brand px-4 py-2 text-sm font-medium text-brand-foreground transition hover:opacity-90 disabled:cursor-not-allowed disabled:opacity-50"
            >
              {loading ? (
                <>
                  <Loader2 className="size-4 animate-spin" /> Analyzing…
                </>
              ) : (
                <>
                  <Sparkles className="size-4" /> Analyze
                </>
              )}
            </button>
          </div>
        </div>
      </Panel>

      {loading && <LoadingWorkspace />}

      {error && !loading && (
        <div className="rounded-xl border border-destructive/30 bg-destructive/5 p-6 text-center">
          <ShieldAlert className="mx-auto size-8 text-destructive" />
          <p className="mt-3 text-sm font-semibold text-foreground">{error}</p>
          <p className="mt-1 text-xs text-muted-foreground">
            Please make sure the FastAPI server is running: <code className="rounded bg-muted px-1.5 py-0.5 font-mono text-xs text-foreground">python -m uvicorn backend.main:app --port 8000</code>
          </p>
          <button
            type="button"
            onClick={() => void run(text)}
            className="mt-4 inline-flex items-center gap-2 rounded-md bg-destructive px-3.5 py-1.5 text-xs font-medium text-destructive-foreground transition hover:opacity-90"
          >
            <RefreshCw className="size-3.5" /> Retry Request
          </button>
        </div>
      )}

      {result && !loading && <Workspace result={result} onRegenerate={() => void run(result.query)} />}

      {!result && !loading && !error && (
        <div className="rounded-xl border border-dashed border-border bg-card/50 p-10 text-center">
          <Wand2 className="mx-auto size-6 text-muted-foreground" />
          <p className="mt-3 text-sm font-medium text-foreground">No analysis yet</p>
          <p className="mt-1 text-xs text-muted-foreground">
            Run a scenario to see intent, risk, retrieval and guardrails for this ticket.
          </p>
        </div>
      )}
    </div>
  );
}

function LoadingWorkspace() {
  return (
    <div className="grid gap-5 lg:grid-cols-2">
      {[0, 1].map((i) => (
        <div key={i} className="space-y-3 rounded-xl border border-border bg-card p-5 shadow-sm">
          {[...Array(5)].map((_, j) => (
            <div
              key={j}
              className="h-4 animate-pulse rounded bg-muted"
              style={{ width: `${90 - j * 12}%` }}
            />
          ))}
        </div>
      ))}
    </div>
  );
}

function Workspace({ result, onRegenerate }: { result: AnalysisResult; onRegenerate: () => void }) {
  const isCritical = result.risk_level === "critical";
  const isOod = result.decision === "out_of_domain";
  const isMulti = result.secondary_intents.length > 0 && !isCritical;

  return (
    <div className="space-y-5">
      {isCritical && <CriticalSafetyView result={result} />}
      {isOod && <OutOfDomainView />}

      <div className="grid gap-5 lg:grid-cols-2">
        {/* Left column */}
        <div className="space-y-5">
          <Panel title="Customer message">
            <p className="text-sm leading-relaxed text-foreground">{result.query}</p>
          </Panel>

          <Panel title="Intent classification">
            <div className="space-y-4">
              <Row label="Detected intent">
                <span className="font-mono text-sm font-medium text-foreground">{result.intent}</span>
              </Row>
              <Row label="Secondary intent">
                {result.secondary_intents.length ? (
                  <div className="flex flex-wrap justify-end gap-1.5">
                    {result.secondary_intents.map((s) => (
                      <Pill key={s} tone="brand">
                        {s}
                      </Pill>
                    ))}
                  </div>
                ) : (
                  <span className="text-sm text-muted-foreground">None</span>
                )}
              </Row>

              <div>
                <div className="mb-1.5 flex items-center justify-between text-xs">
                  <span className="text-muted-foreground">Confidence</span>
                  <span className="font-medium tabular-nums text-foreground">
                    {(result.confidence * 100).toFixed(1)}%
                  </span>
                </div>
                <Meter
                  value={result.confidence}
                  tone={result.confidence > 0.85 ? "success" : result.confidence > 0.7 ? "warning" : "danger"}
                />
              </div>

              <Row label="Escalation">
                {result.escalated ? (
                  <Pill tone="danger">
                    <ShieldAlert className="size-3.5" /> Escalated
                  </Pill>
                ) : (
                  <Pill tone="success">
                    <Check className="size-3.5" /> Not escalated
                  </Pill>
                )}
              </Row>
              {result.escalation_reason && (
                <p className="rounded-md border border-danger/25 bg-danger/5 p-2.5 text-xs text-danger">
                  {result.escalation_reason}
                </p>
              )}
            </div>
          </Panel>

          {isMulti && (
            <Panel title="Multi-intent handling" action={<Pill tone="warning">Multi-intent detected</Pill>}>
              <div className="flex flex-col gap-3 sm:flex-row sm:items-center">
                <div className="flex-1 rounded-lg border border-brand/25 bg-brand/5 p-3">
                  <div className="text-[11px] font-medium uppercase tracking-wide text-brand">
                    Primary — routes the ticket
                  </div>
                  <div className="mt-1 font-mono text-sm text-foreground">{result.intent}</div>
                </div>
                <ArrowRight className="mx-auto size-4 shrink-0 rotate-90 text-muted-foreground sm:rotate-0" />
                <div className="flex-1 rounded-lg border border-border bg-muted/50 p-3">
                  <div className="text-[11px] font-medium uppercase tracking-wide text-muted-foreground">
                    Secondary — retained as context
                  </div>
                  <div className="mt-1 font-mono text-sm text-foreground">
                    {result.secondary_intents.join(", ")}
                  </div>
                </div>
              </div>
              <p className="mt-3 text-xs text-muted-foreground">
                Primary intent is used for routing; secondary intent is retained as response
                context.
              </p>
            </Panel>
          )}
        </div>

        {/* Right column */}
        <div className="space-y-5">
          <Panel title="Agent decision">
            <div className="space-y-4">
              <Row label="Decision">
                <DecisionBadge decision={result.decision} />
              </Row>
              <Row label="Risk level">
                <RiskBadge risk={result.risk_level} />
              </Row>
              <Row label="Routing target">
                <span className="text-sm text-foreground">{result.routing_target}</span>
              </Row>
              <Row label="Processing time">
                <span className="text-sm font-medium tabular-nums text-foreground">
                  {result.latency_ms} ms
                </span>
              </Row>
            </div>
          </Panel>

          <Panel title="Pipeline progress">
            <ol className="space-y-3">
              {result.stages.map((stage) => (
                <StageRow key={stage.id} stage={stage} />
              ))}
            </ol>
          </Panel>
        </div>
      </div>

      <Panel
        title="Retrieved historical cases (FAISS)"
        description={
          result.retrieved_cases.length
            ? result.risk_level === "critical"
              ? "Retrieved for context — not used to generate safety instructions."
              : `${result.retrieved_cases.length} nearest neighbours from 65,239 indexed support conversations`
            : "Retrieval skipped for this query"
        }
      >
        {result.retrieved_cases.length === 0 ? (
          <p className="text-sm text-muted-foreground">
            No cases retrieved — the query falls outside the supported corpus domain.
          </p>
        ) : (
          <div className="space-y-4">
            {result.retrieved_cases.map((c, i) => (
              <article key={i} className="rounded-lg border border-border p-4">
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <div className="flex items-center gap-2">
                    <span className="text-xs font-medium text-muted-foreground">Case #{i + 1}</span>
                    <Pill tone={result.risk_level === "critical" ? "warning" : "brand"}>
                      {result.risk_level === "critical" ? "Context only" : "Grounded"}
                    </Pill>
                  </div>
                  <span className="text-xs tabular-nums text-muted-foreground">
                    similarity {c.similarity.toFixed(2)}
                  </span>
                </div>
                <Meter className="mt-2" value={c.similarity} tone="violet" />
                <div className="mt-3 grid gap-3 md:grid-cols-2">
                  <div>
                    <div className="text-[11px] font-medium uppercase tracking-wide text-muted-foreground">
                      Customer issue
                    </div>
                    <p className="mt-1 text-sm text-foreground">{c.customer_text}</p>
                  </div>
                  <div>
                    <div className="text-[11px] font-medium uppercase tracking-wide text-muted-foreground">
                      Historical support response
                    </div>
                    <p className="mt-1 text-sm text-foreground">{c.support_text}</p>
                  </div>
                </div>
              </article>
            ))}
          </div>
        )}
      </Panel>

      <ResponseCard result={result} onRegenerate={onRegenerate} locked={isCritical} />
    </div>
  );
}

function Row({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="flex items-start justify-between gap-4 border-b border-border/60 pb-3 last:border-0 last:pb-0">
      <span className="text-xs text-muted-foreground">{label}</span>
      <div className="text-right">{children}</div>
    </div>
  );
}

function StageRow({ stage }: { stage: PipelineStage }) {
  const Icon = stageIcons[stage.id] ?? Cpu;
  const skipped = stage.status === "skipped";
  return (
    <li className="flex items-start gap-3">
      <div
        className={cn(
          "mt-0.5 flex size-7 shrink-0 items-center justify-center rounded-md border",
          skipped
            ? "border-border bg-muted text-muted-foreground"
            : "border-success/25 bg-success/10 text-success",
        )}
      >
        <Icon className="size-3.5" />
      </div>
      <div className="min-w-0 flex-1">
        <div className="flex items-center justify-between gap-2">
          <span className="text-sm font-medium text-foreground">{stage.label}</span>
          <span className="shrink-0 text-[11px] tabular-nums text-muted-foreground">
            {skipped ? "skipped" : `${stage.duration_ms.toFixed(1)} ms`}
          </span>
        </div>
        <p className="text-xs text-muted-foreground">{stage.detail}</p>
      </div>
    </li>
  );
}

function ResponseCard({
  result,
  onRegenerate,
  locked,
}: {
  result: AnalysisResult;
  onRegenerate: () => void;
  locked: boolean;
}) {
  const pct = result.response_length / 280;
  return (
    <Panel
      title="Generated response"
      description={
        locked
          ? "Deterministic safety response — generation bypassed"
          : "Sent back to the customer on the original channel"
      }
      action={
        locked ? (
          <Pill tone="danger">
            <Lock className="size-3.5" /> Safety response locked
          </Pill>
        ) : result.retrieved_cases.length ? (
          <Pill tone="success">Grounded</Pill>
        ) : (
          <Pill tone="warning">Template</Pill>
        )
      }
    >
      <p className="rounded-lg border border-border bg-muted/40 p-4 text-sm leading-relaxed text-foreground">
        {result.response}
      </p>

      <div className="mt-4">
        <div className="mb-1.5 flex items-center justify-between text-xs">
          <span className="text-muted-foreground">Twitter limit</span>
          <span
            className={cn(
              "font-medium tabular-nums",
              result.guardrails.length_ok ? "text-foreground" : "text-danger",
            )}
          >
            {result.response_length} / 280
          </span>
        </div>
        <Meter value={pct} tone={pct > 1 ? "danger" : pct > 0.85 ? "warning" : "success"} />
      </div>

      <div className="mt-4 flex flex-wrap items-center gap-x-5 gap-y-2">
        <GuardCheck ok={result.guardrails.length_ok} label="Length OK" />
        <GuardCheck ok={result.guardrails.pii_safe} label="PII Safe" />
        <GuardCheck ok={result.guardrails.actionable} label="Actionable" />
      </div>

      <div className="mt-5 flex flex-wrap gap-2">
        <button
          type="button"
          onClick={() => {
            void navigator.clipboard.writeText(result.response);
            toast.success("Response copied to clipboard");
          }}
          className="inline-flex items-center gap-2 rounded-md border border-border bg-card px-3.5 py-2 text-sm font-medium text-foreground transition hover:bg-accent"
        >
          <Copy className="size-4" /> Copy Response
        </button>
        {locked ? (
          <span className="inline-flex items-center gap-2 rounded-md border border-danger/25 bg-danger/5 px-3.5 py-2 text-sm font-medium text-danger">
            <Lock className="size-4" /> Deterministic safety response
          </span>
        ) : (
          <button
            type="button"
            onClick={onRegenerate}
            className="inline-flex items-center gap-2 rounded-md border border-border bg-card px-3.5 py-2 text-sm font-medium text-foreground transition hover:bg-accent"
          >
            <RefreshCw className="size-4" /> Regenerate
          </button>
        )}
      </div>
    </Panel>
  );
}

function GuardCheck({ ok, label }: { ok: boolean; label: string }) {
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1.5 text-xs font-medium",
        ok ? "text-success" : "text-danger",
      )}
    >
      {ok ? <Check className="size-4" /> : <X className="size-4" />}
      {label}
    </span>
  );
}

function CriticalSafetyView({ result }: { result: AnalysisResult }) {
  return (
    <section className="rounded-xl border-2 border-danger/40 bg-danger/5 p-5 shadow-sm">
      <div className="flex items-start gap-3">
        <div className="flex size-9 shrink-0 items-center justify-center rounded-lg bg-danger text-danger-foreground">
          <ShieldAlert className="size-5" />
        </div>
        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-center gap-2">
            <h2 className="text-base font-semibold text-danger">Critical safety hazard detected</h2>
            <Pill tone="danger">Deterministic override</Pill>
            <Pill tone="danger">{result.routing_target}</Pill>
          </div>
          <dl className="mt-3 grid gap-2 sm:grid-cols-2">
            {[
              ["Risk", result.risk_level.toUpperCase()],
              ["Decision", "HUMAN REVIEW"],
              ["Reason", result.escalation_reason ?? "PHYSICAL_SAFETY_HAZARD"],
              ["Routing", result.routing_target],
            ].map(([k, v]) => (
              <div key={k} className="rounded-md border border-danger/25 bg-card px-3 py-2">
                <dt className="text-[11px] uppercase tracking-wide text-muted-foreground">{k}</dt>
                <dd className="mt-0.5 font-mono text-xs text-foreground">{v}</dd>
              </div>
            ))}
          </dl>
          <p className="mt-3 text-sm text-foreground">
            Standard troubleshooting has been suppressed. The agent returned fixed safety
            instructions and routed the conversation to a human specialist immediately.
          </p>
          <ul className="mt-3 space-y-1.5 text-sm text-foreground">
            {[
              "Stop using and charging the device right now.",
              "Keep the device away from flammable materials and do not puncture the battery.",
              "Do not attempt a battery replacement or DIY repair.",
              "Hardware Safety Team notified for an urgent inspection.",
            ].map((line) => (
              <li key={line} className="flex items-start gap-2">
                <Check className="mt-0.5 size-4 shrink-0 text-danger" />
                {line}
              </li>
            ))}
          </ul>

        </div>
      </div>
    </section>
  );
}

function OutOfDomainView() {
  return (
    <section className="rounded-xl border border-warning/35 bg-warning/5 p-5 shadow-sm">
      <div className="flex items-start gap-3">
        <div className="flex size-9 shrink-0 items-center justify-center rounded-lg bg-warning text-warning-foreground">
          <Ban className="size-5" />
        </div>
        <div>
          <div className="flex flex-wrap items-center gap-2">
            <h2 className="text-base font-semibold text-warning">Out of domain</h2>
            <Pill tone="warning">
              <Layers className="size-3.5" /> No retrieval performed
            </Pill>
          </div>
          <dl className="mt-3 grid gap-2 sm:grid-cols-3">
            {[
              ["Platform", "Windows / Dell"],
              ["Decision", "BOUNDARY RESPONSE"],
              ["Risk", "NONE"],
            ].map(([k, v]) => (
              <div key={k} className="rounded-md border border-warning/25 bg-card px-3 py-2">
                <dt className="text-[11px] uppercase tracking-wide text-muted-foreground">{k}</dt>
                <dd className="mt-0.5 font-mono text-xs text-foreground">{v}</dd>
              </div>
            ))}
          </dl>
          <p className="mt-2 text-sm text-foreground">
            The ticket refers to non-Apple hardware or software. The agent returns a polite boundary
            message instead of guessing an answer, and no case is created.
          </p>
        </div>
      </div>
    </section>
  );
}
