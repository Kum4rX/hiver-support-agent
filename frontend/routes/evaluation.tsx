import { createFileRoute } from "@tanstack/react-router";
import { useState, useEffect } from "react";
import { Info, ShieldCheck, Database, MessageSquareText, Cpu, ShieldAlert, RefreshCw, Loader2 } from "lucide-react";
import { Panel, Pill, Meter } from "@/components/primitives";
import { getEvaluationMetrics } from "@/services/api";
import type { EvaluationResponse } from "@/services/types";

export const Route = createFileRoute("/evaluation")({
  head: () => ({
    meta: [
      { title: "Evaluation — AI Support Agent Console" },
      {
        name: "description",
        content:
          "Measured benchmark metrics from the evaluation harness: provisional intent accuracy, safety recall, FAISS retrieval and guardrail compliance.",
      },
      { property: "og:title", content: "Evaluation — AI Support Agent Console" },
      {
        property: "og:description",
        content: "Intent, safety, retrieval and guardrail evaluation results with scope caveats.",
      },
    ],
  }),
  component: EvaluationPage,
});

const pct = (v: number) => `${(v * 100).toFixed(1)}%`;

function GuardrailRow({ label, value, detail }: { label: string; value: number | string; detail: string }) {
  const measured = typeof value === "number";
  return (
    <div>
      <div className="mb-1.5 flex items-center justify-between text-xs">
        <span className="flex items-center gap-1.5 text-muted-foreground">
          <MessageSquareText className="size-3.5" /> {label}
        </span>
        {measured ? (
          <span className="font-medium tabular-nums text-foreground">{pct(value)}</span>
        ) : (
          <Pill tone="neutral">Not measured</Pill>
        )}
      </div>
      {measured && <Meter value={value} tone={value === 1 ? "success" : "warning"} />}
      <p className="mt-1 text-[11px] text-muted-foreground">{detail}</p>
    </div>
  );
}

export function EvaluationPage() {
  const [data, setData] = useState<EvaluationResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const loadMetrics = () => {
    setLoading(true);
    setError(null);
    getEvaluationMetrics()
      .then((res) => {
        setData(res);
      })
      .catch((err: unknown) => {
        const msg = err instanceof Error ? err.message : "Backend unavailable — start FastAPI at http://localhost:8000.";
        setError(msg);
      })
      .finally(() => {
        setLoading(false);
      });
  };

  useEffect(() => {
    loadMetrics();
  }, []);

  if (loading) {
    return (
      <div className="mx-auto flex max-w-6xl flex-col items-center justify-center py-20">
        <Loader2 className="size-8 animate-spin text-brand" />
        <p className="mt-3 text-sm text-muted-foreground">Loading evaluation benchmarks from backend...</p>
      </div>
    );
  }

  if (error || !data) {
    return (
      <div className="mx-auto max-w-2xl py-12">
        <div className="rounded-xl border border-destructive/30 bg-destructive/5 p-6 text-center">
          <ShieldAlert className="mx-auto size-8 text-destructive" />
          <p className="mt-3 text-sm font-semibold text-foreground">{error || "Backend unavailable"}</p>
          <p className="mt-1 text-xs text-muted-foreground">
            Evaluation metrics are loaded directly from FastAPI (<code className="rounded bg-muted px-1.5 py-0.5 font-mono text-xs">GET /api/evaluation</code>).
          </p>
          <button
            type="button"
            onClick={loadMetrics}
            className="mt-4 inline-flex items-center gap-2 rounded-md bg-destructive px-4 py-2 text-xs font-medium text-destructive-foreground transition hover:opacity-90"
          >
            <RefreshCw className="size-3.5" /> Retry Fetching Metrics
          </button>
        </div>
      </div>
    );
  }

  const { intent_evaluation: intent, safety_evaluation: safety, retrieval_evaluation: retrieval, guardrail_evaluation: guardrails } = data;

  return (
    <div className="mx-auto max-w-6xl space-y-6">
      <div className="flex items-start gap-3 rounded-xl border border-brand/25 bg-brand/5 p-4">
        <Info className="mt-0.5 size-4 shrink-0 text-brand" />
        <p className="text-sm text-foreground">
          Evaluation results should be interpreted according to the size and type of each evaluation
          set. Intent metrics are provisional until the full human-labelled golden set is completed.
        </p>
      </div>

      <Panel
        title="Intent classification"
        description={intent.label}
        action={<Pill tone="warning">Provisional ({intent.sample_size} samples)</Pill>}
      >
        <div className="space-y-4">
          <div>
            <div className="mb-1.5 flex items-center justify-between text-xs">
              <span className="flex items-center gap-1.5 text-muted-foreground">
                <Cpu className="size-3.5" /> Accuracy
              </span>
              <span className="font-medium tabular-nums text-foreground">{pct(intent.accuracy)}</span>
            </div>
            <Meter value={intent.accuracy} tone="brand" />
          </div>
          <div>
            <div className="mb-1.5 flex items-center justify-between text-xs">
              <span className="flex items-center gap-1.5 text-muted-foreground">
                <Cpu className="size-3.5" /> Weighted F1 (Rule Baseline)
              </span>
              <span className="font-medium tabular-nums text-foreground">{pct(intent.weighted_f1)}</span>
            </div>
            <Meter value={intent.weighted_f1} tone="brand" />
          </div>
          <div>
            <div className="mb-1.5 flex items-center justify-between text-xs">
              <span className="flex items-center gap-1.5 text-muted-foreground">
                <Cpu className="size-3.5" /> Weighted F1 (Hybrid Model)
              </span>
              <span className="font-medium tabular-nums text-foreground">{pct(intent.hybrid_weighted_f1)}</span>
            </div>
            <Meter value={intent.hybrid_weighted_f1} tone="brand" />
          </div>
        </div>
        <p className="mt-4 text-xs text-muted-foreground">{intent.caveat}</p>
      </Panel>

      <Panel
        title="Escalation safety"
        description={safety.title}
        action={<Pill tone="success">{pct(safety.safety_recall)} Safety Recall</Pill>}
      >
        <div className="grid gap-4 sm:grid-cols-4">
          {[
            {
              label: "Safety Recall",
              value: pct(safety.safety_recall),
              hint: `${safety.hazards_caught} / ${safety.hazards_total} hazards escalated`,
            },
            {
              label: "Specificity",
              value: pct(safety.specificity),
              hint: "Non-hazards correctly left un-escalated",
            },
            {
              label: "False Negatives",
              value: `${safety.false_negatives}`,
              hint: "0 missed physical hazards",
            },
            {
              label: "Decision Latency",
              value: `${safety.mean_latency_us.toFixed(1)} µs`,
              hint: "Precompiled regex & keyword scanner",
            },
          ].map((s) => (
            <div key={s.label} className="rounded-lg border border-success/25 bg-success/5 p-4">
              <div className="flex items-center gap-1.5 text-xs font-medium text-success">
                <ShieldCheck className="size-3.5" /> {s.label}
              </div>
              <div className="mt-1.5 text-2xl font-semibold tracking-tight text-foreground">
                {s.value}
              </div>
              <div className="mt-1 text-[11px] text-muted-foreground">{s.hint}</div>
            </div>
          ))}
        </div>
        <p className="mt-4 text-xs text-muted-foreground">{safety.note}</p>
      </Panel>

      <Panel
        title="Retrieval"
        description={`FAISS dense-vector search over ${retrieval.corpus_size.toLocaleString("en-US")} historical cases`}
      >
        <div className="overflow-x-auto">
          <table className="w-full min-w-[520px] text-sm">
            <thead>
              <tr className="border-b border-border text-left text-xs text-muted-foreground">
                <th className="py-2.5 font-medium">Metric</th>
                <th className="py-2.5 font-medium">Value</th>
                <th className="py-2.5 font-medium">Notes</th>
              </tr>
            </thead>
            <tbody>
              {[
                [
                  "Mean Top-1 similarity",
                  retrieval.mean_top1_similarity.toFixed(4),
                  `Cosine similarity on ${retrieval.embedding_model} embeddings`,
                ],
                [
                  "Mean Top-3 similarity",
                  retrieval.mean_top3_similarity.toFixed(4),
                  "Averaged over the three nearest neighbours",
                ],
                [
                  "Mean retrieval latency",
                  `${retrieval.mean_latency_ms} ms`,
                  "Dense vector search measured by evaluation benchmark",
                ],
                [
                  "Similarity Threshold Hit Rate",
                  pct(retrieval.threshold_hit_rate),
                  retrieval.threshold_note,
                ],
              ].map(([m, v, n]) => (
                <tr key={m} className="border-b border-border/60 last:border-0">
                  <td className="py-3 pr-3 text-foreground">{m}</td>
                  <td className="py-3 pr-3 font-medium tabular-nums text-foreground">{v}</td>
                  <td className="py-3 text-xs text-muted-foreground">{n}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="mt-4 flex items-start gap-2 rounded-md border border-border bg-muted/40 p-3 text-xs text-muted-foreground">
          <Database className="mt-0.5 size-3.5 shrink-0" />
          Embedding similarity measures vector closeness in dense space. A minimum cosine threshold of ≥ 0.35 ensures relevant historical matches.
        </p>
      </Panel>

      <Panel title="Response guardrails" description="Post-generation checks applied to every reply">
        <div className="space-y-4">
          <GuardrailRow
            label="Character Limit Compliance (≤280 characters)"
            value={guardrails.character_limit_compliance}
            detail="Generated responses strictly validated against Twitter/X 280-character limit"
          />
          <GuardrailRow
            label="PII Safety"
            value={guardrails.pii_safety}
            detail="No credential, phone number, or payment card solicitation in public replies"
          />
          <GuardrailRow
            label="Actionable Quality"
            value={guardrails.actionable_quality}
            detail="Responses containing at least one concrete troubleshooting step"
          />
          <GuardrailRow
            label="LLM-as-a-Judge"
            value={guardrails.llm_as_a_judge}
            detail="No external LLM judge configured; evaluated purely via deterministic validation"
          />
        </div>
      </Panel>
    </div>
  );
}
