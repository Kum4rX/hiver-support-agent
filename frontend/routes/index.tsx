import { createFileRoute, Link } from "@tanstack/react-router";
import { useState, useEffect } from "react";
import { ArrowUpRight, PlayCircle } from "lucide-react";
import { Panel, Pill, Stat, RiskBadge, DecisionBadge } from "@/components/primitives";
import { CURATED_SCENARIOS, getEvaluationMetrics } from "@/services/api";
import {
  ENVIRONMENT_LABEL,
  ENVIRONMENT_SUBTITLE,
  NOT_MEASURED,
} from "@/services/demoData";
import type { EvaluationResponse } from "@/services/types";

export const Route = createFileRoute("/")({
  head: () => ({
    meta: [
      { title: "Overview — AI Support Agent Console" },
      {
        name: "description",
        content:
          "Curated evaluation scenarios and measured benchmark metrics for the Hiver AI customer support agent.",
      },
      { property: "og:title", content: "Overview — AI Support Agent Console" },
      {
        property: "og:description",
        content: "Curated demo scenarios and measured evaluation metrics at a glance.",
      },
    ],
  }),
  component: Overview,
});

export function Overview() {
  const [metrics, setMetrics] = useState<EvaluationResponse | null>(null);

  useEffect(() => {
    getEvaluationMetrics()
      .then((data) => setMetrics(data))
      .catch(() => {
        // Graceful fallback if backend is offline on overview load
      });
  }, []);

  const corpusCount = metrics?.retrieval_evaluation?.corpus_size?.toLocaleString("en-US") ?? "65,239";
  const safetyRecall = metrics?.safety_evaluation?.safety_recall
    ? `${(metrics.safety_evaluation.safety_recall * 100).toFixed(0)}%`
    : "100%";
  const retrievalLatency = metrics?.retrieval_evaluation?.mean_latency_ms
    ? `${metrics.retrieval_evaluation.mean_latency_ms} ms`
    : "93.73 ms";
  const sampleSize = metrics?.intent_evaluation?.sample_size ?? 11;

  const kpis = [
    {
      key: "runs",
      label: "Operational Run History",
      value: NOT_MEASURED,
      hint: "Run history is not persisted by this demo",
    },
    {
      key: "safety-tests",
      label: "Curated Safety Tests",
      value: "13",
      hint: "13 / 13 hazards escalated (0 missed)",
    },
    {
      key: "golden",
      label: "Human-Labelled Golden Examples",
      value: `${sampleSize}`,
      hint: "Provisional intent evaluation set",
    },
    {
      key: "corpus",
      label: "Retrieval Corpus",
      value: corpusCount,
      hint: "Historical support conversations indexed in FAISS",
    },
    {
      key: "recall",
      label: "Safety Recall",
      value: safetyRecall,
      hint: "Measured on the curated safety test set",
    },
    {
      key: "latency",
      label: "End-to-End Latency",
      value: NOT_MEASURED,
      hint: `Retrieval stage measured at ${retrievalLatency} mean`,
    },
  ];

  return (
    <div className="mx-auto max-w-7xl space-y-6">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h2 className="text-lg font-semibold tracking-tight text-foreground">
            {ENVIRONMENT_LABEL}
          </h2>
          <p className="mt-0.5 text-sm text-muted-foreground">{ENVIRONMENT_SUBTITLE}</p>
        </div>
        <Pill tone="warning">{ENVIRONMENT_LABEL}</Pill>
      </div>

      <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-3">
        {kpis.map((kpi) => (
          <Stat
            key={kpi.key}
            label={kpi.label}
            value={kpi.value}
            hint={kpi.value === NOT_MEASURED ? `Not measured — ${kpi.hint}` : kpi.hint}
          />
        ))}
      </div>

      <Panel
        title="Curated Demo Scenarios"
        description="Curated evaluation/demo scenarios — not live production traffic."
        action={
          <Link
            to="/analyze"
            className="inline-flex items-center gap-1 text-xs font-medium text-brand hover:underline"
          >
            Open analyzer <ArrowUpRight className="size-3.5" />
          </Link>
        }
        className="overflow-hidden"
      >
        <div className="-m-5 overflow-x-auto">
          <table className="w-full min-w-[820px] text-sm">
            <thead>
              <tr className="border-b border-border text-left text-xs text-muted-foreground">
                <th className="px-5 py-2.5 font-medium">Scenario</th>
                <th className="px-3 py-2.5 font-medium">Customer Query</th>
                <th className="px-3 py-2.5 font-medium">Intent</th>
                <th className="px-3 py-2.5 font-medium">Expected Decision</th>
                <th className="px-3 py-2.5 font-medium">Expected Risk</th>
                <th className="px-5 py-2.5 text-right font-medium">Action</th>
              </tr>
            </thead>
            <tbody>
              {CURATED_SCENARIOS.map((sc) => (
                <tr key={sc.id} className="border-b border-border/60 last:border-0 hover:bg-accent/40">
                  <td className="px-5 py-3 font-medium text-foreground">{sc.label}</td>
                  <td className="max-w-[280px] truncate px-3 py-3 text-muted-foreground">{sc.query}</td>
                  <td className="px-3 py-3 font-mono text-xs text-muted-foreground">{sc.intent}</td>
                  <td className="px-3 py-3">
                    <DecisionBadge decision={sc.expected_decision} />
                  </td>
                  <td className="px-3 py-3">
                    <RiskBadge risk={sc.expected_risk} />
                  </td>
                  <td className="px-5 py-3 text-right">
                    <Link
                      to="/analyze"
                      search={{ q: sc.query }}
                      className="inline-flex items-center gap-1 text-xs font-medium text-brand hover:underline"
                    >
                      <PlayCircle className="size-3.5" /> Run in Pipeline
                    </Link>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Panel>
    </div>
  );
}
