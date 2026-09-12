import { createFileRoute } from "@tanstack/react-router";
import { Info, ShieldCheck, Database, MessageSquareText, Cpu } from "lucide-react";
import { Panel, Pill, Meter } from "@/components/primitives";
import {
  GUARDRAIL_EVAL,
  INTENT_EVAL,
  NOT_MEASURED,
  RETRIEVAL_EVAL,
  SAFETY_EVAL,
} from "@/services/demoData";

export const Route = createFileRoute("/evaluation")({
  head: () => ({
    meta: [
      { title: "Evaluation — AI Support Agent Console" },
      {
        name: "description",
        content:
          "Provisional intent metrics, curated safety-recall results, retrieval similarity and guardrail compliance for the AI support agent.",
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
          <Pill>{NOT_MEASURED}</Pill>
        )}
      </div>
      {measured && <Meter value={value} tone={value === 1 ? "success" : "warning"} />}
      <p className="mt-1 text-[11px] text-muted-foreground">{detail}</p>
    </div>
  );
}

function EvaluationPage() {
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
        description={INTENT_EVAL.label}
        action={<Pill tone="warning">Provisional</Pill>}
      >
        <div className="space-y-4">
          {INTENT_EVAL.metrics.map((m) => (
            <div key={m.label}>
              <div className="mb-1.5 flex items-center justify-between text-xs">
                <span className="flex items-center gap-1.5 text-muted-foreground">
                  <Cpu className="size-3.5" /> {m.label}
                </span>
                <span className="font-medium tabular-nums text-foreground">{pct(m.value)}</span>
              </div>
              <Meter value={m.value} tone="brand" />
            </div>
          ))}
        </div>
        <p className="mt-4 text-xs text-muted-foreground">{INTENT_EVAL.caveat}</p>
      </Panel>

      <Panel
        title="Escalation safety"
        description={SAFETY_EVAL.title}
        action={<Pill tone="success">{pct(SAFETY_EVAL.safetyRecall)} Safety Recall</Pill>}
      >
        <div className="grid gap-4 sm:grid-cols-3">
          {[
            {
              label: "Safety Recall",
              value: pct(SAFETY_EVAL.safetyRecall),
              hint: `${SAFETY_EVAL.hazardsCaught} / ${SAFETY_EVAL.hazardsTotal} hazards escalated`,
            },
            {
              label: "Specificity",
              value: pct(SAFETY_EVAL.specificity),
              hint: "Non-hazards correctly left un-escalated",
            },
            {
              label: "False Negatives",
              value: `${SAFETY_EVAL.falseNegatives}`,
              hint: "No missed safety hazard",
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
        <p className="mt-4 text-xs text-muted-foreground">{SAFETY_EVAL.note}</p>
      </Panel>

      <Panel
        title="Retrieval"
        description={`FAISS dense-vector search over ${RETRIEVAL_EVAL.corpusSize.toLocaleString("en-US")} historical cases`}
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
                  RETRIEVAL_EVAL.meanTop1Similarity.toFixed(4),
                  "Cosine similarity on all-MiniLM-L6-v2 embeddings",
                ],
                [
                  "Mean Top-3 similarity",
                  RETRIEVAL_EVAL.meanTop3Similarity.toFixed(4),
                  "Averaged over the three nearest neighbours",
                ],
                [
                  "Mean retrieval latency",
                  `${RETRIEVAL_EVAL.meanLatencyMs} ms`,
                  "Retrieval stage only, measured by the evaluation harness",
                ],
                [
                  "Similarity Threshold Hit Rate",
                  pct(RETRIEVAL_EVAL.thresholdHitRate),
                  RETRIEVAL_EVAL.thresholdNote,
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
          Embedding similarity measures vector closeness, not human-judged relevance. A high score
          means the retrieved case is textually similar; a human relevance study is still pending.
        </p>
      </Panel>

      <Panel title="Response guardrails" description="Post-generation checks applied to every reply">
        <div className="space-y-4">
          <GuardrailRow
            label="Character Limit Compliance (≤280 characters)"
            value={GUARDRAIL_EVAL.characterLimitCompliance}
            detail="Generated responses within the 280-character channel limit"
          />
          <GuardrailRow
            label="PII Safety"
            value={GUARDRAIL_EVAL.piiSafety}
            detail="No identifiers leaked in the evaluation set"
          />
          <GuardrailRow
            label="Actionability"
            value={GUARDRAIL_EVAL.actionability}
            detail="Responses containing at least one concrete next step, per the latest evaluation run"
          />
          <GuardrailRow
            label="LLM-as-a-Judge"
            value={GUARDRAIL_EVAL.llmAsAJudge}
            detail="No LLM judging has been run against this evaluation set"
          />
        </div>
      </Panel>
    </div>
  );
}
