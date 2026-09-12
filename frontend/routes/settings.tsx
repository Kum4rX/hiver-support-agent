import { createFileRoute } from "@tanstack/react-router";
import { Lock } from "lucide-react";
import { Panel, Pill } from "@/components/primitives";
import { INTENT_CLASSES } from "@/services/demoData";

export const Route = createFileRoute("/settings")({
  head: () => ({
    meta: [
      { title: "Settings — AI Support Agent Console" },
      {
        name: "description",
        content:
          "Read-only configuration for the hybrid rule + TF-IDF intent classifier, embedding model, FAISS index, response generation and guardrails.",
      },
      { property: "og:title", content: "Settings — AI Support Agent Console" },
      {
        property: "og:description",
        content: "Read-only view of the deployed agent configuration.",
      },
    ],
  }),
  component: SettingsPage,
});

const groups: { title: string; description: string; rows: [string, string][] }[] = [
  {
    title: "Intent classifier",
    description: "Hybrid Rule + TF-IDF classifier",
    rows: [
      ["Architecture", "Hybrid Rule + TF-IDF classifier (rule taxonomy engine with TF-IDF statistical fallback)"],
      ["Intent classes", "11"],
      ["Confidence threshold", "0.65"],
      ["Multi-intent detection", "Enabled — primary routes, secondary retained"],
    ],
  },
  {
    title: "Embedding model",
    description: "Used for retrieval over the historical corpus",
    rows: [
      ["Model", "sentence-transformers/all-MiniLM-L6-v2"],
      ["Dimensions", "384"],
      ["Normalisation", "L2 normalised"],
      ["Max sequence length", "256 tokens"],
    ],
  },
  {
    title: "FAISS index",
    description: "Historical support corpus",
    rows: [
      ["Index type", "Dense Vector Similarity (IndexFlatIP)"],
      ["Documents", "65,239"],
      ["Top-K retrieved", "3"],
      ["Minimum similarity", "≥ 0.35"],
    ],
  },
  {
    title: "Response generation",
    description: "Grounded on retrieved cases",
    rows: [
      ["Strategy", "Retrieval-grounded template synthesis"],
      ["Character limit", "280 (Twitter/X)"],
      ["Safety override", "Deterministic template, bypasses generation"],
      ["Out-of-domain", "Fixed boundary message, no retrieval"],
    ],
  },
  {
    title: "Guardrails",
    description: "Applied post-generation to every reply",
    rows: [
      ["Length check", "Reject responses over 280 characters"],
      ["PII scan", "Email, phone, order ID and serial patterns"],
      ["Actionability check", "Requires at least one concrete next step"],
      ["LLM-as-a-Judge", "Not measured"],
      ["Failure behaviour", "Fall back to human review"],
    ],
  },
];

function SettingsPage() {
  return (
    <div className="mx-auto max-w-4xl space-y-6">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h2 className="text-lg font-semibold tracking-tight text-foreground">Configuration</h2>
          <p className="mt-0.5 text-sm text-muted-foreground">
            Values are deployed with the service and cannot be edited from this console.
          </p>
        </div>
        <Pill>
          <Lock className="size-3.5" /> Read-only
        </Pill>
      </div>

      {groups.map((g) => (
        <Panel key={g.title} title={g.title} description={g.description}>
          <dl className="divide-y divide-border/60">
            {g.rows.map(([k, v]) => (
              <div key={k} className="flex flex-col gap-1 py-3 first:pt-0 last:pb-0 sm:flex-row sm:justify-between sm:gap-6">
                <dt className="text-xs text-muted-foreground">{k}</dt>
                <dd className="text-sm text-foreground sm:max-w-[60%] sm:text-right">{v}</dd>
              </div>
            ))}
          </dl>
        </Panel>
      ))}

      <Panel title="Intent taxonomy" description="11 classes used by the classifier and router">
        <ol className="grid gap-2 sm:grid-cols-2">
          {INTENT_CLASSES.map((c, i) => (
            <li
              key={c}
              className="flex items-center gap-2 rounded-md border border-border px-3 py-2 text-sm"
            >
              <span className="w-5 shrink-0 text-xs tabular-nums text-muted-foreground">{i + 1}</span>
              <span className="font-mono text-xs text-foreground">{c}</span>
            </li>
          ))}
        </ol>
      </Panel>
    </div>
  );
}
