import { createFileRoute } from "@tanstack/react-router";
import { useMemo, useState } from "react";
import { Filter, X } from "lucide-react";
import { Panel, Pill, RiskBadge } from "@/components/primitives";
import { ESCALATIONS, type EscalationRecord, type RiskLevel } from "@/services/api";
import { cn } from "@/lib/utils";

export const Route = createFileRoute("/escalations")({
  head: () => ({
    meta: [
      { title: "Escalations — AI Support Agent Console" },
      {
        name: "description",
        content:
          "Review every conversation the AI support agent escalated: risk level, escalation reason, routing target and review status.",
      },
      { property: "og:title", content: "Escalations — AI Support Agent Console" },
      {
        property: "og:description",
        content: "Escalated conversations with risk, reason and routing detail.",
      },
    ],
  }),
  component: EscalationsPage,
});

const FILTERS = ["all", "critical", "high", "medium"] as const;

function statusTone(status: EscalationRecord["status"]) {
  return status === "resolved" ? "success" : status === "in_review" ? "warning" : "brand";
}

function EscalationsPage() {
  const [filter, setFilter] = useState<(typeof FILTERS)[number]>("all");
  const [selected, setSelected] = useState<EscalationRecord | null>(null);

  const rows = useMemo(
    () => (filter === "all" ? ESCALATIONS : ESCALATIONS.filter((e) => e.risk_level === (filter as RiskLevel))),
    [filter],
  );

  return (
    <div className="mx-auto max-w-7xl space-y-6">
      <Panel
        title="Curated / Demo Escalations"
        description="Demo tickets handed to a human specialist by the escalation policy — not live customers"
        action={
          <div className="flex items-center gap-1 rounded-md border border-border bg-card p-0.5">
            <Filter className="mx-1.5 size-3.5 text-muted-foreground" />
            {FILTERS.map((f) => (
              <button
                key={f}
                type="button"
                onClick={() => setFilter(f)}
                className={cn(
                  "rounded px-2.5 py-1 text-xs font-medium capitalize transition",
                  filter === f
                    ? "bg-accent text-accent-foreground"
                    : "text-muted-foreground hover:text-foreground",
                )}
              >
                {f}
              </button>
            ))}
          </div>
        }
      >
        <div className="-m-5 overflow-x-auto">
          <table className="w-full min-w-[880px] text-sm">
            <thead>
              <tr className="border-b border-border text-left text-xs text-muted-foreground">
                <th className="px-5 py-2.5 font-medium">Case</th>
                <th className="px-3 py-2.5 font-medium">Intent</th>
                <th className="px-3 py-2.5 font-medium">Risk</th>
                <th className="px-3 py-2.5 font-medium">Escalation reason</th>
                <th className="px-3 py-2.5 font-medium">Time</th>
                <th className="px-3 py-2.5 font-medium">Status</th>
                <th className="px-5 py-2.5" />
              </tr>
            </thead>
            <tbody>
              {rows.map((e) => (
                <tr key={e.id} className="border-b border-border/60 last:border-0 hover:bg-accent/40">
                  <td className="px-5 py-3 font-medium text-foreground">{e.customer}</td>
                  <td className="px-3 py-3 font-mono text-xs text-muted-foreground">{e.intent}</td>
                  <td className="px-3 py-3">
                    <RiskBadge risk={e.risk_level} />
                  </td>
                  <td className="max-w-[280px] truncate px-3 py-3 text-muted-foreground">
                    {e.escalation_reason}
                  </td>
                  <td className="px-3 py-3 text-muted-foreground">{e.created_at}</td>
                  <td className="px-3 py-3">
                    <Pill tone={statusTone(e.status)}>{e.status.replace("_", " ")}</Pill>
                  </td>
                  <td className="px-5 py-3 text-right">
                    <button
                      type="button"
                      onClick={() => setSelected(e)}
                      className="text-xs font-medium text-brand hover:underline"
                    >
                      Review
                    </button>
                  </td>
                </tr>
              ))}
              {rows.length === 0 && (
                <tr>
                  <td colSpan={7} className="px-5 py-10 text-center text-sm text-muted-foreground">
                    No escalations match this filter.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </Panel>

      {selected && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-foreground/25 p-4 backdrop-blur-sm"
          onClick={() => setSelected(null)}
        >
          <div
            className="w-full max-w-xl rounded-xl border border-border bg-card p-6 shadow-lg"
            onClick={(ev) => ev.stopPropagation()}
          >
            <div className="flex items-start justify-between gap-4">
              <div>
                <h2 className="text-base font-semibold text-foreground">
                  Escalation {selected.id}
                </h2>
                <p className="text-xs text-muted-foreground">
                  {selected.customer} · {selected.created_at}
                </p>
              </div>
              <button
                type="button"
                onClick={() => setSelected(null)}
                className="rounded-md p-1 text-muted-foreground hover:bg-accent hover:text-foreground"
                aria-label="Close"
              >
                <X className="size-4" />
              </button>
            </div>

            <p className="mt-4 rounded-lg border border-border bg-muted/40 p-3.5 text-sm text-foreground">
              {selected.query}
            </p>

            <dl className="mt-4 space-y-3 text-sm">
              <div className="flex items-center justify-between gap-4">
                <dt className="text-xs text-muted-foreground">Intent</dt>
                <dd className="font-mono text-xs text-foreground">{selected.intent}</dd>
              </div>
              <div className="flex items-center justify-between gap-4">
                <dt className="text-xs text-muted-foreground">Risk level</dt>
                <dd>
                  <RiskBadge risk={selected.risk_level} />
                </dd>
              </div>
              <div className="flex items-start justify-between gap-4">
                <dt className="text-xs text-muted-foreground">Reason</dt>
                <dd className="text-right text-sm text-foreground">{selected.escalation_reason}</dd>
              </div>
              <div className="flex items-center justify-between gap-4">
                <dt className="text-xs text-muted-foreground">Routing target</dt>
                <dd className="text-sm text-foreground">{selected.routing_target}</dd>
              </div>
              <div className="flex items-center justify-between gap-4">
                <dt className="text-xs text-muted-foreground">Source</dt>
                <dd className="text-xs font-medium text-muted-foreground">{selected.source || "Curated safety/evaluation scenario"}</dd>
              </div>
              <div className="flex items-center justify-between gap-4">
                <dt className="text-xs text-muted-foreground">Status</dt>
                <dd>
                  <Pill tone={statusTone(selected.status)}>{selected.status.replace("_", " ")}</Pill>
                </dd>
              </div>
            </dl>
          </div>
        </div>
      )}
    </div>
  );
}
