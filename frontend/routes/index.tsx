import { createFileRoute, Link } from "@tanstack/react-router";
import { ArrowUpRight } from "lucide-react";
import { Panel, Pill, Stat, RiskBadge, DecisionBadge } from "@/components/primitives";
import { RECENT_RUNS } from "@/services/api";
import {
  ENVIRONMENT_LABEL,
  ENVIRONMENT_SUBTITLE,
  NOT_MEASURED,
  OVERVIEW_KPIS,
} from "@/services/demoData";


export const Route = createFileRoute("/")({
  head: () => ({
    meta: [
      { title: "Overview — AI Support Agent Console" },
      {
        name: "description",
        content:
          "Live KPIs for the AI customer support agent: auto-resolution rate, escalations, safety recall, latency and retrieval corpus size.",
      },
      { property: "og:title", content: "Overview — AI Support Agent Console" },
      {
        property: "og:description",
        content: "Auto-resolution, escalations, safety recall and latency at a glance.",
      },
    ],
  }),
  component: Overview,
});

function Overview() {
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
        {OVERVIEW_KPIS.map((kpi) => (
          <Stat
            key={kpi.key}
            label={kpi.label}
            value={kpi.value}
            hint={kpi.value === NOT_MEASURED ? `Not measured — ${kpi.hint}` : kpi.hint}
          />
        ))}
      </div>

      <Panel
        title="Demo / Evaluation Runs"
        description="Scenario tickets processed by the pipeline — not live production traffic"

        action={
          <Link
            to="/analyze"
            className="inline-flex items-center gap-1 text-xs font-medium text-brand hover:underline"
          >
            New analysis <ArrowUpRight className="size-3.5" />
          </Link>
        }
        className="overflow-hidden"
      >
        <div className="-m-5 overflow-x-auto">
          <table className="w-full min-w-[820px] text-sm">
            <thead>
              <tr className="border-b border-border text-left text-xs text-muted-foreground">
                <th className="px-5 py-2.5 font-medium">Query</th>
                <th className="px-3 py-2.5 font-medium">Intent</th>
                <th className="px-3 py-2.5 font-medium">Decision</th>
                <th className="px-3 py-2.5 font-medium">Risk</th>
                <th className="px-3 py-2.5 font-medium">Conf.</th>
                <th className="px-3 py-2.5 font-medium">Latency</th>
                <th className="px-3 py-2.5 font-medium">When</th>
                <th className="px-5 py-2.5" />
              </tr>
            </thead>
            <tbody>
              {RECENT_RUNS.map((run) => (
                <tr key={run.id} className="border-b border-border/60 last:border-0 hover:bg-accent/40">
                  <td className="max-w-[280px] truncate px-5 py-3 text-foreground">{run.query}</td>
                  <td className="px-3 py-3 font-mono text-xs text-muted-foreground">{run.intent}</td>
                  <td className="px-3 py-3">
                    <DecisionBadge decision={run.decision} />
                  </td>
                  <td className="px-3 py-3">
                    <RiskBadge risk={run.risk_level} />
                  </td>
                  <td className="px-3 py-3 tabular-nums text-muted-foreground">
                    {(run.confidence * 100).toFixed(0)}%
                  </td>
                  <td className="px-3 py-3 tabular-nums text-muted-foreground">{run.latency_ms} ms</td>
                  <td className="px-3 py-3 text-muted-foreground">{run.created_at}</td>
                  <td className="px-5 py-3 text-right">
                    <Link
                      to="/analyze"
                      search={{ q: run.query }}
                      className="text-xs font-medium text-brand hover:underline"
                    >
                      Inspect
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
