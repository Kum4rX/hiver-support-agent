import type { ReactNode } from "react";
import { cn } from "@/lib/utils";
import type { AgentDecision, RiskLevel } from "@/services/api";

export function Panel({
  title,
  description,
  action,
  children,
  className,
}: {
  title?: string;
  description?: string;
  action?: ReactNode;
  children: ReactNode;
  className?: string;
}) {
  return (
    <section className={cn("rounded-xl border border-border bg-card shadow-sm", className)}>
      {(title || action) && (
        <header className="flex items-start justify-between gap-3 border-b border-border px-5 py-3.5">
          <div>
            {title && <h2 className="text-sm font-semibold text-foreground">{title}</h2>}
            {description && (
              <p className="mt-0.5 text-xs text-muted-foreground">{description}</p>
            )}
          </div>
          {action}
        </header>
      )}
      <div className="p-5">{children}</div>
    </section>
  );
}

const toneMap = {
  neutral: "border-border bg-muted text-muted-foreground",
  brand: "border-brand/25 bg-brand/10 text-brand",
  success: "border-success/25 bg-success/10 text-success",
  warning: "border-warning/30 bg-warning/10 text-warning",
  danger: "border-danger/30 bg-danger/10 text-danger",
} as const;

export type Tone = keyof typeof toneMap;

export function Pill({
  tone = "neutral",
  children,
  className,
}: {
  tone?: Tone;
  children: ReactNode;
  className?: string;
}) {
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1.5 rounded-full border px-2.5 py-0.5 text-[11px] font-medium",
        toneMap[tone],
        className,
      )}
    >
      {children}
    </span>
  );
}

export function riskTone(risk: RiskLevel): Tone {
  switch (risk) {
    case "critical":
    case "high":
      return "danger";
    case "medium":
      return "warning";
    case "low":
      return "success";
    default:
      return "neutral";
  }
}

export function RiskBadge({ risk }: { risk: RiskLevel }) {
  return <Pill tone={riskTone(risk)}>{risk.toUpperCase()}</Pill>;
}

export const decisionLabel: Record<AgentDecision, string> = {
  auto_response: "Auto-response",
  human_review: "Human Review",
  out_of_domain: "Out-of-Domain",
};

export function DecisionBadge({ decision }: { decision: AgentDecision }) {
  const tone: Tone =
    decision === "auto_response" ? "success" : decision === "human_review" ? "danger" : "warning";
  return <Pill tone={tone}>{decisionLabel[decision]}</Pill>;
}

export function Meter({
  value,
  tone = "brand",
  className,
}: {
  value: number; // 0..1
  tone?: "brand" | "success" | "warning" | "danger" | "violet";
  className?: string;
}) {
  const bg = {
    brand: "bg-brand",
    success: "bg-success",
    warning: "bg-warning",
    danger: "bg-danger",
    violet: "bg-violet",
  }[tone];
  return (
    <div className={cn("h-2 w-full overflow-hidden rounded-full bg-muted", className)}>
      <div
        className={cn("h-full rounded-full transition-all duration-500", bg)}
        style={{ width: `${Math.max(0, Math.min(1, value)) * 100}%` }}
      />
    </div>
  );
}

export function Stat({
  label,
  value,
  hint,
  tone = "neutral",
  icon,
}: {
  label: string;
  value: ReactNode;
  hint?: string;
  tone?: Tone;
  icon?: ReactNode;
}) {
  return (
    <div className="rounded-xl border border-border bg-card p-4 shadow-sm">
      <div className="flex items-center justify-between gap-2">
        <span className="text-xs font-medium text-muted-foreground">{label}</span>
        {icon && <span className={cn("text-muted-foreground", tone !== "neutral" && "")}>{icon}</span>}
      </div>
      <div className="mt-2 text-2xl font-semibold tracking-tight text-foreground">{value}</div>
      {hint && <div className="mt-1 text-[11px] text-muted-foreground">{hint}</div>}
    </div>
  );
}
