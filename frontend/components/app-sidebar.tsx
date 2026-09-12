import { Link, useRouterState } from "@tanstack/react-router";
import {
  LayoutDashboard,
  ScanSearch,
  AlertTriangle,
  BarChart3,
  Library,
  Settings2,
  CircleDot,
  Bot,
} from "lucide-react";
import { cn } from "@/lib/utils";

const items = [
  { title: "Overview", url: "/", icon: LayoutDashboard },
  { title: "Analyze Ticket", url: "/analyze", icon: ScanSearch },
  { title: "Escalations", url: "/escalations", icon: AlertTriangle },
  { title: "Evaluation", url: "/evaluation", icon: BarChart3 },
  { title: "Knowledge Base", url: "/knowledge-base", icon: Library },
  { title: "Settings", url: "/settings", icon: Settings2 },
] as const;

export function AppSidebar() {
  const pathname = useRouterState({ select: (r) => r.location.pathname });

  return (
    <aside className="hidden w-64 shrink-0 flex-col border-r border-border bg-sidebar md:flex">
      <div className="flex h-16 items-center gap-2.5 border-b border-border px-5">
        <div className="flex size-8 items-center justify-center rounded-lg bg-primary text-primary-foreground">
          <Bot className="size-4" />
        </div>
        <div className="leading-tight">
          <div className="text-sm font-semibold text-foreground">Hiver Support Agent</div>
          <div className="text-[11px] text-muted-foreground">Eval & Operations</div>
        </div>
      </div>

      <nav className="flex-1 space-y-0.5 p-3">
        <div className="px-2 pb-2 text-[11px] font-medium uppercase tracking-wider text-muted-foreground">
          Workspace
        </div>
        {items.map((item) => {
          const active = pathname === item.url;
          return (
            <Link
              key={item.url}
              to={item.url}
              className={cn(
                "flex items-center gap-2.5 rounded-md px-2.5 py-2 text-sm transition-colors",
                active
                  ? "bg-accent font-medium text-accent-foreground"
                  : "text-muted-foreground hover:bg-accent/60 hover:text-foreground",
              )}
            >
              <item.icon className="size-4" />
              {item.title}
            </Link>
          );
        })}
      </nav>

      <div className="m-3 rounded-lg border border-border bg-card p-3 text-xs">
        <div className="flex items-center gap-1.5 font-medium text-success">
          <CircleDot className="size-3.5" />
          All systems operational
        </div>
        <dl className="mt-2.5 space-y-1.5 text-muted-foreground">
          <div className="flex justify-between gap-2">
            <dt>Model</dt>
            <dd className="text-right font-medium text-foreground">Hybrid Intent Classifier</dd>
          </div>
          <div className="flex justify-between gap-2">
            <dt>Retrieval</dt>
            <dd className="font-medium text-foreground">FAISS</dd>
          </div>
        </dl>
      </div>
    </aside>
  );
}
