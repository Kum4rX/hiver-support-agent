import { useRouterState } from "@tanstack/react-router";
import { Search, FlaskConical } from "lucide-react";
import type { ReactNode } from "react";
import { AppSidebar } from "./app-sidebar";

const TITLES: Record<string, string> = {
  "/": "Overview",
  "/analyze": "Analyze Ticket",
  "/escalations": "Escalations",
  "/evaluation": "Evaluation",
  "/knowledge-base": "Knowledge Base",
  "/settings": "Settings",
};

export function AppShell({ children }: { children: ReactNode }) {
  const pathname = useRouterState({ select: (r) => r.location.pathname });

  return (
    <div className="flex min-h-screen w-full bg-background">
      <AppSidebar />
      <div className="flex min-w-0 flex-1 flex-col">
        <header className="sticky top-0 z-20 flex h-16 items-center gap-4 border-b border-border bg-background/85 px-5 backdrop-blur">
          <h1 className="shrink-0 text-sm font-semibold text-foreground">
            {TITLES[pathname] ?? "Support Agent"}
          </h1>

          <div className="relative hidden min-w-0 flex-1 md:block">
            <Search className="pointer-events-none absolute left-3 top-1/2 size-4 -translate-y-1/2 text-muted-foreground" />
            <input
              type="search"
              placeholder="Search tickets, intents, corpus…"
              className="h-9 w-full max-w-md rounded-md border border-border bg-card pl-9 pr-16 text-sm text-foreground outline-none transition placeholder:text-muted-foreground focus:border-ring focus:ring-2 focus:ring-ring/25"
            />
            <kbd className="pointer-events-none absolute left-[calc(min(100%,28rem)-3.2rem)] top-1/2 hidden -translate-y-1/2 rounded border border-border bg-muted px-1.5 py-0.5 text-[10px] text-muted-foreground lg:block">
              ⌘K
            </kbd>
          </div>

          <div className="ml-auto flex items-center gap-3">
            <span className="hidden items-center gap-1.5 rounded-full border border-warning/30 bg-warning/10 px-2.5 py-1 text-[11px] font-medium text-warning sm:inline-flex">
              <FlaskConical className="size-3.5" />
              Evaluation / Demo Environment
            </span>
            <div className="flex items-center gap-2 rounded-md border border-border bg-card px-3 py-1.5 text-xs font-medium text-foreground">
              <span className="size-1.5 rounded-full bg-primary" />
              <span>Hiver AI Support Agent</span>
            </div>
          </div>
        </header>

        <main className="min-w-0 flex-1 p-5 lg:p-7">{children}</main>
      </div>
    </div>
  );
}
