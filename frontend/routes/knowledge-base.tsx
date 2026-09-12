import { createFileRoute } from "@tanstack/react-router";
import { useState, useEffect } from "react";
import { Search, Loader2, Database, CircleCheck, ShieldAlert, RefreshCw } from "lucide-react";
import { Panel, Pill, Meter } from "@/components/primitives";
import { searchKnowledgeBase, getEvaluationMetrics, type KnowledgeDoc } from "@/services/api";
import type { RetrievalEvaluationMetrics } from "@/services/types";

export const Route = createFileRoute("/knowledge-base")({
  head: () => ({
    meta: [
      { title: "Knowledge Base — AI Support Agent Console" },
      {
        name: "description",
        content:
          "Search the FAISS retrieval index of 65,239 historical support conversations used to ground every agent response.",
      },
      { property: "og:title", content: "Knowledge Base — AI Support Agent Console" },
      {
        property: "og:description",
        content: "Semantic search over the historical support corpus behind the agent.",
      },
    ],
  }),
  component: KnowledgeBasePage,
});

export function KnowledgeBasePage() {
  const [query, setQuery] = useState("");
  const [loading, setLoading] = useState(false);
  const [searched, setSearched] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [results, setResults] = useState<KnowledgeDoc[]>([]);
  const [retrievalStats, setRetrievalStats] = useState<RetrievalEvaluationMetrics>({
    corpus_size: 65239,
    embedding_model: "sentence-transformers/all-MiniLM-L6-v2",
    dimensions: 384,
    index_type: "Dense Vector Similarity (IndexFlatIP)",
    top_k: 3,
    mean_top1_similarity: 0.751,
    mean_top3_similarity: 0.7318,
    threshold_hit_rate: 1.0,
    mean_latency_ms: 93.73,
    threshold_note: "All evaluated queries exceeded the similarity threshold.",
  });

  useEffect(() => {
    getEvaluationMetrics()
      .then((data) => {
        if (data.retrieval_evaluation) {
          setRetrievalStats(data.retrieval_evaluation);
        }
      })
      .catch(() => {
        // Keeps verified defaults if backend hasn't answered yet
      });
  }, []);

  async function search(e?: React.FormEvent) {
    if (e) e.preventDefault();
    if (!query.trim()) return;
    setLoading(true);
    setError(null);
    try {
      const data = await searchKnowledgeBase(query.trim(), 6);
      setResults(data);
      setSearched(true);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : "Backend unavailable — start FastAPI at http://localhost:8000.";
      setError(msg);
      setResults([]);
      setSearched(true);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="mx-auto max-w-5xl space-y-6">
      <Panel
        title="FAISS retrieval index"
        description="Dense-vector index backing every grounded response"
        action={
          <Pill tone="success">
            <CircleCheck className="size-3.5" /> Ready
          </Pill>
        }
      >
        <dl className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          {[
            ["Documents", retrievalStats.corpus_size.toLocaleString("en-US")],
            ["Embedding model", retrievalStats.embedding_model],
            ["Index type", retrievalStats.index_type],
            ["Dimensions", `${retrievalStats.dimensions}`],
            ["Top-K", `${retrievalStats.top_k}`],
            ["Minimum similarity", "≥ 0.35"],
          ].map(([k, v]) => (
            <div key={k} className="rounded-lg border border-border p-3.5">
              <dt className="text-[11px] uppercase tracking-wide text-muted-foreground">{k}</dt>
              <dd className="mt-1 text-sm font-medium text-foreground">{v}</dd>
            </div>
          ))}
        </dl>
        <p className="mt-4 text-xs text-muted-foreground">
          Historical support conversations are indexed in FAISS and retrieved as factual grounding context.
        </p>
      </Panel>

      <Panel title="Semantic search" description="Query the historical support corpus directly via FAISS">
        <form onSubmit={search} className="flex flex-col gap-2 sm:flex-row">
          <div className="relative flex-1">
            <Search className="pointer-events-none absolute left-3 top-1/2 size-4 -translate-y-1/2 text-muted-foreground" />
            <input
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="e.g. battery drains overnight after update"
              className="h-10 w-full rounded-md border border-border bg-background pl-9 pr-3 text-sm text-foreground outline-none transition placeholder:text-muted-foreground focus:border-ring focus:ring-2 focus:ring-ring/25"
            />
          </div>
          <button
            type="submit"
            disabled={loading || !query.trim()}
            className="inline-flex h-10 items-center justify-center gap-2 rounded-md bg-brand px-4 text-sm font-medium text-brand-foreground transition hover:opacity-90 disabled:opacity-50"
          >
            {loading ? <Loader2 className="size-4 animate-spin" /> : <Database className="size-4" />}
            Search
          </button>
        </form>

        {error && (
          <div className="mt-5 rounded-xl border border-destructive/30 bg-destructive/5 p-6 text-center">
            <ShieldAlert className="mx-auto size-7 text-destructive" />
            <p className="mt-2.5 text-sm font-semibold text-foreground">{error}</p>
            <p className="mt-1 text-xs text-muted-foreground">
              Please start the FastAPI backend: <code className="rounded bg-muted px-1.5 py-0.5 font-mono text-xs text-foreground">python -m uvicorn backend.main:app --port 8000</code>
            </p>
            <button
              type="button"
              onClick={() => void search()}
              className="mt-3.5 inline-flex items-center gap-1.5 rounded-md bg-destructive px-3 py-1.5 text-xs font-medium text-destructive-foreground transition hover:opacity-90"
            >
              <RefreshCw className="size-3.5" /> Retry Search
            </button>
          </div>
        )}

        {!searched && !loading && !error && (
          <div className="mt-5 rounded-xl border border-dashed border-border bg-card/40 p-10 text-center">
            <Database className="mx-auto size-6 text-muted-foreground" />
            <p className="mt-2.5 text-sm font-medium text-foreground">Search the index to load results</p>
            <p className="mt-1 text-xs text-muted-foreground">
              Query the 65,239 pre-filtered historical AppleSupport conversations via FAISS.
            </p>
          </div>
        )}

        {searched && !loading && !error && results.length === 0 && (
          <div className="mt-5 rounded-xl border border-dashed border-border bg-card/40 p-8 text-center">
            <p className="text-sm font-medium text-foreground">No matching support cases found</p>
            <p className="mt-1 text-xs text-muted-foreground">
              Try broader keywords or a different phrasing.
            </p>
          </div>
        )}

        <div className="mt-5 space-y-4">
          {results.map((doc) => (
            <article key={doc.id} className="rounded-lg border border-border p-4">
              <div className="flex flex-wrap items-center justify-between gap-2">
                <div className="flex flex-wrap items-center gap-1.5">
                  {doc.tags.map((t) => (
                    <Pill key={t}>{t}</Pill>
                  ))}
                </div>
                <span className="text-xs tabular-nums text-muted-foreground">
                  similarity {doc.similarity.toFixed(4)}
                </span>
              </div>
              <Meter className="mt-2" value={doc.similarity} tone="violet" />
              <div className="mt-3 grid gap-3 md:grid-cols-2">
                <div>
                  <div className="text-[11px] font-medium uppercase tracking-wide text-muted-foreground">
                    Customer issue
                  </div>
                  <p className="mt-1 text-sm text-foreground">{doc.customer_text}</p>
                </div>
                <div>
                  <div className="text-[11px] font-medium uppercase tracking-wide text-muted-foreground">
                    Support response
                  </div>
                  <p className="mt-1 text-sm text-foreground">{doc.support_text}</p>
                </div>
              </div>
            </article>
          ))}
        </div>
      </Panel>
    </div>
  );
}
