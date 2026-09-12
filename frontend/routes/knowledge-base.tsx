import { createFileRoute } from "@tanstack/react-router";
import { useState } from "react";
import { Search, Loader2, Database, CircleCheck } from "lucide-react";
import { Panel, Pill, Meter } from "@/components/primitives";
import { searchKnowledgeBase, KNOWLEDGE_CORPUS, type KnowledgeDoc } from "@/services/api";
import { RETRIEVAL_EVAL } from "@/services/demoData";

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

function KnowledgeBasePage() {
  const [query, setQuery] = useState("");
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState<KnowledgeDoc[]>(KNOWLEDGE_CORPUS.slice(0, 5));

  async function search(e: React.FormEvent) {
    e.preventDefault();
    setLoading(true);
    try {
      setResults(await searchKnowledgeBase(query));
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
            ["Documents", RETRIEVAL_EVAL.corpusSize.toLocaleString("en-US")],
            ["Embedding model", RETRIEVAL_EVAL.embeddingModel],
            ["Index type", RETRIEVAL_EVAL.indexType],
            ["Dimensions", `${RETRIEVAL_EVAL.dimensions}`],
            ["Top-K", `${RETRIEVAL_EVAL.topK}`],
            ["Minimum similarity", `${RETRIEVAL_EVAL.minSimilarity}`],
          ].map(([k, v]) => (
            <div key={k} className="rounded-lg border border-border p-3.5">
              <dt className="text-[11px] uppercase tracking-wide text-muted-foreground">{k}</dt>
              <dd className="mt-1 text-sm font-medium text-foreground">{v}</dd>
            </div>
          ))}
        </dl>
        <p className="mt-4 text-xs text-muted-foreground">
          Historical support conversations are used as retrieval context for grounded responses.
        </p>
      </Panel>

      <Panel title="Semantic search" description="Query the historical support corpus directly">
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
            disabled={loading}
            className="inline-flex h-10 items-center justify-center gap-2 rounded-md bg-brand px-4 text-sm font-medium text-brand-foreground transition hover:opacity-90 disabled:opacity-50"
          >
            {loading ? <Loader2 className="size-4 animate-spin" /> : <Database className="size-4" />}
            Search
          </button>
        </form>

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
                  similarity {doc.similarity.toFixed(2)}
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
