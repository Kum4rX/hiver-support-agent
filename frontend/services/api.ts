/**
 * Real API client connecting directly to the Hiver FastAPI backend.
 *
 * All AI pipeline operations, FAISS retrieval searches, and evaluation metrics
 * are fetched strictly from the live backend. There are NO silent local fallbacks
 * or fabricated responses.
 */

import type { AnalysisResult, KnowledgeDoc, EvaluationResponse } from "./types";
export * from "./types";
export { EXAMPLE_QUERIES, CURATED_ESCALATIONS as ESCALATIONS, CURATED_SCENARIOS } from "./curatedDemoScenarios";

const API_BASE_URL =
  (typeof import.meta !== "undefined" && import.meta.env && import.meta.env.VITE_API_BASE_URL) ||
  "http://localhost:8000";

/**
 * Analyze an inbound customer support tweet using the real FastAPI backend pipeline.
 * Throws an error if the backend is unreachable or returns an error status.
 */
export async function analyzeCustomerMessage(message: string): Promise<AnalysisResult> {
  let res: Response;
  try {
    res = await fetch(`${API_BASE_URL}/api/analyze`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message }),
    });
  } catch (networkError) {
    throw new Error("Backend unavailable — start FastAPI at http://localhost:8000.");
  }

  if (!res.ok) {
    const errorBody = await res.text().catch(() => "");
    throw new Error(
      `FastAPI returned error status ${res.status}: ${errorBody || res.statusText || "Unable to analyze message"}`
    );
  }

  return (await res.json()) as AnalysisResult;
}

/** @deprecated use analyzeCustomerMessage */
export const analyzeTicket = analyzeCustomerMessage;

/**
 * Perform dense vector semantic search over the 65,239 historical AppleSupport FAISS index.
 * Throws an error if the backend is unreachable or returns an error status.
 */
export async function searchKnowledgeBase(query: string, k: number = 6): Promise<KnowledgeDoc[]> {
  const trimmed = query.trim();
  if (!trimmed) {
    return [];
  }

  let res: Response;
  try {
    res = await fetch(`${API_BASE_URL}/api/search?query=${encodeURIComponent(trimmed)}&k=${k}`, {
      method: "GET",
      headers: { "Content-Type": "application/json" },
    });
  } catch (networkError) {
    throw new Error("Backend unavailable — start FastAPI at http://localhost:8000.");
  }

  if (!res.ok) {
    throw new Error(`FastAPI returned error status ${res.status}: Unable to search FAISS index.`);
  }

  const data = await res.json();
  return (data.results || []) as KnowledgeDoc[];
}

/**
 * Fetch measured offline benchmark evaluation metrics from the backend.
 * Single source of truth for evaluation metrics.
 */
export async function getEvaluationMetrics(): Promise<EvaluationResponse> {
  let res: Response;
  try {
    res = await fetch(`${API_BASE_URL}/api/evaluation`, {
      method: "GET",
      headers: { "Content-Type": "application/json" },
    });
  } catch (networkError) {
    throw new Error("Backend unavailable — start FastAPI at http://localhost:8000.");
  }

  if (!res.ok) {
    throw new Error(`FastAPI returned error status ${res.status}: Unable to retrieve evaluation metrics.`);
  }

  return (await res.json()) as EvaluationResponse;
}
