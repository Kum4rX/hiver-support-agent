/**
 * Single source of truth for every metric rendered in the console.
 *
 * All values below come from the offline evaluation harness. Anything that has
 * not actually been measured is represented as `NOT_MEASURED` and must be
 * rendered as "Not measured" — never replaced with an invented number.
 *
 * When the FastAPI backend exposes an evaluation endpoint, this module is the
 * only file that needs to change: the UI reads exclusively from here.
 */

export const NOT_MEASURED = "Not measured" as const;
export type Measured<T> = T | typeof NOT_MEASURED;

export const ENVIRONMENT_LABEL = "Evaluation / Demo Environment";
export const ENVIRONMENT_SUBTITLE =
  "Metrics shown here come from the offline evaluation harness, not production traffic.";

/* ---------------------------------------------------------------- */
/* Intent taxonomy — 11 classes                                       */
/* ---------------------------------------------------------------- */

export const INTENT_CLASSES = [
  "BATTERY_POWER",
  "CONNECTIVITY",
  "CALLS_COMMUNICATION",
  "DEVICE_PERFORMANCE",
  "KEYBOARD_INPUT",
  "APPS_MEDIA",
  "DISPLAY_AUDIO_CAMERA",
  "ACCOUNT_ICLOUD",
  "PURCHASE_PAYMENT",
  "HOW_TO_OTHER",
  "SECURITY",
] as const;

export type IntentClass = (typeof INTENT_CLASSES)[number];

/* ---------------------------------------------------------------- */
/* Evaluation metrics                                                 */
/* ---------------------------------------------------------------- */

export const INTENT_EVAL = {
  label: "Provisional — 11 human-labelled examples",
  sampleSize: 11,
  caveat:
    "With only 11 labelled examples, these metrics are directional rather than definitive.",
  metrics: [
    { label: "Accuracy", value: 0.9091 },
    { label: "Macro F1", value: 0.48 },
    { label: "Weighted F1", value: 0.8658 },
  ],
} as const;

export const SAFETY_EVAL = {
  title: "Curated Safety Test Set",
  note: "Safety results are measured on a curated test set and do not represent production incident rates.",
  safetyRecall: 1.0,
  hazardsCaught: 13,
  hazardsTotal: 13,
  falseNegatives: 0,
  specificity: 1.0,
} as const;

export const RETRIEVAL_EVAL = {
  corpusSize: 65239,
  embeddingModel: "sentence-transformers/all-MiniLM-L6-v2",
  dimensions: 384,
  indexType: "Dense Vector Similarity (IndexFlatIP)",
  topK: 3,
  minSimilarity: 0.55,
  meanTop1Similarity: 0.751,
  meanTop3Similarity: 0.7318,
  meanLatencyMs: 58.85,
  thresholdHitRate: 1.0,
  thresholdNote:
    "All evaluated queries exceeded the predefined similarity threshold. This is not equivalent to human-judged relevance.",
} as const;

export const GUARDRAIL_EVAL = {
  characterLimitCompliance: 1.0 as Measured<number>,
  piiSafety: 1.0 as Measured<number>,
  actionability: 0.571 as Measured<number>,
  llmAsAJudge: NOT_MEASURED as Measured<number>,
} as const;

/* ---------------------------------------------------------------- */
/* Overview KPIs — evaluation/demo only, never production traffic     */
/* ---------------------------------------------------------------- */

export interface OverviewKpi {
  key: string;
  label: string;
  value: string;
  hint: string;
}

export const OVERVIEW_KPIS: OverviewKpi[] = [
  {
    key: "runs",
    label: "Evaluation Runs",
    value: NOT_MEASURED,
    hint: "Run counts are not tracked by the offline harness",
  },
  {
    key: "safety-tests",
    label: "Curated Safety Tests",
    value: `${SAFETY_EVAL.hazardsTotal}`,
    hint: `${SAFETY_EVAL.hazardsCaught} / ${SAFETY_EVAL.hazardsTotal} hazards caught`,
  },
  {
    key: "golden",
    label: "Human-Labelled Golden Examples",
    value: `${INTENT_EVAL.sampleSize}`,
    hint: "Provisional intent evaluation set",
  },
  {
    key: "corpus",
    label: "Retrieval Corpus",
    value: RETRIEVAL_EVAL.corpusSize.toLocaleString("en-US"),
    hint: "Historical support conversations indexed in FAISS",
  },
  {
    key: "recall",
    label: "Safety Recall",
    value: `${(SAFETY_EVAL.safetyRecall * 100).toFixed(0)}%`,
    hint: "Measured on the curated safety test set",
  },
  {
    key: "latency",
    label: "End-to-End Latency",
    value: NOT_MEASURED,
    hint: `Retrieval stage measured at ${RETRIEVAL_EVAL.meanLatencyMs} ms mean`,
  },
];
