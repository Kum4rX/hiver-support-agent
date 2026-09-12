/**
 * Presentation constants for the Hiver Support Agent console.
 *
 * All measured evaluation numbers, retrieval stats, and pipeline metrics are fetched
 * dynamically from the live FastAPI backend (GET /api/evaluation) as the single source
 * of truth.
 */

export const NOT_MEASURED = "Not measured" as const;
export type Measured<T> = T | typeof NOT_MEASURED;

export const ENVIRONMENT_LABEL = "Evaluation / Demo Environment";
export const ENVIRONMENT_SUBTITLE =
  "Metrics shown here come from the evaluation harness via FastAPI, not simulated traffic.";

/* ---------------------------------------------------------------- */
/* Intent taxonomy — 11 classes                                     */
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
