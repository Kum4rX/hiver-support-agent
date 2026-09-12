export type RiskLevel = "none" | "low" | "medium" | "high" | "critical";
export type AgentDecision = "auto_response" | "human_review" | "out_of_domain";

export interface RetrievedCase {
  similarity: number; // 0.0 - 1.0
  customer_text: string;
  support_text: string;
}

export interface Guardrails {
  length_ok: boolean;
  pii_safe: boolean;
  actionable: boolean;
}

export interface PipelineStage {
  id: string;
  label: string;
  status: "complete" | "skipped" | "blocked";
  duration_ms: number;
  detail: string;
}

export interface AnalysisResult {
  query: string;
  intent: string;
  secondary_intents: string[];
  confidence: number; // 0.0 - 1.0
  escalated: boolean;
  escalation_reason: string | null;
  risk_level: RiskLevel;
  decision: AgentDecision;
  routing_target: string;
  retrieved_cases: RetrievedCase[];
  response: string;
  response_length: number;
  guardrails: Guardrails;
  latency_ms: number;
  stages: PipelineStage[];
}

export interface EscalationRecord {
  id: string;
  customer: string;
  query: string;
  intent: string;
  risk_level: RiskLevel;
  escalation_reason: string;
  routing_target: string;
  created_at: string;
  status: "open" | "in_review" | "resolved";
  source?: string;
}

export interface KnowledgeDoc {
  id: string;
  similarity: number;
  customer_text: string;
  support_text: string;
  tags: string[];
}

export interface ExampleQuery {
  id: string;
  label: string;
  query: string;
  tone: "neutral" | "critical" | "warning" | "info";
}

export interface CuratedScenario {
  id: string;
  label: string;
  query: string;
  intent: string;
  expected_decision: AgentDecision;
  expected_risk: RiskLevel;
  expected_routing: string;
  description: string;
}

export interface IntentEvaluationMetrics {
  label: string;
  sample_size: number;
  caveat: string;
  accuracy: number;
  weighted_f1: number;
  hybrid_weighted_f1: number;
}

export interface SafetyEvaluationMetrics {
  title: string;
  note: string;
  safety_recall: number;
  hazards_caught: number;
  hazards_total: number;
  false_negatives: number;
  specificity: number;
  mean_latency_us: number;
}

export interface RetrievalEvaluationMetrics {
  corpus_size: number;
  embedding_model: string;
  dimensions: number;
  index_type: string;
  top_k: number;
  mean_top1_similarity: number;
  mean_top3_similarity: number;
  threshold_hit_rate: number;
  mean_latency_ms: number;
  threshold_note: string;
}

export interface GuardrailEvaluationMetrics {
  character_limit_compliance: number;
  pii_safety: number;
  actionable_quality: number;
  llm_as_a_judge: string;
}

export interface EvaluationResponse {
  status: string;
  label: string;
  intent_evaluation: IntentEvaluationMetrics;
  safety_evaluation: SafetyEvaluationMetrics;
  retrieval_evaluation: RetrievalEvaluationMetrics;
  guardrail_evaluation: GuardrailEvaluationMetrics;
}
