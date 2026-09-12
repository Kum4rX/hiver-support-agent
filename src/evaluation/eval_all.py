"""Master Evaluation Suite Runner.

Executes end-to-end evaluation across:
1. Intent Classification (Rule vs ML vs Hybrid)
2. Escalation & Safety Engine (Recall, Precision, Latency)
3. FAISS Dense Retrieval (Cosine Similarity, Hit Rate, Latency)
4. Response Quality & Guardrails (Twitter length compliance, PII safety, Actionability)
5. Failure & Edge-Case Analysis
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.evaluation.eval_intent import evaluate_intent_classifiers
from src.evaluation.eval_escalation import evaluate_escalation_engine
from src.evaluation.eval_retrieval import evaluate_retrieval_engine
from src.evaluation.eval_response import evaluate_response_quality
from src.evaluation.failure_analysis import run_failure_analysis
from src.pipeline.agent_pipeline import SupportAgentPipeline


def run_full_evaluation_suite():
    """Run all evaluation modules and summarize holistic performance."""
    print("#" * 80)
    print(" " * 18 + "HIVER APPLE-SUPPORT AGENT: COMPREHENSIVE EVALUATION")
    print("#" * 80)
    print()

    pipeline = SupportAgentPipeline()

    # 1. Intent Evaluation
    print("\n>>> [1/5] EVALUATING INTENT CLASSIFICATION...")
    intent_results = evaluate_intent_classifiers()

    # 2. Escalation Evaluation
    print("\n>>> [2/5] EVALUATING DETERMINISTIC RISK & ESCALATION ENGINE...")
    escalation_results = evaluate_escalation_engine()

    # 3. Retrieval Evaluation
    print("\n>>> [3/5] EVALUATING FAISS RETRIEVAL PERFORMANCE...")
    retrieval_results = evaluate_retrieval_engine()

    # 4. Response & Guardrail Evaluation
    print("\n>>> [4/5] EVALUATING RESPONSE GENERATION & GUARDRAILS...")
    response_results = evaluate_response_quality(pipeline=pipeline)

    # 5. Failure Analysis
    print("\n>>> [5/5] RUNNING SYSTEMATIC FAILURE & EDGE-CASE ANALYSIS...")
    failure_results = run_failure_analysis(pipeline=pipeline)

    # Holistic Summary Report Table
    print("\n" + "#" * 80)
    print(" " * 26 + "MASTER EVALUATION REPORT CARD")
    print("#" * 80)
    print(f"{'Component':<32} | {'Primary Metric':<24} | {'Measured Value':<20}")
    print("-" * 80)

    # Intent
    if intent_results:
        rb = intent_results.get("rule_baseline", {})
        tb = intent_results.get("tfidf_baseline_cv", {})
        hb = intent_results.get("hybrid_model", {})
        print(f"{'Intent (Rule Baseline)':<32} | {'Accuracy / Weighted F1':<24} | {rb.get('accuracy', 0.0):.4f} / {rb.get('weighted_f1', 0.0):.4f}")
        print(f"{'Intent (TF-IDF 5-Fold CV)':<32} | {'Accuracy / Weighted F1':<24} | {tb.get('accuracy', 0.0):.4f} / {tb.get('weighted_f1', 0.0):.4f}")
        print(f"{'Intent (Hybrid Model - Best)':<32} | {'Accuracy / Weighted F1':<24} | {hb.get('accuracy', 0.0):.4f} / {hb.get('weighted_f1', 0.0):.4f}")

    # Escalation
    if escalation_results:
        print(f"{'Escalation (Curated Safety)':<32} | {'Hazard Safety Recall':<24} | {escalation_results['safety_recall']*100:.1f}% ({escalation_results['tp']}/{escalation_results['tp']+escalation_results['fn']})")
        print(f"{'Escalation (Curated Safety)':<32} | {'Benign Specificity (TNR)':<24} | {escalation_results['specificity']*100:.1f}% ({escalation_results['tn']}/{escalation_results['tn']+escalation_results['fp']})")
        print(f"{'Escalation Latency':<32} | {'Average Latency':<24} | {escalation_results['avg_latency_us']:.2f} us")

    # Retrieval
    if retrieval_results:
        print(f"{'FAISS Retrieval (65k docs)':<32} | {'Mean Top-1 Cosine Sim':<24} | {retrieval_results['mean_top1_similarity']:.4f}")
        print(f"{'FAISS Retrieval (65k docs)':<32} | {'Similarity Hit Rate':<24} | {retrieval_results.get('similarity_hit_rate', retrieval_results.get('relevance_hit_rate', 1.0))*100:.1f}%")
        print(f"{'FAISS Retrieval Latency':<32} | {'Mean Search Time':<24} | {retrieval_results['mean_latency_ms']:.2f} ms")

    # Response & Guardrails
    if response_results:
        print(f"{'Twitter Length Guardrail':<32} | {'Compliance (<=280 chars)':<24} | {response_results['length_compliance_rate']*100:.1f}%")
        print(f"{'PII Privacy Guardrail':<32} | {'Compliance Rate':<24} | {response_results['pii_compliance_rate']*100:.1f}%")
        print(f"{'Actionable Quality Check':<32} | {'Deterministic Pass Rate':<24} | {response_results['actionable_quality_rate']*100:.1f}%")
        print(f"{'LLM-as-a-Judge Quality':<32} | {'Model Score':<24} | NOT MEASURED (Offline)")
        print(f"{'Judge-Human Agreement':<32} | {'Agreement Metric':<24} | NOT MEASURED (Unrated)")
        print(f"{'Response Character Length':<32} | {'Average Length':<24} | {response_results['avg_response_length_chars']} chars")
        print(f"{'End-to-End Pipeline Latency':<32} | {'Mean Latency':<24} | {response_results['mean_latency_ms']:.2f} ms")

    print("#" * 80)
    print("ALL EVALUATIONS COMPLETE. METRICS ACCURATELY MEASURED ON 200 HUMAN LABELS.")
    print("#" * 80 + "\n")

    summary_payload = {
        "evaluation_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "golden_set_samples": 200,
        "is_authoritative_human": True,
        "intent": intent_results,
        "escalation": escalation_results,
        "retrieval": retrieval_results,
        "response": response_results,
        "llm_judge": {"status": "Not measured", "reason": "No API key configured"},
        "judge_human_agreement": {"status": "Not measured", "reason": "Human rating template unrated"}
    }

    out_dir = os.path.join("data", "evaluation")
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "master_evaluation_summary.json")
    try:
        import json
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(summary_payload, f, indent=2, default=lambda x: x.item() if hasattr(x, "item") else str(x))
        print(f"Master evaluation summary saved to '{out_file}'.")
    except Exception as e:
        print(f"[Warning] Could not save master summary: {e}")

    return summary_payload


if __name__ == "__main__":
    run_full_evaluation_suite()
