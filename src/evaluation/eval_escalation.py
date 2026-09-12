"""Evaluation Harness for Deterministic Risk & Escalation Engine.

Evaluates safety recall, false positive rates, and latency across critical risk categories
versus standard benign customer support queries.
"""

import os
import sys
import time
from typing import Any, Dict, List
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.models.escalation_engine import EscalationEngine


# Test suite containing safety hazards, security compromises, fraud, legal threats, and benign queries
ESCALATION_TEST_CASES = [
    # Critical Physical Safety Hazards (Must Escalate)
    {"text": "Help! My iPhone battery is swollen and bulging out of the screen!", "expected_escalated": True, "category": "PHYSICAL_SAFETY_HAZARD"},
    {"text": "My charger started sparking and smoking while plugged into the wall.", "expected_escalated": True, "category": "PHYSICAL_SAFETY_HAZARD"},
    {"text": "The phone got extremely hot and caught fire on my desk.", "expected_escalated": True, "category": "PHYSICAL_SAFETY_HAZARD"},
    {"text": "I got an electrical shock when touching the melted charging wire.", "expected_escalated": True, "category": "PHYSICAL_SAFETY_HAZARD"},

    # Account Security Compromise (Must Escalate)
    {"text": "Someone hacked my Apple ID and changed my password and email address!", "expected_escalated": True, "category": "ACCOUNT_SECURITY_COMPROMISE"},
    {"text": "I received a suspicious phishing message asking for my iCloud credentials.", "expected_escalated": True, "category": "ACCOUNT_SECURITY_COMPROMISE"},
    {"text": "My iPhone was stolen and locked out with unauthorized password reset.", "expected_escalated": True, "category": "ACCOUNT_SECURITY_COMPROMISE"},

    # Financial Fraud & Billing (Must Escalate)
    {"text": "There is a fraudulent charge of $499 on my Apple Pay that I never authorized!", "expected_escalated": True, "category": "FINANCIAL_FRAUD_DISPUTE"},
    {"text": "I was charged without my permission for an app subscription.", "expected_escalated": True, "category": "FINANCIAL_FRAUD_DISPUTE"},

    # Legal / Regulatory (Must Escalate)
    {"text": "I am hiring a lawyer and filing a lawsuit against Apple for data loss.", "expected_escalated": True, "category": "LEGAL_OR_REGULATORY"},
    {"text": "I am filing a police report and FTC complaint regarding this issue.", "expected_escalated": True, "category": "LEGAL_OR_REGULATORY"},

    # Chronic Unresolved Escalation (Must Escalate)
    {"text": "I have called 10 times and nobody helps. Transfer me to a supervisor right now.", "expected_escalated": True, "category": "CHRONIC_UNRESOLVED_SERVICE"},
    {"text": "This is my 5th replacement device and it's also broken. Unacceptable!", "expected_escalated": True, "category": "CHRONIC_UNRESOLVED_SERVICE"},

    # Benign Standard Queries (Must NOT Escalate)
    {"text": "My iPhone battery is draining fast after the new iOS update.", "expected_escalated": False, "category": "NONE"},
    {"text": "How do I connect my iPad to my home Wi-Fi network?", "expected_escalated": False, "category": "NONE"},
    {"text": "The keyboard is lagging when I type fast in iMessage.", "expected_escalated": False, "category": "NONE"},
    {"text": "Can I trade in my iPhone 7 for a new model?", "expected_escalated": False, "category": "NONE"},
    {"text": "How do I turn on Dark Mode in iOS settings?", "expected_escalated": False, "category": "NONE"},
    {"text": "My camera is blurry when taking pictures in low light.", "expected_escalated": False, "category": "NONE"},
    {"text": "How do I back up my photos to iCloud?", "expected_escalated": False, "category": "NONE"},
    {"text": "My volume buttons are a bit stiff after dropping the case.", "expected_escalated": False, "category": "NONE"},
]


def evaluate_escalation_engine() -> Dict[str, Any]:
    """Evaluate escalation engine precision, recall, and latency."""
    print("=" * 75)
    print("DETERMINISTIC ESCALATION & SAFETY ENGINE EVALUATION")
    print("=" * 75)

    engine = EscalationEngine()
    
    tp, fp, tn, fn = 0, 0, 0, 0
    total_eval_time = 0.0

    print(f"{'Query Snippet':<45} | {'Expected':<10} | {'Predicted':<10} | {'Result':<6}")
    print("-" * 75)

    for case in ESCALATION_TEST_CASES:
        start = time.perf_counter()
        result = engine.evaluate(case["text"])
        elapsed_us = (time.perf_counter() - start) * 1_000_000
        total_eval_time += elapsed_us

        expected = case["expected_escalated"]
        predicted = result["is_escalated"]

        if expected and predicted:
            tp += 1
            status = "PASS (TP)"
        elif not expected and not predicted:
            tn += 1
            status = "PASS (TN)"
        elif not expected and predicted:
            fp += 1
            status = "FAIL (FP)"
        else:
            fn += 1
            status = "FAIL (FN)"

        snippet = (case["text"][:42] + "...") if len(case["text"]) > 45 else case["text"]
        print(f"{snippet:<45} | {str(expected):<10} | {str(predicted):<10} | {status:<6}")

    total = len(ESCALATION_TEST_CASES)
    accuracy = (tp + tn) / total
    recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 1.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    avg_latency_us = total_eval_time / total

    print("-" * 75)
    print("CURATED TEST SUITE SUMMARY:")
    print(f"  Total Test Cases           : {total}")
    print(f"  True Positives (Hazards)   : {tp} / {tp + fn}")
    print(f"  True Negatives (Benign)    : {tn} / {tn + fp}")
    print(f"  False Positives (Over-esc) : {fp}")
    print(f"  False Negatives (Missed)   : {fn}")
    print(f"  Safety Recall (Target 100%): {recall * 100:.2f}%")
    print(f"  Precision                  : {precision * 100:.2f}%")
    print(f"  Specificity (True Neg Rate): {specificity * 100:.2f}%")
    print(f"  F1-Score                   : {f1 * 100:.2f}%")
    print(f"  Overall Accuracy           : {accuracy * 100:.2f}%")
    print(f"  Average Engine Latency     : {avg_latency_us:.2f} us")
    print("=" * 75)

    curated_metrics = {
        "suite": "curated_hazard_tests",
        "total_cases": total,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "accuracy": round(accuracy, 4),
        "safety_recall": round(recall, 4),
        "precision": round(precision, 4),
        "specificity": round(specificity, 4),
        "f1": round(f1, 4),
        "avg_latency_us": round(avg_latency_us, 2)
    }

    # -----------------------------------------------------------------------
    # Golden Set Evaluation (200 In-The-Wild Human Confirmed Labels)
    # -----------------------------------------------------------------------
    golden_metrics = None
    golden_path = "golden_set.csv"
    if os.path.exists(golden_path):
        try:
            df = pd.read_csv(golden_path)
            y_true = (df["escalation"].astype(str).str.lower() == "yes").tolist()
            texts = df["customer_text"].tolist()

            g_tp, g_fp, g_tn, g_fn = 0, 0, 0, 0
            g_latencies = []

            for text, true_esc in zip(texts, y_true):
                st = time.perf_counter()
                pred_esc = engine.evaluate(text)["is_escalated"]
                g_latencies.append((time.perf_counter() - st) * 1_000_000)

                if true_esc and pred_esc:
                    g_tp += 1
                elif not true_esc and not pred_esc:
                    g_tn += 1
                elif not true_esc and pred_esc:
                    g_fp += 1
                else:
                    g_fn += 1

            g_total = len(texts)
            g_acc = (g_tp + g_tn) / g_total
            g_rec = g_tp / (g_tp + g_fn) if (g_tp + g_fn) > 0 else 0.0
            g_prec = g_tp / (g_tp + g_fp) if (g_tp + g_fp) > 0 else 0.0
            g_spec = g_tn / (g_tn + g_fp) if (g_tn + g_fp) > 0 else 1.0
            g_f1 = 2 * g_prec * g_rec / (g_prec + g_rec) if (g_prec + g_rec) > 0 else 0.0

            print("\n" + "=" * 75)
            print("GOLDEN SET IN-THE-WILD ESCALATION EVALUATION (N=200)")
            print("=" * 75)
            print(f"  Total Evaluated            : {g_total}")
            print(f"  Human Escalated Cases      : {sum(y_true)} (Physical hazards, account security, legal)")
            print(f"  Human Non-Escalated Cases  : {g_total - sum(y_true)}")
            print(f"  True Positives (Detected)  : {g_tp} / {sum(y_true)}")
            print(f"  True Negatives (Passed)    : {g_tn} / {g_total - sum(y_true)}")
            print(f"  False Positives (Over-esc) : {g_fp}")
            print(f"  False Negatives (Missed)   : {g_fn}")
            print(f"  Safety Recall              : {g_rec * 100:.2f}%")
            print(f"  Specificity (Zero Over-esc): {g_spec * 100:.2f}%")
            print(f"  Precision                  : {g_prec * 100:.2f}%")
            print(f"  F1-Score                   : {g_f1 * 100:.2f}%")
            print(f"  Overall Accuracy           : {g_acc * 100:.2f}%")
            print(f"  Mean Latency               : {float(np.mean(g_latencies)):.2f} us")
            print("=" * 75)

            golden_metrics = {
                "suite": "golden_set_200_human",
                "total_cases": g_total,
                "human_escalated": sum(y_true),
                "human_non_escalated": g_total - sum(y_true),
                "tp": g_tp, "tn": g_tn, "fp": g_fp, "fn": g_fn,
                "accuracy": round(g_acc, 4),
                "safety_recall": round(g_rec, 4),
                "precision": round(g_prec, 4),
                "specificity": round(g_spec, 4),
                "f1": round(g_f1, 4),
                "avg_latency_us": round(float(np.mean(g_latencies)), 2)
            }
        except Exception as e:
            print(f"[Warning] Could not evaluate golden set escalation: {e}")

    output_results = {
        "curated_suite": curated_metrics,
        "golden_set_evaluation": golden_metrics
    }

    out_dir = os.path.join("data", "evaluation")
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "escalation_evaluation_results.json")
    try:
        import json
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(output_results, f, indent=2)
        print(f"Escalation results saved to '{out_file}'.")
    except Exception as e:
        print(f"[Warning] Could not save escalation results: {e}")

    return curated_metrics


if __name__ == "__main__":
    evaluate_escalation_engine()
