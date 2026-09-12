"""Evaluation Harness for Deterministic Risk & Escalation Engine.

Evaluates safety recall, false positive rates, and latency across critical risk categories
versus standard benign customer support queries.
"""

import os
import sys
import time
from typing import Any, Dict, List
import pandas as pd

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
    avg_latency_us = total_eval_time / total

    print("-" * 75)
    print("EVALUATION SUMMARY METRICS:")
    print(f"  Total Test Cases           : {total}")
    print(f"  True Positives (Hazards)   : {tp}")
    print(f"  True Negatives (Benign)    : {tn}")
    print(f"  False Positives (Over-esc) : {fp}")
    print(f"  False Negatives (Missed)   : {fn}")
    print(f"  Safety Recall (Target 100%): {recall * 100:.2f}%")
    print(f"  Benign Precision           : {precision * 100:.2f}%")
    print(f"  Specificity (True Neg Rate): {specificity * 100:.2f}%")
    print(f"  Overall Accuracy           : {accuracy * 100:.2f}%")
    print(f"  Average Engine Latency     : {avg_latency_us:.2f} microseconds (µs)")
    print("=" * 75)

    return {
        "total_cases": total,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "accuracy": round(accuracy, 4),
        "safety_recall": round(recall, 4),
        "precision": round(precision, 4),
        "specificity": round(specificity, 4),
        "avg_latency_us": round(avg_latency_us, 2)
    }


if __name__ == "__main__":
    evaluate_escalation_engine()
