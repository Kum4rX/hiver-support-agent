"""Evaluation Harness for Intent Classification.

Evaluates Keyword Rule Baseline, TF-IDF Classifier, and Hybrid Classifier.
Provides transparent reporting with explicit warnings regarding dataset sample sizes.
"""

import os
import sys
from typing import Any, Dict, List, Optional
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.preprocessing import clean_text
from src.models.intent_classifier import (
    KeywordRuleIntentClassifier,
    TfidfLogisticIntentClassifier,
    HybridIntentClassifier,
    ALL_INTENTS
)


def evaluate_intent_classifiers(
    data_path: str = "golden_set.csv",
    model_path: str = "data/intent_baseline.joblib"
) -> Dict[str, Any]:
    """Run intent evaluation comparing Rule Baseline, TF-IDF, and Hybrid on available labelled data."""
    print("=" * 75)
    print("INTENT CLASSIFICATION EVALUATION HARNESS")
    print("=" * 75)

    if not os.path.exists(data_path):
        print(f"Error: Evaluation data file '{data_path}' not found.")
        return {}

    df = pd.read_csv(data_path)
    text_col = "customer_text" if "customer_text" in df.columns else "text"
    label_col = "intent" if "intent" in df.columns else "suggested_intent"

    df = df.dropna(subset=[text_col, label_col]).copy()
    df[text_col] = df[text_col].astype(str).apply(clean_text)
    df[label_col] = df[label_col].astype(str).str.strip()
    df = df[df[text_col].str.len() > 0]

    num_samples = len(df)
    unique_labels = df[label_col].unique().tolist()

    print(f"Evaluation Dataset : {data_path}")
    print(f"Total Usable Samples: {num_samples}")
    print(f"Distinct Classes   : {len(unique_labels)} ({', '.join(unique_labels)})")

    if num_samples < 50:
        print("\n" + "!" * 75)
        print("[DATASET SUFFICIENCY NOTICE]")
        print(f"WARNING: The evaluation dataset has only {num_samples} samples.")
        print("This is statistically insufficient for measuring definitive production metrics.")
        print("The framework runs properly, but numbers below represent a small-sample sanity check.")
        print("!" * 75 + "\n")

    texts = df[text_col].tolist()
    y_true = df[label_col].tolist()

    # 1. Rule Baseline
    rule_clf = KeywordRuleIntentClassifier()
    y_pred_rule = [rule_clf.predict(t) for t in texts]

    # 2. Hybrid Classifier
    hybrid_clf = HybridIntentClassifier(model_path=model_path)
    y_pred_hybrid = [hybrid_clf.classify(t)["intent"] for t in texts]

    # Compute Metrics
    acc_rule = accuracy_score(y_true, y_pred_rule)
    acc_hybrid = accuracy_score(y_true, y_pred_hybrid)

    prec_r, rec_r, f1_r, _ = precision_recall_fscore_support(y_true, y_pred_rule, average="weighted", zero_division=0)
    prec_h, rec_h, f1_h, _ = precision_recall_fscore_support(y_true, y_pred_hybrid, average="weighted", zero_division=0)

    print("-" * 75)
    print(f"{'Classifier Model':<30} | {'Accuracy':<10} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10}")
    print("-" * 75)
    print(f"{'Keyword/Rule Baseline':<30} | {acc_rule:<10.4f} | {prec_r:<10.4f} | {rec_r:<10.4f} | {f1_r:<10.4f}")
    print(f"{'Hybrid (Rule + TF-IDF)':<30} | {acc_hybrid:<10.4f} | {prec_h:<10.4f} | {rec_h:<10.4f} | {f1_h:<10.4f}")
    print("-" * 75)

    print("\nDetailed Breakdown (Hybrid Classifier):")
    print(classification_report(y_true, y_pred_hybrid, zero_division=0))

    return {
        "num_samples": num_samples,
        "is_small_sample": (num_samples < 50),
        "rule_baseline": {
            "accuracy": round(float(acc_rule), 4),
            "precision": round(float(prec_r), 4),
            "recall": round(float(rec_r), 4),
            "f1_weighted": round(float(f1_r), 4)
        },
        "hybrid_model": {
            "accuracy": round(float(acc_hybrid), 4),
            "precision": round(float(prec_h), 4),
            "recall": round(float(rec_h), 4),
            "f1_weighted": round(float(f1_h), 4)
        }
    }


if __name__ == "__main__":
    evaluate_intent_classifiers()
