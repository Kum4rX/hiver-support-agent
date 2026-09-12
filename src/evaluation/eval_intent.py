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
    model_path: str = "data/intent_baseline.joblib",
    subset: Optional[str] = None
) -> Dict[str, Any]:
    """Run intent evaluation comparing Rule Baseline, TF-IDF, and Hybrid on available labelled data.

    Args:
        data_path: Path to CSV dataset (golden_set.csv or data/golden_evaluation_provisional.csv).
        model_path: Path to trained TF-IDF model joblib.
        subset: Optional filter by 'label_source' ('human', 'auto_provisional', or None for all).
    """
    subset_label = f" [Subset: {subset}]" if subset else ""
    print("=" * 75)
    print(f"INTENT CLASSIFICATION EVALUATION HARNESS{subset_label}")
    print("=" * 75)

    if not os.path.exists(data_path):
        print(f"Error: Evaluation data file '{data_path}' not found.")
        return {}

    df = pd.read_csv(data_path)

    # Filter subset if requested and available
    if subset and "label_source" in df.columns:
        df = df[df["label_source"] == subset].copy()

    text_col = "customer_text" if "customer_text" in df.columns else "text"
    label_col = "intent" if "intent" in df.columns else "suggested_intent"

    df = df.dropna(subset=[text_col, label_col]).copy()
    df[text_col] = df[text_col].astype(str).apply(clean_text)
    df[label_col] = df[label_col].astype(str).str.strip()
    df = df[df[text_col].str.len() > 0]

    num_samples = len(df)
    unique_labels = df[label_col].unique().tolist()

    has_provisional = "label_source" in df.columns and (df["label_source"] == "auto_provisional").any()

    print(f"Evaluation Dataset : {data_path}")
    if subset:
        print(f"Dataset Subset     : {subset}")
    print(f"Total Usable Samples: {num_samples}")
    print(f"Distinct Classes   : {len(unique_labels)} ({', '.join(unique_labels)})")

    if has_provisional:
        print("\n" + "~" * 75)
        print("[PROVISIONAL EVALUATION NOTICE]")
        print("NOTE: This evaluation includes 'auto_provisional' labels derived from candidate")
        print("suggestions. These are NOT human ground truth labels and metrics must be treated")
        print("as exploratory / sanity checks rather than definitive production benchmarks.")
        print("~" * 75 + "\n")
    elif num_samples < 50:
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

    if num_samples >= 10:
        print("\nDetailed Breakdown (Hybrid Classifier):")
        print(classification_report(y_true, y_pred_hybrid, zero_division=0))

    return {
        "num_samples": num_samples,
        "subset": subset,
        "has_provisional": has_provisional,
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


def evaluate_all_subsets(
    data_path: str = "data/golden_evaluation_provisional.csv",
    model_path: str = "data/intent_baseline.joblib"
) -> Dict[str, Any]:
    """Evaluate Human-only, Auto-Provisional, and Combined datasets side-by-side."""
    print("#" * 80)
    print(" " * 16 + "INTENT EVALUATION ACROSS ALL PROVENANCE SUBSETS")
    print("#" * 80)

    results = {}

    # A. Human-only (label_source == 'human')
    print("\n>>> SUBSET A: HUMAN-LABELLED GROUND TRUTH (11 samples)")
    results["human"] = evaluate_intent_classifiers(data_path=data_path, model_path=model_path, subset="human")

    # B. Auto-provisional (label_source == 'auto_provisional')
    print("\n>>> SUBSET B: AUTO-PROVISIONAL (189 samples — not ground truth)")
    results["auto_provisional"] = evaluate_intent_classifiers(data_path=data_path, model_path=model_path, subset="auto_provisional")

    # C. Combined Provisional (all 200 rows)
    print("\n>>> SUBSET C: COMBINED PROVISIONAL (200 samples — 11 human + 189 provisional)")
    results["combined"] = evaluate_intent_classifiers(data_path=data_path, model_path=model_path, subset=None)

    # Summary comparison table
    print("\n" + "=" * 80)
    print(f"{'Evaluation Subset':<28} | {'Source':<16} | {'Samples':<8} | {'Hybrid Acc':<11} | {'Hybrid F1':<10}")
    print("-" * 80)
    for key, label, src in [
        ("human", "Human-Only", "Human Verified"),
        ("auto_provisional", "Auto-Provisional", "Auto-Labelled"),
        ("combined", "Combined Provisional", "11 Human + 189 Prov")
    ]:
        res = results.get(key, {})
        h = res.get("hybrid_model", {})
        n = res.get("num_samples", 0)
        acc = h.get("accuracy", 0.0)
        f1 = h.get("f1_weighted", 0.0)
        print(f"{label:<28} | {src:<16} | {n:<8} | {acc:<11.4f} | {f1:<10.4f}")
    print("=" * 80)
    print("NOTE: Auto-provisional metrics reflect concordance with candidate suggestions,")
    print("      NOT independent human verification. Benchmark metrics remain provisional.")
    print("=" * 80 + "\n")

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate intent classification models.")
    parser.add_argument("--data", default="golden_set.csv", help="Path to evaluation CSV file")
    parser.add_argument("--model", default="data/intent_baseline.joblib", help="Path to trained model")
    parser.add_argument("--subset", choices=["human", "auto_provisional"], default=None, help="Filter by label_source")
    parser.add_argument("--all-subsets", action="store_true", help="Evaluate human, provisional, and combined subsets")
    args = parser.parse_args()

    if args.all_subsets or (args.data == "golden_set.csv" and os.path.exists("data/golden_evaluation_provisional.csv") and len(sys.argv) == 1):
        # Default behavior when run directly: evaluate all subsets from provisional file
        evaluate_all_subsets()
    else:
        evaluate_intent_classifiers(data_path=args.data, model_path=args.model, subset=args.subset)
