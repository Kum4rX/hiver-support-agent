"""Evaluation Harness for Intent Classification.

Evaluates:
1. Keyword/Rule Baseline (deterministic zero-shot heuristics)
2. TF-IDF + Logistic Regression Baseline (Stratified 5-Fold Cross-Validation & full-set fit)
3. Hybrid Classifier (Rule heuristics + ML fallback + OOD/Security overrides)

Computes Accuracy, Macro F1, Weighted F1, per-class metrics, and confusion matrix.
Saves structured JSON results to data/evaluation/intent_evaluation_results.json.
"""

import os
import sys
import json
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

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
    subset: Optional[str] = None,
    output_json_path: str = "data/evaluation/intent_evaluation_results.json"
) -> Dict[str, Any]:
    """Run comprehensive intent evaluation comparing Rule Baseline, TF-IDF, and Hybrid."""
    subset_label = f" [Subset: {subset}]" if subset else ""
    print("=" * 80)
    print(f"INTENT CLASSIFICATION EVALUATION HARNESS{subset_label}")
    print("=" * 80)

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
    unique_labels = sorted(df[label_col].unique().tolist())
    is_authoritative = bool("label_source" in df.columns and (df["label_source"] == "human").all())

    print(f"Evaluation Dataset     : {data_path}")
    if subset:
        print(f"Dataset Subset         : {subset}")
    print(f"Total Usable Samples   : {num_samples}")
    print(f"Human-Confirmed Status : {'100% Genuine Human Verified' if is_authoritative else 'Contains provisional/unverified'}")
    print(f"Distinct Classes ({len(unique_labels)}) : {', '.join(unique_labels)}")
    print("-" * 80)

    texts = df[text_col].tolist()
    y_true = df[label_col].tolist()

    # -----------------------------------------------------------------------
    # 1. Keyword / Rule Baseline (Zero-Shot Deterministic)
    # -----------------------------------------------------------------------
    rule_clf = KeywordRuleIntentClassifier()
    y_pred_rule = [rule_clf.predict(t) for t in texts]

    acc_rule = accuracy_score(y_true, y_pred_rule)
    p_rule_macro, r_rule_macro, f1_rule_macro, _ = precision_recall_fscore_support(
        y_true, y_pred_rule, average="macro", zero_division=0
    )
    p_rule_wt, r_rule_wt, f1_rule_wt, _ = precision_recall_fscore_support(
        y_true, y_pred_rule, average="weighted", zero_division=0
    )

    # -----------------------------------------------------------------------
    # 2. Standalone TF-IDF Baseline (Stratified 5-Fold Cross-Validation)
    # -----------------------------------------------------------------------
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    X_arr = np.array(texts)
    y_arr = np.array(y_true)
    oof_preds_tfidf = np.empty_like(y_arr)

    for train_idx, test_idx in skf.split(X_arr, y_arr):
        fold_pipe = Pipeline([
            ("tfidf", TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=1, sublinear_tf=True)),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42))
        ])
        fold_pipe.fit(X_arr[train_idx], y_arr[train_idx])
        oof_preds_tfidf[test_idx] = fold_pipe.predict(X_arr[test_idx])

    acc_tfidf_cv = accuracy_score(y_true, oof_preds_tfidf)
    p_tfidf_cv_macro, r_tfidf_cv_macro, f1_tfidf_cv_macro, _ = precision_recall_fscore_support(
        y_true, oof_preds_tfidf, average="macro", zero_division=0
    )
    p_tfidf_cv_wt, r_tfidf_cv_wt, f1_tfidf_cv_wt, _ = precision_recall_fscore_support(
        y_true, oof_preds_tfidf, average="weighted", zero_division=0
    )

    # In-sample fit of TF-IDF model on full dataset
    tfidf_fitted = TfidfLogisticIntentClassifier(model_path=model_path)
    if not tfidf_fitted.is_trained() or len(tfidf_fitted.classes_ or []) < len(unique_labels):
        full_pipe = Pipeline([
            ("tfidf", TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=1, sublinear_tf=True)),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42))
        ])
        full_pipe.fit(texts, y_true)
        tfidf_fitted.pipeline = full_pipe
        tfidf_fitted.classes_ = list(full_pipe.classes_)
        tfidf_fitted.save(model_path)

    y_pred_tfidf_fit = [tfidf_fitted.predict(t) for t in texts]
    acc_tfidf_fit = accuracy_score(y_true, y_pred_tfidf_fit)
    p_tfidf_fit_macro, r_tfidf_fit_macro, f1_tfidf_fit_macro, _ = precision_recall_fscore_support(
        y_true, y_pred_tfidf_fit, average="macro", zero_division=0
    )
    p_tfidf_fit_wt, r_tfidf_fit_wt, f1_tfidf_fit_wt, _ = precision_recall_fscore_support(
        y_true, y_pred_tfidf_fit, average="weighted", zero_division=0
    )

    # -----------------------------------------------------------------------
    # 3. Hybrid Classifier (Rule + ML + OOD/Security Overrides)
    # -----------------------------------------------------------------------
    hybrid_clf = HybridIntentClassifier(model_path=model_path)
    y_pred_hybrid = [hybrid_clf.classify(t)["intent"] for t in texts]

    acc_hybrid = accuracy_score(y_true, y_pred_hybrid)
    p_hyb_macro, r_hyb_macro, f1_hyb_macro, _ = precision_recall_fscore_support(
        y_true, y_pred_hybrid, average="macro", zero_division=0
    )
    p_hyb_wt, r_hyb_wt, f1_hyb_wt, _ = precision_recall_fscore_support(
        y_true, y_pred_hybrid, average="weighted", zero_division=0
    )

    # Classification reports and confusion matrices
    report_rule = classification_report(y_true, y_pred_rule, zero_division=0, output_dict=True)
    report_tfidf_cv = classification_report(y_true, oof_preds_tfidf, zero_division=0, output_dict=True)
    report_hybrid = classification_report(y_true, y_pred_hybrid, zero_division=0, output_dict=True)

    cm_labels = sorted(list(set(y_true) | set(y_pred_hybrid)))
    cm_hybrid = confusion_matrix(y_true, y_pred_hybrid, labels=cm_labels).tolist()

    # -----------------------------------------------------------------------
    # Display Formatted Summary Table
    # -----------------------------------------------------------------------
    print(f"{'Classifier Model':<35} | {'Accuracy':<10} | {'Macro F1':<10} | {'Weighted F1':<12} | {'Evaluation Mode'}")
    print("-" * 88)
    print(f"{'Keyword/Rule Baseline':<35} | {acc_rule:<10.4f} | {f1_rule_macro:<10.4f} | {f1_rule_wt:<12.4f} | Zero-Shot Domain Rules")
    print(f"{'TF-IDF + Logistic Reg (5-Fold CV)':<35} | {acc_tfidf_cv:<10.4f} | {f1_tfidf_cv_macro:<10.4f} | {f1_tfidf_cv_wt:<12.4f} | Out-of-Sample Stratified CV")
    print(f"{'TF-IDF + Logistic Reg (Fitted)':<35} | {acc_tfidf_fit:<10.4f} | {f1_tfidf_fit_macro:<10.4f} | {f1_tfidf_fit_wt:<12.4f} | In-Sample Fit (Overfit Reference)")
    print(f"{'Hybrid Model (Rule + ML + Guards)':<35} | {acc_hybrid:<10.4f} | {f1_hyb_macro:<10.4f} | {f1_hyb_wt:<12.4f} | Production Agent Pipeline")
    print("-" * 88)

    print("\nDetailed Per-Class Breakdown (Hybrid Model):")
    print(classification_report(y_true, y_pred_hybrid, zero_division=0))

    results = {
        "dataset": data_path,
        "num_samples": num_samples,
        "is_authoritative_human": is_authoritative,
        "rule_baseline": {
            "accuracy": round(float(acc_rule), 4),
            "macro_precision": round(float(p_rule_macro), 4),
            "macro_recall": round(float(r_rule_macro), 4),
            "macro_f1": round(float(f1_rule_macro), 4),
            "weighted_precision": round(float(p_rule_wt), 4),
            "weighted_recall": round(float(r_rule_wt), 4),
            "weighted_f1": round(float(f1_rule_wt), 4),
            "per_class": report_rule
        },
        "tfidf_baseline_cv": {
            "evaluation_mode": "stratified_5_fold_cv",
            "accuracy": round(float(acc_tfidf_cv), 4),
            "macro_precision": round(float(p_tfidf_cv_macro), 4),
            "macro_recall": round(float(r_tfidf_cv_macro), 4),
            "macro_f1": round(float(f1_tfidf_cv_macro), 4),
            "weighted_precision": round(float(p_tfidf_cv_wt), 4),
            "weighted_recall": round(float(r_tfidf_cv_wt), 4),
            "weighted_f1": round(float(f1_tfidf_cv_wt), 4),
            "per_class": report_tfidf_cv
        },
        "tfidf_baseline_fitted": {
            "evaluation_mode": "in_sample_fit",
            "accuracy": round(float(acc_tfidf_fit), 4),
            "macro_f1": round(float(f1_tfidf_fit_macro), 4),
            "weighted_f1": round(float(f1_tfidf_fit_wt), 4)
        },
        "hybrid_model": {
            "accuracy": round(float(acc_hybrid), 4),
            "macro_precision": round(float(p_hyb_macro), 4),
            "macro_recall": round(float(r_hyb_macro), 4),
            "macro_f1": round(float(f1_hyb_macro), 4),
            "weighted_precision": round(float(p_hyb_wt), 4),
            "weighted_recall": round(float(r_hyb_wt), 4),
            "weighted_f1": round(float(f1_hyb_wt), 4),
            "per_class": report_hybrid,
            "confusion_matrix": {
                "labels": cm_labels,
                "matrix": cm_hybrid
            }
        }
    }

    if output_json_path:
        os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, default=lambda x: x.item() if hasattr(x, "item") else str(x))
        print(f"Results saved to '{output_json_path}'.")

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate intent classification models.")
    parser.add_argument("--data", default="golden_set.csv", help="Path to evaluation CSV file")
    parser.add_argument("--model", default="data/intent_baseline.joblib", help="Path to trained model")
    parser.add_argument("--subset", choices=["human", "auto_provisional"], default=None, help="Filter by label_source")
    parser.add_argument("--output", default="data/evaluation/intent_evaluation_results.json", help="Path to save JSON metrics")
    args = parser.parse_args()

    evaluate_intent_classifiers(
        data_path=args.data,
        model_path=args.model,
        subset=args.subset,
        output_json_path=args.output
    )
