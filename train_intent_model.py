"""Script to train and serialize the TF-IDF intent classification model.

IMPORTANT:
If the dataset has few human-reviewed samples, this script clearly marks the resulting model
as PROVISIONAL for pipeline orchestration and does not claim production benchmark readiness.
"""

import os
import sys
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.preprocessing import clean_text
from src.models.intent_classifier import TfidfLogisticIntentClassifier, ALL_INTENTS


def train_intent_model(
    data_path: str = "golden_set.csv",
    output_path: str = "data/intent_baseline.joblib",
    min_samples_for_final: int = 100
) -> TfidfLogisticIntentClassifier:
    """Train and serialize the TF-IDF intent classifier."""
    print("=" * 70)
    print("INTENT CLASSIFIER TRAINING PIPELINE")
    print("=" * 70)
    print(f"Loading data from: {data_path}")

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}")

    df = pd.read_csv(data_path)
    
    # Check for required columns
    text_col = "customer_text" if "customer_text" in df.columns else "text"
    label_col = "intent" if "intent" in df.columns else "suggested_intent"

    df = df.dropna(subset=[text_col, label_col]).copy()
    df[text_col] = df[text_col].astype(str).apply(clean_text)
    df[label_col] = df[label_col].astype(str).str.strip()
    df = df[df[text_col].str.len() > 0]

    num_samples = len(df)
    print(f"Usable samples: {num_samples}")
    print("\nClass distribution in training data:")
    print(df[label_col].value_counts())

    is_provisional = num_samples < min_samples_for_final
    if is_provisional:
        print("\n" + "!" * 70)
        print("[PROVISIONAL MODEL NOTICE]")
        print(f"Training dataset contains only {num_samples} samples (< {min_samples_for_final}).")
        print("This model is trained as a functional component for pipeline integration.")
        print("Its evaluation metrics should NOT be cited as final production benchmark performance.")
        print("!" * 70 + "\n")

    X = df[text_col].tolist()
    y = df[label_col].tolist()

    # Build Pipeline
    # Using min_df=1 so it works properly even with few samples
    pipeline = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                lowercase=True,
                ngram_range=(1, 2),
                min_df=1,
                max_df=1.0,
                sublinear_tf=True
            )
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=42
            )
        )
    ])

    print("Fitting model...")
    pipeline.fit(X, y)

    # In-sample evaluation for sanity check
    preds = pipeline.predict(X)
    print("\nIn-Sample Fit Report (Sanity Check Only):")
    print(classification_report(y, preds, zero_division=0))

    # Save model
    classifier = TfidfLogisticIntentClassifier()
    classifier.pipeline = pipeline
    classifier.classes_ = list(pipeline.classes_)
    classifier.save(output_path)
    print(f"\nModel saved successfully to: {output_path}")
    print("=" * 70)

    return classifier


if __name__ == "__main__":
    train_intent_model()
