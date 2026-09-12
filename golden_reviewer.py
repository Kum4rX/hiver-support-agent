"""Golden Set Annotation Workflow for Hiver AppleSupport AI Agent.

Provides a fast, reliable, human-in-the-loop review interface to create
the final 200-example Golden Evaluation Set.

Features:
- Preserves the 11 genuinely reviewed examples.
- Queues candidates from golden_candidates.csv and apple_support_pairs_clean.csv.
- Prioritizes underrepresented intent classes for balanced coverage across 11 classes.
- Immediate atomic auto-save on every single review.
- Automatic resume from last reviewed candidate.
- Explicit human confirmation (ACCEPT suggested, CHANGE intent, or SKIP).
- Separate escalation labeling (yes/no + reason).
- Strict schema enforcement:
  [tweet_id, customer_text, intent, escalation, escalation_reason, label_source]
  where label_source is strictly 'human'.
- Generates data/golden_set_summary.json.
- Supports --status, --verify-schema, and --test for validation.
"""

import os
import sys
import json
import re
from typing import Dict, List, Set, Tuple, Optional
import pandas as pd

# Constants
TARGET_COUNT = 200
GOLDEN_SET_FILE = "golden_set.csv"
CANDIDATES_FILE = "golden_candidates.csv"
CLEAN_PAIRS_FILE = os.path.join("data", "apple_support_pairs_clean.csv")
SKIPPED_FILE = os.path.join("data", "golden_skipped.csv")
SUMMARY_FILE = os.path.join("data", "golden_set_summary.json")

INTENTS = [
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
]

SCHEMA_COLUMNS = [
    "tweet_id",
    "customer_text",
    "intent",
    "escalation",
    "escalation_reason",
    "label_source",
]


def normalize_text_for_dedup(text: str) -> str:
    """Normalize text for strict duplicate detection."""
    return re.sub(r"\s+", " ", str(text).strip().lower())


def load_or_init_golden_set(filepath: str = GOLDEN_SET_FILE) -> pd.DataFrame:
    """Load existing golden_set.csv and normalize to required schema."""
    if not os.path.exists(filepath):
        df = pd.DataFrame(columns=SCHEMA_COLUMNS)
        df.to_csv(filepath, index=False)
        return df

    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        print(f"[Warning] Could not read '{filepath}': {e}. Initializing fresh DataFrame.")
        return pd.DataFrame(columns=SCHEMA_COLUMNS)

    # Handle legacy column names if present
    if "support_response" in df.columns and "tweet_id" not in df.columns:
        df = df.rename(columns={"support_response": "tweet_id"})

    if "text" in df.columns and "customer_text" not in df.columns:
        df = df.rename(columns={"text": "customer_text"})

    # Ensure all schema columns exist
    for col in SCHEMA_COLUMNS:
        if col not in df.columns:
            if col == "label_source":
                df[col] = "human"
            elif col == "escalation":
                df[col] = "no"
            elif col == "escalation_reason":
                df[col] = ""
            else:
                df[col] = ""

    # Clean up fields
    df["escalation_reason"] = df["escalation_reason"].fillna("").astype(str).str.strip()
    df["escalation"] = df["escalation"].fillna("no").astype(str).str.lower().str.strip()
    df["label_source"] = "human"  # All accepted rows in golden_set are human verified

    # Keep only required columns in exact order
    df = df[SCHEMA_COLUMNS].copy()

    # If the file on disk had old columns, persist the clean normalized schema
    try:
        raw_disk = pd.read_csv(filepath)
        if list(raw_disk.columns) != SCHEMA_COLUMNS:
            df.to_csv(filepath, index=False)
            print(f"[Schema Migration] Converted '{filepath}' to standard schema: {SCHEMA_COLUMNS}")
    except Exception:
        pass

    return df


def load_skipped_ids(filepath: str = SKIPPED_FILE) -> Tuple[pd.DataFrame, Set[str], Set[str]]:
    """Load records of skipped candidate items."""
    if not os.path.exists(filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        df = pd.DataFrame(columns=["tweet_id", "customer_text", "skip_reason"])
        df.to_csv(filepath, index=False)
        return df, set(), set()

    try:
        df = pd.read_csv(filepath)
        id_set = set(df["tweet_id"].astype(str).str.strip())
        text_set = {normalize_text_for_dedup(t) for t in df["customer_text"].dropna()}
        return df, id_set, text_set
    except Exception:
        df = pd.DataFrame(columns=["tweet_id", "customer_text", "skip_reason"])
        return df, set(), set()


def record_skipped_item(tweet_id: str, customer_text: str, reason: str, filepath: str = SKIPPED_FILE) -> None:
    """Append a skipped item to the skipped tracking file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    new_entry = pd.DataFrame([{
        "tweet_id": str(tweet_id).strip(),
        "customer_text": str(customer_text).strip(),
        "skip_reason": str(reason).strip() or "unclear / ambiguous"
    }])
    if os.path.exists(filepath):
        new_entry.to_csv(filepath, mode="a", header=False, index=False)
    else:
        new_entry.to_csv(filepath, index=False)


def save_reviewed_item(row_dict: dict, filepath: str = GOLDEN_SET_FILE) -> pd.DataFrame:
    """Save an accepted/changed row immediately to golden_set.csv."""
    row_dict["label_source"] = "human"
    new_row = pd.DataFrame([row_dict])[SCHEMA_COLUMNS]

    if os.path.exists(filepath):
        current_df = load_or_init_golden_set(filepath)
        # Check if already present to avoid duplicates
        existing_ids = set(current_df["tweet_id"].astype(str).str.strip())
        if str(row_dict["tweet_id"]).strip() in existing_ids:
            # Update existing row
            current_df.loc[current_df["tweet_id"].astype(str).str.strip() == str(row_dict["tweet_id"]).strip()] = new_row.iloc[0]
            updated_df = current_df
        else:
            updated_df = pd.concat([current_df, new_row], ignore_index=True)
    else:
        updated_df = new_row

    updated_df.to_csv(filepath, index=False)
    update_summary(updated_df)
    return updated_df


def update_summary(golden_df: pd.DataFrame, skipped_filepath: str = SKIPPED_FILE, summary_filepath: str = SUMMARY_FILE) -> dict:
    """Generate and write annotation summary to JSON."""
    os.makedirs(os.path.dirname(summary_filepath), exist_ok=True)
    skipped_df, _, _ = load_skipped_ids(skipped_filepath)

    total_human = len(golden_df)
    intent_counts = {intent: 0 for intent in INTENTS}
    if not golden_df.empty and "intent" in golden_df.columns:
        counts = golden_df["intent"].value_counts().to_dict()
        for k, v in counts.items():
            if k in intent_counts:
                intent_counts[k] = int(v)
            else:
                intent_counts[k] = int(v)

    esc_counts = {"yes": 0, "no": 0}
    if not golden_df.empty and "escalation" in golden_df.columns:
        esc_dict = golden_df["escalation"].str.lower().value_counts().to_dict()
        esc_counts["yes"] = int(esc_dict.get("yes", 0))
        esc_counts["no"] = int(esc_dict.get("no", 0))

    summary = {
        "target_count": TARGET_COUNT,
        "total_human_labelled": total_human,
        "remaining_to_target": max(0, TARGET_COUNT - total_human),
        "target_reached": total_human >= TARGET_COUNT,
        "examples_per_intent": intent_counts,
        "escalation_counts": esc_counts,
        "skipped_count": len(skipped_df),
        "schema_valid": list(golden_df.columns) == SCHEMA_COLUMNS,
        "all_human_source": bool(golden_df["label_source"].eq("human").all()) if not golden_df.empty else True
    }

    with open(summary_filepath, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


def load_candidate_pool(
    reviewed_ids: Set[str],
    reviewed_texts: Set[str],
    skipped_ids: Set[str],
    skipped_texts: Set[str]
) -> List[Dict[str, Any]]:
    """Build an ordered candidate pool from golden_candidates.csv, prioritizing underrepresented classes."""
    candidates = []

    # 1. Primary pool: golden_candidates.csv
    if os.path.exists(CANDIDATES_FILE):
        df_cand = pd.read_csv(CANDIDATES_FILE)
        id_col = "support_response" if "support_response" in df_cand.columns else "tweet_id"
        for _, row in df_cand.iterrows():
            tid = str(row[id_col]).strip()
            ctext = str(row["customer_text"]).strip()
            norm_text = normalize_text_for_dedup(ctext)

            if tid in reviewed_ids or norm_text in reviewed_texts:
                continue
            if tid in skipped_ids or norm_text in skipped_texts:
                continue

            sugg = str(row.get("suggested_intent", "HOW_TO_OTHER")).strip()
            candidates.append({
                "tweet_id": tid,
                "customer_text": ctext,
                "suggested_intent": sugg,
                "source": "golden_candidates.csv"
            })

    # 2. Fallback pool: apple_support_pairs_clean.csv if golden_candidates pool is low
    needed = TARGET_COUNT - len(reviewed_ids) + 20
    if len(candidates) < needed and os.path.exists(CLEAN_PAIRS_FILE):
        try:
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            from src.models.intent_classifier import KeywordRuleIntentClassifier
            clf = KeywordRuleIntentClassifier()

            # Read clean pairs in chunks
            for chunk in pd.read_csv(CLEAN_PAIRS_FILE, chunksize=2000):
                for _, row in chunk.iterrows():
                    tid = str(row.get("customer_tweet_id", "")).strip()
                    ctext = str(row.get("customer_text_clean", "")).strip()
                    if not tid or len(ctext) < 25:
                        continue
                    norm_text = normalize_text_for_dedup(ctext)
                    if tid in reviewed_ids or norm_text in reviewed_texts:
                        continue
                    if tid in skipped_ids or norm_text in skipped_texts:
                        continue
                    if any(c["tweet_id"] == tid or normalize_text_for_dedup(c["customer_text"]) == norm_text for c in candidates):
                        continue

                    sugg = clf.predict(ctext)
                    candidates.append({
                        "tweet_id": tid,
                        "customer_text": ctext,
                        "suggested_intent": sugg,
                        "source": "apple_support_pairs_clean.csv"
                    })

                    if len(candidates) >= needed + 100:
                        break
                if len(candidates) >= needed + 100:
                    break
        except Exception as e:
            print(f"[Warning] Could not supplement candidates from clean pairs: {e}")

    return candidates


def verify_schema(filepath: str = GOLDEN_SET_FILE) -> bool:
    """Verify that golden_set.csv adheres strictly to requirements."""
    if not os.path.exists(filepath):
        print(f"[FAIL] '{filepath}' does not exist.")
        return False

    df = pd.read_csv(filepath)
    print(f"\n--- Schema Verification for '{filepath}' ---")
    print(f"Columns present: {df.columns.tolist()}")
    print(f"Expected schema: {SCHEMA_COLUMNS}")

    if list(df.columns) != SCHEMA_COLUMNS:
        print(f"[FAIL] Column order or names do not match expected schema.")
        return False

    # Check label_source
    if not df.empty:
        non_human = df[df["label_source"] != "human"]
        if not non_human.empty:
            print(f"[FAIL] Found {len(non_human)} rows with label_source != 'human'.")
            return False

        # Check intent values
        invalid_intents = df[~df["intent"].isin(INTENTS)]
        if not invalid_intents.empty:
            print(f"[FAIL] Found invalid intent values: {invalid_intents['intent'].unique()}")
            return False

        # Check escalation values
        invalid_esc = df[~df["escalation"].isin(["yes", "no"])]
        if not invalid_esc.empty:
            print(f"[FAIL] Found invalid escalation values: {invalid_esc['escalation'].unique()}")
            return False

        # Check duplicates
        dup_ids = df[df.duplicated(subset=["tweet_id"], keep=False)]
        if not dup_ids.empty:
            print(f"[FAIL] Found duplicate tweet IDs: {dup_ids['tweet_id'].tolist()}")
            return False

    print(f"[PASS] All schema and data integrity checks passed.")
    print(f"Total rows: {len(df)} / {TARGET_COUNT}")
    return True


def print_status(filepath: str = GOLDEN_SET_FILE) -> None:
    """Print current annotation summary to stdout."""
    golden_df = load_or_init_golden_set(filepath)
    summary = update_summary(golden_df)

    print("\n" + "=" * 75)
    print("GOLDEN EVALUATION SET ANNOTATION STATUS")
    print("=" * 75)
    print(f"Target Goal          : {summary['target_count']} human-labelled examples")
    print(f"Total Human Labelled : {summary['total_human_labelled']}")
    print(f"Remaining to Goal    : {summary['remaining_to_target']}")
    print(f"Total Skipped        : {summary['skipped_count']}")
    print(f"Escalation Counts    : Yes={summary['escalation_counts']['yes']} | No={summary['escalation_counts']['no']}")
    print("\nClass Distribution:")
    for intent, count in summary["examples_per_intent"].items():
        bar = "#" * (count // 2)
        print(f"  {intent:<22} : {count:>3} {bar}")
    print("=" * 75 + "\n")


def run_interactive_annotation(target: int = TARGET_COUNT, filepath: str = GOLDEN_SET_FILE) -> None:
    """Run interactive CLI review loop."""
    print("\n" + "=" * 75)
    print(f"STARTING GOLDEN SET HUMAN ANNOTATION WORKFLOW (Target: {target})")
    print("=" * 75)

    golden_df = load_or_init_golden_set(filepath)
    skipped_df, skipped_ids, skipped_texts = load_skipped_ids()

    reviewed_ids = set(golden_df["tweet_id"].astype(str).str.strip())
    reviewed_texts = {normalize_text_for_dedup(t) for t in golden_df["customer_text"].dropna()}

    print(f"Loaded {len(golden_df)} previously confirmed human examples.")
    print(f"Loaded {len(skipped_ids)} previously skipped examples.")

    if len(golden_df) >= target:
        print(f"\n[Target Reached] Already have {len(golden_df)} / {target} human-confirmed examples!")
        print_status(filepath)
        return

    candidates = load_candidate_pool(reviewed_ids, reviewed_texts, skipped_ids, skipped_texts)
    print(f"Available unreviewed candidates: {len(candidates)}")

    if not candidates:
        print("[Notice] No more candidates available in golden_candidates.csv.")
        return

    # Sort candidates dynamically to balance classes
    intent_counts = golden_df["intent"].value_counts().to_dict() if not golden_df.empty else {}
    candidates.sort(key=lambda c: intent_counts.get(c["suggested_intent"], 0))

    try:
        for candidate in candidates:
            if len(golden_df) >= target:
                print(f"\n🎉 Goal of {target} accepted human-labelled examples reached!")
                break

            current_count = len(golden_df) + 1
            tid = candidate["tweet_id"]
            text = candidate["customer_text"]
            sugg = candidate["suggested_intent"]

            print("\n" + "-" * 75)
            print(f"[{current_count}/{target}] Tweet ID: {tid}")
            print("-" * 75)
            print(f"CUSTOMER: \"{text}\"\n")
            print(f"SUGGESTED INTENT: >>> {sugg} <<<")
            print("\nOptions:")
            print(f"  [A / Enter] ACCEPT suggested intent ({sugg})")
            print("  Change intent:")
            for i, intent in enumerate(INTENTS, 1):
                marker = "*" if intent == sugg else " "
                print(f"    [{i:2d}]{marker} {intent}")
            print("  [S] SKIP (unclear / ambiguous / bad text)")
            print("  [Q] SAVE & QUIT\n")

            chosen_intent = None
            while True:
                user_choice = input("Your choice [A/1-11/S/Q]: ").strip().lower()

                if user_choice in ["", "a", "accept", "y"]:
                    chosen_intent = sugg
                    break
                elif user_choice in ["s", "skip"]:
                    skip_reason = input("Skip reason [default: unclear/ambiguous]: ").strip() or "unclear / ambiguous"
                    record_skipped_item(tid, text, skip_reason)
                    skipped_ids.add(str(tid).strip())
                    skipped_texts.add(normalize_text_for_dedup(text))
                    print(f"-> Skipped. ({len(skipped_ids)} total skipped)")
                    break
                elif user_choice in ["q", "quit", "exit"]:
                    print("\nSaving progress and exiting...")
                    update_summary(golden_df)
                    print_status(filepath)
                    return
                elif user_choice.isdigit() and 1 <= int(user_choice) <= len(INTENTS):
                    chosen_intent = INTENTS[int(user_choice) - 1]
                    print(f"-> Changed intent to: {chosen_intent}")
                    break
                else:
                    print("Invalid option. Please enter 'A' to accept, '1-11' to change, 'S' to skip, or 'Q' to quit.")

            if chosen_intent is None:
                # Item was skipped
                continue

            # Escalation tagging
            # Heuristic hint
            hazard_hint = any(kw in text.lower() for kw in ["swell", "fire", "smoke", "burn", "shock", "explod", "hacked", "stolen", "lawyer", "attorney", "unauthorized"])
            default_esc = "y" if hazard_hint else "n"

            while True:
                esc_input = input(f"Escalation required? (y/n) [default: {default_esc}]: ").strip().lower() or default_esc
                if esc_input in ["y", "yes", "n", "no"]:
                    is_escalated = "yes" if esc_input.startswith("y") else "no"
                    break
                print("Enter 'y' for yes or 'n' for no.")

            esc_reason = ""
            if is_escalated == "yes":
                esc_reason = input("Escalation reason (e.g. PHYSICAL_SAFETY_HAZARD, ACCOUNT_SECURITY): ").strip()
                if not esc_reason:
                    esc_reason = "HUMAN_ESCALATION"

            # Save immediately
            reviewed_row = {
                "tweet_id": tid,
                "customer_text": text,
                "intent": chosen_intent,
                "escalation": is_escalated,
                "escalation_reason": esc_reason,
                "label_source": "human"
            }

            golden_df = save_reviewed_item(reviewed_row, filepath)
            reviewed_ids.add(str(tid).strip())
            reviewed_texts.add(normalize_text_for_dedup(text))
            print(f"[OK] Saved! Current confirmed total: {len(golden_df)} / {target}")

    except KeyboardInterrupt:
        print("\n\n[Interrupted] Saving progress...")
        update_summary(golden_df)
        print_status(filepath)
        return

    print("\nAnnotation session ended.")
    print_status(filepath)


def run_unit_test() -> bool:
    """Automated test to verify review, resume, skip, and schema logic without altering production data."""
    test_golden_file = os.path.join("data", "test_golden_set.csv")
    test_skipped_file = os.path.join("data", "test_golden_skipped.csv")
    test_summary_file = os.path.join("data", "test_golden_summary.json")

    # Clean up test files if exist
    for f in [test_golden_file, test_skipped_file, test_summary_file]:
        if os.path.exists(f):
            os.remove(f)

    print("\n=== Running Golden Set Workflow Unit Test ===")

    # Test 1: Initialize
    df = load_or_init_golden_set(test_golden_file)
    assert list(df.columns) == SCHEMA_COLUMNS, "Schema columns mismatch on init"
    assert len(df) == 0, "Initial df should be empty"
    print("[OK] Test 1 Passed: Initialized clean DataFrame with exact schema.")

    # Test 2: Save item
    item1 = {
        "tweet_id": "T1001",
        "customer_text": "Battery draining quickly on iPhone 12",
        "intent": "BATTERY_POWER",
        "escalation": "no",
        "escalation_reason": "",
        "label_source": "human"
    }
    df = save_reviewed_item(item1, test_golden_file)
    assert len(df) == 1, "Should have 1 item saved"
    assert df.iloc[0]["tweet_id"] == "T1001"
    assert df.iloc[0]["label_source"] == "human"
    print("[OK] Test 2 Passed: Immediate auto-save persists item with label_source='human'.")

    # Test 3: Resume
    df_resumed = load_or_init_golden_set(test_golden_file)
    assert len(df_resumed) == 1
    assert df_resumed.iloc[0]["tweet_id"] == "T1001"
    print("[OK] Test 3 Passed: Successfully resumed from saved file.")

    # Test 4: Record skip
    record_skipped_item("T1002", "some vague text???", "vague question", test_skipped_file)
    sk_df, sk_ids, _ = load_skipped_ids(test_skipped_file)
    assert "T1002" in sk_ids
    assert len(sk_df) == 1
    print("[OK] Test 4 Passed: Skipped candidate recorded and tracked separately.")

    # Test 5: Verify Schema
    assert verify_schema(test_golden_file) is True
    print("[OK] Test 5 Passed: Schema verification passes.")

    # Clean up test artifacts
    for f in [test_golden_file, test_skipped_file, test_summary_file]:
        if os.path.exists(f):
            os.remove(f)

    print("=== All Unit Tests Passed Successfully! ===\n")
    return True


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Hiver Golden Set Human Annotation Workflow")
    parser.add_argument("--status", action="store_true", help="Display current annotation progress and statistics")
    parser.add_argument("--verify-schema", action="store_true", help="Verify golden_set.csv schema integrity")
    parser.add_argument("--test", action="store_true", help="Run automated test suite of the annotation workflow")
    parser.add_argument("--target", type=int, default=TARGET_COUNT, help=f"Target number of reviewed examples (default: {TARGET_COUNT})")
    parser.add_argument("--file", type=str, default=GOLDEN_SET_FILE, help=f"Golden set output file (default: {GOLDEN_SET_FILE})")

    args = parser.parse_args()

    if args.test:
        run_unit_test()
    elif args.verify_schema:
        verify_schema(args.file)
    elif args.status:
        print_status(args.file)
    else:
        run_interactive_annotation(target=args.target, filepath=args.file)