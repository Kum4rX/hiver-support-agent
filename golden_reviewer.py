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
from datetime import datetime, timezone
from typing import Dict, List, Set, Tuple, Optional, Any
import pandas as pd

# Constants
TARGET_COUNT = 200
GOLDEN_SET_FILE = "golden_set.csv"
PROVISIONAL_FILE = os.path.join("data", "golden_evaluation_provisional.csv")
CANDIDATES_FILE = "golden_candidates.csv"
CLEAN_PAIRS_FILE = os.path.join("data", "apple_support_pairs_clean.csv")
SKIPPED_FILE = os.path.join("data", "golden_skipped.csv")
SUMMARY_FILE = os.path.join("data", "golden_set_summary.json")
AUDIT_LOG_FILE = os.path.join("data", "golden_annotation_audit.jsonl")
BACKUP_BEFORE_BULK_FILE = "golden_set_backup_before_bulk.csv"

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


def record_audit_log(
    reviewer_id: str,
    tweet_id: str,
    customer_text: str,
    suggested_intent: str,
    final_intent: str,
    action: str,
    escalation: str,
    escalation_reason: str,
    label_source: str,
    filepath: str = AUDIT_LOG_FILE
) -> None:
    """Record an audit trail log entry for a human annotation decision."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "reviewer_id": str(reviewer_id).strip(),
        "tweet_id": str(tweet_id).strip(),
        "customer_text": str(customer_text).strip(),
        "suggested_intent": str(suggested_intent).strip(),
        "final_intent": str(final_intent).strip(),
        "action": action,
        "escalation": str(escalation).strip().lower(),
        "escalation_reason": str(escalation_reason).strip(),
        "label_source": str(label_source).strip()
    }
    with open(filepath, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def load_candidate_pool(
    reviewed_ids: Set[str],
    reviewed_texts: Set[str],
    skipped_ids: Set[str],
    skipped_texts: Set[str]
) -> List[Dict[str, Any]]:
    """Build an ordered candidate pool prioritizing unreviewed provisional rows, then golden_candidates.csv."""
    candidates = []
    queued_ids = set()
    queued_texts = set()

    # 1. Primary candidate pool: unreviewed provisional rows from data/golden_evaluation_provisional.csv
    if os.path.exists(PROVISIONAL_FILE):
        try:
            df_prov = pd.read_csv(PROVISIONAL_FILE)
            for _, row in df_prov.iterrows():
                # Only take rows that are NOT already human verified
                if str(row.get("label_source", "")).strip().lower() == "human":
                    continue
                tid = str(row.get("tweet_id", "")).strip()
                ctext = str(row.get("customer_text", "")).strip()
                if not tid or not ctext:
                    continue
                norm_text = normalize_text_for_dedup(ctext)
                if tid in reviewed_ids or norm_text in reviewed_texts:
                    continue
                if tid in skipped_ids or norm_text in skipped_texts:
                    continue
                if tid in queued_ids or norm_text in queued_texts:
                    continue

                sugg = str(row.get("intent", "HOW_TO_OTHER")).strip()
                candidates.append({
                    "tweet_id": tid,
                    "customer_text": ctext,
                    "suggested_intent": sugg,
                    "source": "golden_evaluation_provisional.csv"
                })
                queued_ids.add(tid)
                queued_texts.add(norm_text)
        except Exception as e:
            print(f"[Warning] Could not read provisional candidates: {e}")

    # 2. Secondary pool: golden_candidates.csv
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
            if tid in queued_ids or norm_text in queued_texts:
                continue

            sugg = str(row.get("suggested_intent", "HOW_TO_OTHER")).strip()
            candidates.append({
                "tweet_id": tid,
                "customer_text": ctext,
                "suggested_intent": sugg,
                "source": "golden_candidates.csv"
            })
            queued_ids.add(tid)
            queued_texts.add(norm_text)

    # 3. Fallback pool: apple_support_pairs_clean.csv if candidate pool is low
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
                    if tid in queued_ids or norm_text in queued_texts:
                        continue

                    sugg = clf.predict(ctext)
                    candidates.append({
                        "tweet_id": tid,
                        "customer_text": ctext,
                        "suggested_intent": sugg,
                        "source": "apple_support_pairs_clean.csv"
                    })
                    queued_ids.add(tid)
                    queued_texts.add(norm_text)

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

        # Check duplicate tweet IDs
        dup_ids = df[df.duplicated(subset=["tweet_id"], keep=False)]
        if not dup_ids.empty:
            print(f"[FAIL] Found duplicate tweet IDs: {dup_ids['tweet_id'].tolist()}")
            return False

        # Check duplicate normalized text
        norm_texts = df["customer_text"].apply(normalize_text_for_dedup)
        dup_text_mask = norm_texts.duplicated(keep=False)
        if dup_text_mask.any():
            dup_texts = df.loc[dup_text_mask, "customer_text"].tolist()
            print(f"[FAIL] Found duplicate normalized customer texts: {dup_texts[:3]} (total duplicates: {dup_text_mask.sum()})")
            return False

        # Check preservation of pre-existing human labels from backup if present
        if os.path.exists(BACKUP_BEFORE_BULK_FILE) and filepath == GOLDEN_SET_FILE:
            try:
                df_bk = pd.read_csv(BACKUP_BEFORE_BULK_FILE)
                bk_id_map = dict(zip(df_bk["tweet_id"].astype(str).str.strip(), df_bk["intent"].astype(str).str.strip()))
                bk_esc_map = dict(zip(df_bk["tweet_id"].astype(str).str.strip(), df_bk["escalation"].astype(str).str.strip()))
                current_id_map = dict(zip(df["tweet_id"].astype(str).str.strip(), df["intent"].astype(str).str.strip()))
                current_esc_map = dict(zip(df["tweet_id"].astype(str).str.strip(), df["escalation"].astype(str).str.strip()))

                missing_bk = [tid for tid in bk_id_map if tid not in current_id_map]
                if missing_bk:
                    print(f"[FAIL] Pre-existing human label(s) missing from {filepath}: {missing_bk}")
                    return False

                mismatched_bk = [tid for tid in bk_id_map if bk_id_map[tid] != current_id_map.get(tid)]
                if mismatched_bk:
                    print(f"[FAIL] Pre-existing human label intent modified for: {mismatched_bk}")
                    return False

                mismatched_esc = [tid for tid in bk_esc_map if bk_esc_map[tid] != current_esc_map.get(tid)]
                if mismatched_esc:
                    print(f"[FAIL] Pre-existing human label escalation modified for: {mismatched_esc}")
                    return False

                print(f"[PASS] Preserved all {len(df_bk)} pre-existing human-labelled rows exactly intact.")
            except Exception as e:
                print(f"[Warning] Could not verify backup preservation: {e}")

    print(f"[PASS] All schema and data integrity checks passed.")
    if len(df) == TARGET_COUNT:
        print(f"[PASS] Target count verified: exactly {TARGET_COUNT} human-confirmed examples!")
    else:
        print(f"[PROGRESS] Current count: {len(df)} / {TARGET_COUNT} human-confirmed examples ({TARGET_COUNT - len(df)} remaining to complete).")
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


def export_clean_golden_set(filepath: str = GOLDEN_SET_FILE) -> bool:
    """Validate and export a clean golden_set.csv containing ONLY genuinely human-reviewed rows."""
    print(f"\n--- Exporting Clean Golden Set from '{filepath}' ---")
    if not os.path.exists(filepath):
        print(f"[FAIL] '{filepath}' does not exist.")
        return False

    df = pd.read_csv(filepath)
    initial_count = len(df)
    print(f"Total input rows: {initial_count}")

    # 1. Filter strictly for label_source == 'human'
    human_mask = df["label_source"].astype(str).str.strip().str.lower() == "human"
    df_clean = df[human_mask].copy()
    if len(df_clean) < initial_count:
        print(f"[Warning] Excluded {initial_count - len(df_clean)} non-human rows.")

    # 2. Schema check
    for col in SCHEMA_COLUMNS:
        if col not in df_clean.columns:
            df_clean[col] = ""

    # 3. Deduplicate by tweet_id and customer_text
    df_clean = df_clean.drop_duplicates(subset=["tweet_id"], keep="first")
    df_clean["_norm_text"] = df_clean["customer_text"].apply(normalize_text_for_dedup)
    df_clean = df_clean.drop_duplicates(subset=["_norm_text"], keep="first")
    df_clean = df_clean.drop(columns=["_norm_text"])

    # 4. Validate intent
    valid_intents = df_clean["intent"].isin(INTENTS)
    if not valid_intents.all():
        invalid = df_clean[~valid_intents]["intent"].tolist()
        print(f"[FAIL] Found invalid intents: {invalid}")
        return False

    # 5. Reorder schema
    df_clean = df_clean[SCHEMA_COLUMNS].copy()

    # 6. Save back
    df_clean.to_csv(filepath, index=False)
    summary = update_summary(df_clean)
    print(f"[PASS] Successfully exported clean golden set ({len(df_clean)} rows).")
    print(f"Target count: {TARGET_COUNT} | Confirmed human labels: {summary['total_human_labelled']}")
    return True


def run_interactive_annotation(
    target: int = TARGET_COUNT,
    filepath: str = GOLDEN_SET_FILE,
    reviewer_id: str = "human_reviewer"
) -> None:
    """Run interactive CLI review loop."""
    print("\n" + "=" * 78)
    print(f"STARTING GOLDEN SET HUMAN ANNOTATION WORKFLOW (Target: {target})")
    print(f"Active Reviewer ID: {reviewer_id}")
    print("=" * 78)

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
        print("[Notice] No more candidates available in provisional pool or golden_candidates.csv.")
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
            src = candidate.get("source", "candidate_pool")

            print("\n" + "=" * 78)
            print("  [SUGGESTED INTENT ONLY — NOT GROUND TRUTH — REQUIRES HUMAN AUDIT]")
            print("=" * 78)
            print(f"[{current_count}/{target}] Tweet ID: {tid} (Source: {src})")
            print("-" * 78)
            print(f"CUSTOMER: \"{text}\"\n")
            print(f"SUGGESTED INTENT: >>> {sugg} <<< (Candidate heuristic prediction)")
            print("-" * 78)
            print("FAST REVIEW COMMANDS:")
            print(f"  [Enter / A]  : ACCEPT suggested intent ({sugg}) | Escalation: NO")
            print("  [1 - 11]     : CHANGE intent to number below    | Escalation: NO")
            print(f"  [E]          : ACCEPT suggested intent ({sugg}) | Escalation: YES (prompt reason)")
            print("  [<num> E]    : CHANGE intent to <num>           | Escalation: YES (e.g. '8e')")
            print("  [S]          : SKIP candidate (unclear / noisy / corrupt text)")
            print("  [Q]          : SAVE progress & QUIT\n")
            print("Taxonomy Classes:")
            for i, intent in enumerate(INTENTS, 1):
                marker = "  <-- SUGGESTED" if intent == sugg else ""
                print(f"    [{i:2d}] {intent}{marker}")
            print("-" * 78)

            chosen_intent = None
            is_escalated = "no"
            esc_reason = ""
            action = "accept"

            while True:
                user_choice = input(f"Reviewer [{reviewer_id}] choice [Enter/1-11/E/S/Q]: ").strip().lower()

                # 1. Accept suggestion, no escalation
                if user_choice in ["", "a", "accept", "y"]:
                    chosen_intent = sugg
                    is_escalated = "no"
                    action = "accepted_suggestion"
                    break

                # 2. Skip candidate
                elif user_choice in ["s", "skip"]:
                    skip_reason = input("Skip reason [default: unclear/ambiguous]: ").strip() or "unclear / ambiguous"
                    record_skipped_item(tid, text, skip_reason)
                    record_audit_log(
                        reviewer_id=reviewer_id,
                        tweet_id=tid,
                        customer_text=text,
                        suggested_intent=sugg,
                        final_intent="",
                        action="skipped",
                        escalation="no",
                        escalation_reason=skip_reason,
                        label_source="skipped"
                    )
                    skipped_ids.add(str(tid).strip())
                    skipped_texts.add(normalize_text_for_dedup(text))
                    print(f"-> Skipped. ({len(skipped_ids)} total skipped)")
                    break

                # 3. Quit
                elif user_choice in ["q", "quit", "exit"]:
                    print("\nSaving progress and exiting...")
                    update_summary(golden_df)
                    print_status(filepath)
                    return

                # 4. Accept suggestion with escalation
                elif user_choice in ["e", "escalate"]:
                    chosen_intent = sugg
                    is_escalated = "yes"
                    esc_reason = input("Escalation reason (e.g. PHYSICAL_SAFETY_HAZARD, ACCOUNT_SECURITY): ").strip() or "HUMAN_ESCALATION"
                    action = "accepted_escalated"
                    break

                # 5. Change intent with escalation (e.g. '8e' or '8 e')
                elif re.match(r"^(\d+)\s*e(scalate)?$", user_choice):
                    m = re.match(r"^(\d+)\s*e(scalate)?$", user_choice)
                    idx = int(m.group(1))
                    if 1 <= idx <= len(INTENTS):
                        chosen_intent = INTENTS[idx - 1]
                        is_escalated = "yes"
                        esc_reason = input("Escalation reason (e.g. PHYSICAL_SAFETY_HAZARD, ACCOUNT_SECURITY): ").strip() or "HUMAN_ESCALATION"
                        action = "changed_escalated"
                        print(f"-> Changed intent to {chosen_intent} with Escalation: YES")
                        break
                    else:
                        print(f"Invalid intent number: {idx}. Must be between 1 and {len(INTENTS)}.")

                # 6. Change intent without escalation (1-11)
                elif user_choice.isdigit() and 1 <= int(user_choice) <= len(INTENTS):
                    chosen_intent = INTENTS[int(user_choice) - 1]
                    is_escalated = "no"
                    action = "changed_intent"
                    print(f"-> Changed intent to: {chosen_intent} (Escalation: NO)")
                    break

                else:
                    print("Invalid option. Press [Enter] to accept, [1-11] to change, [E] to escalate, [S] to skip, [Q] to quit.")

            if chosen_intent is None:
                # Item was skipped
                continue

            # Save immediately to golden_set.csv
            reviewed_row = {
                "tweet_id": tid,
                "customer_text": text,
                "intent": chosen_intent,
                "escalation": is_escalated,
                "escalation_reason": esc_reason,
                "label_source": "human"
            }

            golden_df = save_reviewed_item(reviewed_row, filepath)
            record_audit_log(
                reviewer_id=reviewer_id,
                tweet_id=tid,
                customer_text=text,
                suggested_intent=sugg,
                final_intent=chosen_intent,
                action=action,
                escalation=is_escalated,
                escalation_reason=esc_reason,
                label_source="human"
            )
            reviewed_ids.add(str(tid).strip())
            reviewed_texts.add(normalize_text_for_dedup(text))
            print(f"[OK] Saved! Confirmed human total: {len(golden_df)} / {target}")

    except KeyboardInterrupt:
        print("\n\n[Interrupted] Saving progress...")
        update_summary(golden_df)
        print_status(filepath)
        return

    print("\nAnnotation session ended.")
    print_status(filepath)


def analyze_candidate(candidate: Dict[str, Any], classifier=None) -> Dict[str, Any]:
    """Analyze a single candidate to produce suggested intent, escalation, confidence, and compact reason."""
    text = str(candidate.get("customer_text", "")).strip()
    tid = str(candidate.get("tweet_id", "")).strip()

    # 1. Suggested Intent
    suggested_intent = candidate.get("suggested_intent")
    confidence = "HIGH"
    if not suggested_intent or suggested_intent not in INTENTS:
        if classifier:
            try:
                suggested_intent = classifier.predict(text)
                confidence = "HIGH"
            except Exception:
                suggested_intent = "HOW_TO_OTHER"
                confidence = "LOW"
        else:
            suggested_intent = "HOW_TO_OTHER"
            confidence = "LOW"

    # 2. Suggested Escalation
    text_lower = text.lower()
    hazard_words = ["swell", "swollen", "bulg", "fire", "smoke", "burn", "spark", "shock", "explod", "melt"]
    security_words = ["hacked", "stolen", "unauthorized", "phishing", "fraud", "stole", "compromised", "lockout"]
    legal_words = ["lawyer", "attorney", "lawsuit", "police", "legal action", "court"]

    is_escalated = "no"
    escalation_reason = ""
    if any(kw in text_lower for kw in hazard_words):
        is_escalated = "yes"
        escalation_reason = "PHYSICAL_SAFETY_HAZARD"
    elif any(kw in text_lower for kw in security_words):
        is_escalated = "yes"
        escalation_reason = "ACCOUNT_SECURITY_RISK"
    elif any(kw in text_lower for kw in legal_words):
        is_escalated = "yes"
        escalation_reason = "LEGAL_DISPUTE"

    # 3. Compact Reason / Summary
    clean_summary = re.sub(r"\s+", " ", text).strip()
    clean_summary = re.sub(r"^(@\w+\s*)+", "", clean_summary).strip()
    if len(clean_summary) > 55:
        clean_summary = clean_summary[:52] + "..."

    return {
        "tweet_id": tid,
        "customer_text": text,
        "suggested_intent": suggested_intent,
        "suggested_escalation": is_escalated,
        "suggested_escalation_reason": escalation_reason,
        "confidence": confidence,
        "short_reason": clean_summary,
        "source": candidate.get("source", "candidate_pool")
    }


def run_bulk_review(
    target: int = TARGET_COUNT,
    batch_size: int = 25,
    filepath: str = GOLDEN_SET_FILE,
    reviewer_id: str = "human_reviewer"
) -> None:
    """Run fast bulk human-in-the-loop review workflow in batches."""
    print("\n" + "=" * 90)
    print(f"STARTING BULK HUMAN-IN-THE-LOOP REVIEW WORKFLOW (Target: {target} | Batch Size: {batch_size})")
    print(f"Active Reviewer ID: {reviewer_id}")
    print("=" * 90)

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

    # Load keyword classifier if available for fallback suggestion refinement
    clf = None
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from src.models.intent_classifier import KeywordRuleIntentClassifier
        clf = KeywordRuleIntentClassifier()
    except Exception:
        pass

    candidates = load_candidate_pool(reviewed_ids, reviewed_texts, skipped_ids, skipped_texts)
    print(f"Available unreviewed candidates: {len(candidates)}")

    if not candidates:
        print("[Notice] No candidates available for bulk review.")
        return

    # Sort candidates dynamically to balance classes across the 11 intents
    intent_counts = golden_df["intent"].value_counts().to_dict() if not golden_df.empty else {}
    candidates.sort(key=lambda c: intent_counts.get(c["suggested_intent"], 0))

    batch_number = 0

    while len(golden_df) < target and candidates:
        batch_number += 1
        needed = target - len(golden_df)
        cur_batch_size = min(batch_size, needed, len(candidates))
        raw_batch = candidates[:cur_batch_size]
        candidates = candidates[cur_batch_size:]

        # Analyze candidates in current batch
        batch = [analyze_candidate(c, classifier=clf) for c in raw_batch]

        start_num = len(golden_df) + 1
        end_num = len(golden_df) + len(batch)

        def print_batch_table(current_batch: List[Dict[str, Any]]):
            print("\n" + "=" * 105)
            print(f"Human confirmed: {len(golden_df)} / {target}")
            print(f"Remaining: {target - len(golden_df)}")
            print(f"Current batch: #{start_num}-#{end_num} ({len(current_batch)} candidates)")
            print("=" * 105)
            print("INTENT TAXONOMY REFERENCE:")
            print("  [1] BATTERY_POWER        [2] CONNECTIVITY         [3] CALLS_COMMUNICATION")
            print("  [4] DEVICE_PERFORMANCE   [5] KEYBOARD_INPUT       [6] APPS_MEDIA")
            print("  [7] DISPLAY_AUDIO_CAMERA [8] ACCOUNT_ICLOUD       [9] PURCHASE_PAYMENT")
            print("  [10] HOW_TO_OTHER        [11] SECURITY")
            print("-" * 105)
            print(f" {'#':<5} | {'INTENT':<22} | {'ESCALATION':<11} | {'CONF':<6} | {'CUSTOMER TWEET PREVIEW'}")
            print("-" * 105)
            for idx, item in enumerate(current_batch):
                rnum = start_num + idx
                sugg_intent = item["suggested_intent"]
                esc_str = "YES" if item["suggested_escalation"] == "yes" else "NO"
                if item["suggested_escalation"] == "yes" and item.get("suggested_escalation_reason"):
                    esc_str = f"YES:{item['suggested_escalation_reason'][:6]}"
                conf = item["confidence"]
                reason = item["short_reason"]
                print(f" #{rnum:<4} | {sugg_intent:<22} | {esc_str:<11} | {conf:<6} | {reason}")
            print("-" * 105)

        print_batch_table(batch)

        # Batch human confirmation loop
        while True:
            print("\n==================================================")
            print("BATCH HUMAN CONFIRMATION")
            print("==================================================")
            print("[A] ACCEPT ALL displayed suggestions as HUMAN REVIEWED")
            print("[C] CHANGE selected rows (e.g. '19 2', '20 4', '21 8E')")
            print("[R] REVIEW rows individually")
            print("[S] SKIP batch")
            print("[Q] SAVE & EXIT")
            print("==================================================")

            user_action = input(f"Reviewer [{reviewer_id}] Batch Action [A/C/R/S/Q]: ").strip().lower()

            if user_action in ["a", "accept", "y"]:
                # ACCEPT ALL: Explicit human action
                print(f"\n[Human Action: ACCEPT ALL] Recording {len(batch)} candidates as label_source='human'...")
                new_rows = []
                for idx, item in enumerate(batch):
                    tid = item["tweet_id"]
                    text = item["customer_text"]
                    norm_text = normalize_text_for_dedup(text)
                    if tid in reviewed_ids or norm_text in reviewed_texts:
                        continue

                    row_dict = {
                        "tweet_id": tid,
                        "customer_text": text,
                        "intent": item["suggested_intent"],
                        "escalation": item["suggested_escalation"],
                        "escalation_reason": item["suggested_escalation_reason"],
                        "label_source": "human"
                    }
                    new_rows.append(row_dict)
                    reviewed_ids.add(tid)
                    reviewed_texts.add(norm_text)
                    record_audit_log(
                        reviewer_id=reviewer_id,
                        tweet_id=tid,
                        customer_text=text,
                        suggested_intent=item["suggested_intent"],
                        final_intent=item["suggested_intent"],
                        action="bulk_accepted",
                        escalation=item["suggested_escalation"],
                        escalation_reason=item["suggested_escalation_reason"],
                        label_source="human"
                    )

                if new_rows:
                    df_append = pd.DataFrame(new_rows)[SCHEMA_COLUMNS]
                    golden_df = pd.concat([golden_df, df_append], ignore_index=True)
                    golden_df.to_csv(filepath, index=False)
                    update_summary(golden_df)
                print(f"[OK] Saved! Human confirmed total: {len(golden_df)} / {target}")
                break  # Advance to next batch

            elif user_action in ["c", "change"]:
                print("\nEnter row changes (e.g. '19 2', '20 4', '21 8E').")
                print("Format: <row_num> <intent_num(1-11)>[E]")
                print("You can enter multiple modifications separated by commas or on separate lines.")
                print("Press Enter on an empty line when done modifying.")

                while True:
                    chg_input = input("Change row(s) [or Enter when done]: ").strip()
                    if not chg_input:
                        break

                    pattern = r"#?(\d+)\s+([1-9]|1[01])\s*(e)?"
                    matches = list(re.finditer(pattern, chg_input, re.IGNORECASE))
                    if not matches:
                        print(f"Could not parse modification '{chg_input}'. Expected format: '<row_num> <intent_num>' (e.g. '19 2' or '21 8E')")
                        continue

                    for m in matches:
                        row_target = int(m.group(1))
                        intent_idx = int(m.group(2))
                        has_esc = bool(m.group(3))

                        offset = row_target - start_num
                        if 0 <= offset < len(batch):
                            chosen_intent = INTENTS[intent_idx - 1]
                            batch[offset]["suggested_intent"] = chosen_intent
                            if has_esc:
                                batch[offset]["suggested_escalation"] = "yes"
                                if not batch[offset]["suggested_escalation_reason"]:
                                    batch[offset]["suggested_escalation_reason"] = "HUMAN_ESCALATION"
                            else:
                                batch[offset]["suggested_escalation"] = "no"
                                batch[offset]["suggested_escalation_reason"] = ""
                            print(f"-> Updated #{row_target} to INTENT: {chosen_intent} | ESCALATION: {batch[offset]['suggested_escalation'].upper()}")
                        else:
                            print(f"Row #{row_target} is outside current batch range (#{start_num}-#{end_num}).")

                # Re-display updated batch table
                print_batch_table(batch)

            elif user_action in ["r", "review"]:
                target_str = input(f"Enter row number(s) to review individually in #{start_num}-#{end_num} (or Enter for all): ").strip()
                if not target_str:
                    target_indices = list(range(len(batch)))
                else:
                    target_indices = []
                    for tok in re.findall(r"\d+", target_str):
                        rnum = int(tok)
                        off = rnum - start_num
                        if 0 <= off < len(batch):
                            target_indices.append(off)
                        else:
                            print(f"Row #{rnum} outside batch range.")

                for off in target_indices:
                    item = batch[off]
                    rnum = start_num + off
                    print("\n" + "-" * 75)
                    print(f"[Reviewing #{rnum}] Tweet ID: {item['tweet_id']}")
                    print(f"CUSTOMER: \"{item['customer_text']}\"")
                    print(f"CURRENT: >>> {item['suggested_intent']} <<< | Escalation: {item['suggested_escalation']}")
                    print("Options: [Enter] keep, [1-11] change intent, [E] escalate, [1-11E] both:")
                    u_in = input("Choice: ").strip().lower()
                    if u_in and u_in.isdigit() and 1 <= int(u_in) <= len(INTENTS):
                        item["suggested_intent"] = INTENTS[int(u_in) - 1]
                    elif re.match(r"^(\d+)\s*e$", u_in):
                        m = re.match(r"^(\d+)\s*e$", u_in)
                        item["suggested_intent"] = INTENTS[int(m.group(1)) - 1]
                        item["suggested_escalation"] = "yes"
                        item["suggested_escalation_reason"] = "HUMAN_ESCALATION"
                    elif u_in == "e":
                        item["suggested_escalation"] = "yes"
                        item["suggested_escalation_reason"] = "HUMAN_ESCALATION"

                print_batch_table(batch)

            elif user_action in ["s", "skip"]:
                print(f"Skipping batch #{batch_number} ({len(batch)} candidates)...")
                for item in batch:
                    record_skipped_item(item["tweet_id"], item["customer_text"], "skipped_in_bulk")
                    record_audit_log(
                        reviewer_id=reviewer_id,
                        tweet_id=item["tweet_id"],
                        customer_text=item["customer_text"],
                        suggested_intent=item["suggested_intent"],
                        final_intent="",
                        action="bulk_skipped",
                        escalation="no",
                        escalation_reason="skipped_in_bulk",
                        label_source="skipped"
                    )
                    skipped_ids.add(str(item["tweet_id"]))
                    skipped_texts.add(normalize_text_for_dedup(item["customer_text"]))
                break  # Next batch

            elif user_action in ["q", "quit", "exit"]:
                print("\nSaving progress and exiting...")
                update_summary(golden_df)
                print_status(filepath)
                return

            else:
                print("Invalid choice. Please enter 'A' to accept, 'C' to change, 'R' to review, 'S' to skip, or 'Q' to quit.")

    print("\n" + "=" * 90)
    if len(golden_df) >= target:
        print(f"🎉 GOAL REACHED! Exactly {len(golden_df)} human-confirmed examples in '{filepath}'.")
    else:
        print(f"Bulk review ended. Total human confirmed: {len(golden_df)} / {target}.")
    print("=" * 90)
    print_status(filepath)


def run_unit_test() -> bool:
    """Automated test to verify review, resume, skip, audit, bulk review analysis, and schema logic without altering production data."""
    test_golden_file = os.path.join("data", "test_golden_set.csv")
    test_skipped_file = os.path.join("data", "test_golden_skipped.csv")
    test_summary_file = os.path.join("data", "test_golden_summary.json")
    test_audit_file = os.path.join("data", "test_golden_audit.jsonl")

    # Clean up test files if exist
    for f in [test_golden_file, test_skipped_file, test_summary_file, test_audit_file]:
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

    # Test 3: Audit logging
    record_audit_log(
        reviewer_id="tester",
        tweet_id="T1001",
        customer_text="Battery draining quickly on iPhone 12",
        suggested_intent="BATTERY_POWER",
        final_intent="BATTERY_POWER",
        action="accepted_suggestion",
        escalation="no",
        escalation_reason="",
        label_source="human",
        filepath=test_audit_file
    )
    assert os.path.exists(test_audit_file), "Audit log file should exist"
    with open(test_audit_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert entry["reviewer_id"] == "tester"
        assert entry["action"] == "accepted_suggestion"
    print("[OK] Test 3 Passed: Audit trail correctly logged.")

    # Test 4: Resume
    df_resumed = load_or_init_golden_set(test_golden_file)
    assert len(df_resumed) == 1
    assert df_resumed.iloc[0]["tweet_id"] == "T1001"
    print("[OK] Test 4 Passed: Successfully resumed from saved file.")

    # Test 5: Record skip
    record_skipped_item("T1002", "some vague text???", "vague question", test_skipped_file)
    sk_df, sk_ids, _ = load_skipped_ids(test_skipped_file)
    assert "T1002" in sk_ids
    assert len(sk_df) == 1
    print("[OK] Test 5 Passed: Skipped candidate recorded and tracked separately.")

    # Test 6: Verify Schema
    assert verify_schema(test_golden_file) is True
    print("[OK] Test 6 Passed: Schema verification passes.")

    # Test 7: Export Clean
    assert export_clean_golden_set(test_golden_file) is True
    print("[OK] Test 7 Passed: Export clean golden set passes.")

    # Test 8: Candidate Analysis logic
    sample_cand = {
        "tweet_id": "T9999",
        "customer_text": "Battery is swelling and smoking hot",
        "suggested_intent": "BATTERY_POWER"
    }
    analysis = analyze_candidate(sample_cand)
    assert analysis["suggested_intent"] == "BATTERY_POWER"
    assert analysis["suggested_escalation"] == "yes"
    assert analysis["suggested_escalation_reason"] == "PHYSICAL_SAFETY_HAZARD"
    assert analysis["confidence"] == "HIGH"
    print("[OK] Test 8 Passed: Candidate analysis correctly flags physical hazard escalation.")

    # Test 9: Modification pattern regex parsing
    pattern = r"#?(\d+)\s+([1-9]|1[01])\s*(e)?"
    matches = list(re.finditer(pattern, "19 2, 20 4, 21 8E", re.IGNORECASE))
    assert len(matches) == 3
    assert int(matches[0].group(1)) == 19 and int(matches[0].group(2)) == 2 and not matches[0].group(3)
    assert int(matches[1].group(1)) == 20 and int(matches[1].group(2)) == 4 and not matches[1].group(3)
    assert int(matches[2].group(1)) == 21 and int(matches[2].group(2)) == 8 and bool(matches[2].group(3))
    print("[OK] Test 9 Passed: Bulk modification parser parses multi-row modifications correctly.")

    # Clean up test artifacts
    for f in [test_golden_file, test_skipped_file, test_summary_file, test_audit_file]:
        if os.path.exists(f):
            os.remove(f)

    print("=== All Unit Tests Passed Successfully! ===\n")
    return True


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Hiver Golden Set Human Annotation Workflow")
    parser.add_argument("--status", action="store_true", help="Display current annotation progress and statistics")
    parser.add_argument("--verify-schema", action="store_true", help="Verify golden_set.csv schema integrity")
    parser.add_argument("--export-clean", action="store_true", help="Export and validate clean golden_set.csv")
    parser.add_argument("--test", action="store_true", help="Run automated test suite of the annotation workflow")
    parser.add_argument("--bulk-review", action="store_true", help="Launch fast bulk human-in-the-loop review workflow")
    parser.add_argument("--batch-size", type=int, default=25, help="Number of candidates per bulk review batch (default: 25)")
    parser.add_argument("--target", type=int, default=TARGET_COUNT, help=f"Target number of reviewed examples (default: {TARGET_COUNT})")
    parser.add_argument("--reviewer", type=str, default="human_reviewer", help="Reviewer ID for audit trail tracking")
    parser.add_argument("--file", type=str, default=GOLDEN_SET_FILE, help=f"Golden set output file (default: {GOLDEN_SET_FILE})")

    args = parser.parse_args()

    if args.test:
        run_unit_test()
    elif args.verify_schema:
        verify_schema(args.file)
    elif args.export_clean:
        export_clean_golden_set(args.file)
    elif args.status:
        print_status(args.file)
    elif args.bulk_review:
        run_bulk_review(target=args.target, batch_size=args.batch_size, filepath=args.file, reviewer_id=args.reviewer)
    else:
        run_interactive_annotation(target=args.target, filepath=args.file, reviewer_id=args.reviewer)