import pandas as pd
import numpy as np
import re

INPUT = "data/apple_support_pairs_clean.csv"
OUTPUT = "golden_candidates.csv"

TARGET_PER_INTENT = 20

INTENTS = {
    "BATTERY_POWER": [
        "battery", "charging", "charge", "drain", "power", "dies", "dying"
    ],
    "CONNECTIVITY": [
        "wifi", "wi-fi", "bluetooth", "internet", "network", "connection",
        "cellular", "lte", "signal"
    ],
    "CALLS_COMMUNICATION": [
        "call", "calling", "phone call", "facetime", "imessage", "sms",
        "text message", "voicemail"
    ],
    "DEVICE_PERFORMANCE": [
        "slow", "freez", "freeze", "lag", "crash", "restart", "update",
        "ios", "overheat", "performance"
    ],
    "KEYBOARD_INPUT": [
        "keyboard", "autocorrect", "auto correct", "typing", "type",
        "key", "emoji"
    ],
    "APPS_MEDIA": [
        "app", "application", "itunes", "music", "video", "photo",
        "youtube", "instagram", "facebook"
    ],
    "DISPLAY_AUDIO_CAMERA": [
        "screen", "display", "brightness", "camera", "speaker", "sound",
        "audio", "microphone", "volume"
    ],
    "ACCOUNT_ICLOUD": [
        "icloud", "apple id", "appleid", "password", "account",
        "login", "sign in", "signin"
    ],
    "PURCHASE_PAYMENT": [
        "purchase", "buy", "payment", "paid", "refund", "charge",
        "billing", "credit card", "subscription", "app store"
    ],
    "HOW_TO_OTHER": [
        "how do i", "how can i", "where do i", "can i", "how to",
        "setting", "settings"
    ],
    "SECURITY": [
        "hack", "hacked", "phishing", "scam", "fraud", "stolen",
        "security", "suspicious", "privacy"
    ]
}

def score_intent(text, keywords):
    text = str(text).lower()
    return sum(1 for k in keywords if k in text)

df = pd.read_csv(INPUT)

# Customer message column
text_col = "customer_text_clean"
support_col = "support_text_clean"

# Find support response column
if "support_text" in df.columns:
    support_col = "support_text"
elif "response" in df.columns:
    support_col = "response"
else:
    # inspect common names
    possible = [c for c in df.columns if "support" in c.lower() or "response" in c.lower()]
    if possible:
        support_col = possible[0]
    else:
        support_col = None

df[text_col] = df[text_col].fillna("").astype(str)

# Remove very short / vague examples
df = df[df[text_col].str.len() >= 30].copy()

# Remove duplicates
df = df.drop_duplicates(subset=[text_col])

# Score each example
df["_scores"] = df[text_col].apply(
    lambda x: {intent: score_intent(x, kws) for intent, kws in INTENTS.items()}
)

# Select up to TARGET_PER_INTENT examples per category.
# Use deterministic random seed for reproducibility.
selected = []

for intent in INTENTS:
    candidates = df[
        df["_scores"].apply(lambda s: s[intent] > 0)
    ].copy()

    if len(candidates) == 0:
        continue

    candidates["_score"] = candidates["_scores"].apply(lambda s: s[intent])

    # Prefer examples strongly matching this intent,
    # while keeping some randomness for diversity.
    candidates = candidates.sort_values(
        "_score", ascending=False
    )

    # Take a larger pool, shuffle it deterministically,
    # then select the target count.
    pool = candidates.head(min(len(candidates), TARGET_PER_INTENT * 5))
    pool = pool.sample(frac=1, random_state=42)

    chosen = pool.head(TARGET_PER_INTENT).copy()
    chosen["suggested_intent"] = intent
    selected.append(chosen)

result = pd.concat(selected, ignore_index=True)

# Remove duplicate customer messages across suggested intents
result = result.drop_duplicates(subset=[text_col])

# Limit to approximately 200
result = result.head(220).copy()

# Create clean labeling columns
out = pd.DataFrame()

if "tweet_id" in result.columns:
    out["tweet_id"] = result["tweet_id"]

out["customer_text"] = result[text_col]

if support_col:
    out["support_response"] = result[support_col]

out["suggested_intent"] = result["suggested_intent"]

# HUMAN labels must be filled by the reviewer.
out["intent"] = ""
out["escalation"] = ""
out["escalation_reason"] = ""

out.to_csv(OUTPUT, index=False, encoding="utf-8-sig")

print("=" * 70)
print("GOLDEN CANDIDATES CREATED")
print("=" * 70)
print(f"Candidates: {len(out)}")
print(f"Saved to:   {OUTPUT}")
print()
print("IMPORTANT:")
print("- suggested_intent is ONLY a suggestion.")
print("=" * 70)
print("GOLDEN CANDIDATES CREATED")
print("=" * 70)
print(f"Candidates: {len(out)}")
print(f"Saved to:   {OUTPUT}")
print()
print("IMPORTANT:")
print("- suggested_intent is ONLY a suggestion.")
print("- Human reviewer must confirm/change intent.")
print("- escalation must be manually labelled.")
print("- Do NOT report suggested_intent as hand-labelled.")
print()
print(out["suggested_intent"].value_counts())