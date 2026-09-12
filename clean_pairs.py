import pandas as pd
import re
from pathlib import Path

INPUT_FILE = Path("data/apple_support_pairs.csv")
OUTPUT_FILE = Path("data/apple_support_pairs_clean.csv")


def clean_text(text):
    if pd.isna(text):
        return ""

    text = str(text)

    # Remove HTML entities
    text = text.replace("&amp;", "&")
    text = text.replace("&gt;", ">")
    text = text.replace("&lt;", "<")

    # Remove URLs
    text = re.sub(r"https?://\S+", " ", text)

    # Remove Twitter mentions
    text = re.sub(r"@\w+", " ", text)

    # Remove excessive whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


df = pd.read_csv(
    INPUT_FILE,
    dtype=str,
    low_memory=False
)

print(f"Original pairs: {len(df):,}")

# Clean customer and support messages
df["customer_text_clean"] = df["customer_text"].apply(clean_text)
df["support_text_clean"] = df["support_text"].apply(clean_text)

# Remove empty messages
df = df[
    (df["customer_text_clean"].str.len() >= 10) &
    (df["support_text_clean"].str.len() >= 10)
].copy()

# Remove duplicate customer messages
df = df.drop_duplicates(
    subset=["customer_text_clean", "support_text_clean"]
)

# Keep only useful columns
df = df[
    [
        "customer_tweet_id",
        "support_tweet_id",
        "customer_text_clean",
        "support_text_clean"
    ]
]

df.to_csv(
    OUTPUT_FILE,
    index=False
)

print(f"Clean pairs: {len(df):,}")
print(f"Removed: {len(pd.read_csv(INPUT_FILE, dtype=str)) - len(df):,}")
print(f"Saved: {OUTPUT_FILE}")

print("\nSample:")
print(
    df[
        ["customer_text_clean", "support_text_clean"]
    ].head(10).to_string(index=False)
)