import pandas as pd
import re
from pathlib import Path

INPUT_FILE = Path("data/apple_support_pairs_clean.csv")
OUTPUT_FILE = Path("data/retrieval_documents.csv")


def is_generic_response(text):
    text = text.lower().strip()

    generic_patterns = [
        "please dm us",
        "dm us",
        "we're here for you",
        "we are here for you",
        "we'd like to help",
        "we'd like to look into this",
        "we'd like to investigate",
        "let's look at this closer",
        "let's take a closer look",
        "thanks for reaching out",
        "what issue are you having",
        "how can we help",
        "we're happy to help",
        "we are happy to help",
        "please send us a dm",
        "continue there",
        "go to dm",
    ]

    return any(pattern in text for pattern in generic_patterns)


def is_useful_customer_message(text):
    text = text.lower().strip()

    # Remove extremely short / vague messages
    if len(text) < 25:
        return False

    vague_messages = [
        "fix it",
        "help",
        "please help",
        "what is this",
        "what's this",
        "thanks",
        "thank you",
        "same thing",
        "this issue",
        "this problem",
        "what's up",
    ]

    if text in vague_messages:
        return False

    return True


df = pd.read_csv(
    INPUT_FILE,
    dtype=str,
    low_memory=False
)

print(f"Original clean pairs: {len(df):,}")

# Remove generic customer messages
df = df[
    df["customer_text_clean"].apply(is_useful_customer_message)
].copy()

print(f"After customer filtering: {len(df):,}")

# Remove generic support replies
df = df[
    ~df["support_text_clean"].apply(is_generic_response)
].copy()

print(f"After support filtering: {len(df):,}")

# Remove duplicates
df = df.drop_duplicates(
    subset=["customer_text_clean"]
)

print(f"After duplicate removal: {len(df):,}")


# Create retrieval documents
df["document"] = (
    "Customer problem: "
    + df["customer_text_clean"]
    + "\n\nHistorical AppleSupport resolution: "
    + df["support_text_clean"]
)

retrieval_df = df[
    [
        "customer_tweet_id",
        "support_tweet_id",
        "customer_text_clean",
        "support_text_clean",
        "document"
    ]
].copy()


retrieval_df.to_csv(
    OUTPUT_FILE,
    index=False
)

print()
print("==============================")
print("RETRIEVAL DATA READY")
print(f"Documents: {len(retrieval_df):,}")
print(f"Saved: {OUTPUT_FILE}")
print("==============================")

print("\nSample documents:\n")

for i, row in retrieval_df.head(5).iterrows():
    print(f"\n--- {i + 1} ---")
    print(row["document"])