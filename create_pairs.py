import pandas as pd
from pathlib import Path

INPUT_FILE = Path("data/twcs.csv")
APPLE_FILE = Path("data/apple_support_tweets.csv")
OUTPUT_FILE = Path("data/apple_support_pairs.csv")


def normalize_id(series):
    return (
        pd.to_numeric(series, errors="coerce")
        .astype("Int64")
        .astype(str)
        .replace("<NA>", pd.NA)
    )


# Load AppleSupport replies
apple = pd.read_csv(
    APPLE_FILE,
    dtype=str,
    low_memory=False
)

apple["tweet_id"] = normalize_id(apple["tweet_id"])
apple["in_response_to_tweet_id"] = normalize_id(
    apple["in_response_to_tweet_id"]
)

print(f"AppleSupport tweets: {len(apple):,}")

parent_ids = set(
    apple["in_response_to_tweet_id"].dropna()
)

print(f"Unique parent tweet IDs: {len(parent_ids):,}")


# Read original dataset in chunks
columns = [
    "tweet_id",
    "author_id",
    "inbound",
    "created_at",
    "text"
]

matched_chunks = []

for chunk in pd.read_csv(
    INPUT_FILE,
    usecols=columns,
    dtype=str,
    chunksize=100_000,
    low_memory=False
):
    chunk["tweet_id"] = normalize_id(chunk["tweet_id"])

    matched = chunk[
        chunk["tweet_id"].isin(parent_ids)
    ].copy()

    if not matched.empty:
        matched_chunks.append(matched)


customers = pd.concat(
    matched_chunks,
    ignore_index=True
)

customers = customers.rename(columns={
    "tweet_id": "customer_tweet_id",
    "author_id": "customer_author_id",
    "inbound": "customer_inbound",
    "created_at": "customer_created_at",
    "text": "customer_text"
})


# AppleSupport responses
responses = apple[
    [
        "tweet_id",
        "created_at",
        "text",
        "in_response_to_tweet_id"
    ]
].copy()

responses = responses.rename(columns={
    "tweet_id": "support_tweet_id",
    "created_at": "support_created_at",
    "text": "support_text",
    "in_response_to_tweet_id": "customer_tweet_id"
})


# Merge customer → AppleSupport response
pairs = customers.merge(
    responses,
    on="customer_tweet_id",
    how="inner"
)


# Clean empty messages
pairs = pairs[
    pairs["customer_text"].notna() &
    pairs["support_text"].notna()
]

pairs = pairs[
    pairs["customer_text"].str.strip().ne("") &
    pairs["support_text"].str.strip().ne("")
]


# Remove duplicates
pairs = pairs.drop_duplicates(
    subset=["customer_tweet_id", "support_tweet_id"]
)


# Save
pairs.to_csv(
    OUTPUT_FILE,
    index=False
)

print()
print("==============================")
print(f"FINAL PAIRS: {len(pairs):,}")
print(f"SAVED TO: {OUTPUT_FILE}")
print("==============================")

print("\nSample:")
print(
    pairs[
        ["customer_text", "support_text"]
    ].head(5).to_string(index=False)
)