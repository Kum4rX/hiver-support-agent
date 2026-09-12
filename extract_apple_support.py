import pandas as pd
from pathlib import Path

INPUT_FILE = Path("data/twcs.csv")
OUTPUT_FILE = Path("data/apple_support_tweets.csv")

chunks = []

for chunk in pd.read_csv(
    INPUT_FILE,
    chunksize=100_000,
    low_memory=False
):
    apple = chunk[
        chunk["author_id"].astype(str).str.lower() == "applesupport"
    ]

    if not apple.empty:
        chunks.append(apple)

apple_support = pd.concat(chunks, ignore_index=True)

print(f"AppleSupport tweets found: {len(apple_support):,}")

replies = apple_support[
    apple_support["in_response_to_tweet_id"].notna()
].copy()

print(f"AppleSupport replies: {len(replies):,}")

replies.to_csv(OUTPUT_FILE, index=False)

print(f"Saved: {OUTPUT_FILE}")