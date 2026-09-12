import pandas as pd
import numpy as np
import faiss
import pickle
from sentence_transformers import SentenceTransformer
from pathlib import Path

INPUT_FILE = Path("data/retrieval_documents.csv")
INDEX_FILE = Path("data/apple_support_filtered.index")
METADATA_FILE = Path("data/retriever_filtered_metadata.pkl")

print("Loading retrieval documents...")

df = pd.read_csv(
    INPUT_FILE,
    dtype=str,
    low_memory=False
)

documents = df["document"].fillna("").tolist()

print(f"Documents: {len(documents):,}")

print("Loading embedding model...")

model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

print("Creating embeddings...")

embeddings = model.encode(
    documents,
    batch_size=64,
    show_progress_bar=True,
    normalize_embeddings=True
)

embeddings = np.asarray(
    embeddings,
    dtype="float32"
)

print(f"Embedding shape: {embeddings.shape}")

dimension = embeddings.shape[1]

index = faiss.IndexFlatIP(dimension)

index.add(embeddings)

print(f"FAISS index size: {index.ntotal:,}")

faiss.write_index(
    index,
    str(INDEX_FILE)
)

metadata = df[
    [
        "customer_tweet_id",
        "support_tweet_id",
        "customer_text_clean",
        "support_text_clean",
        "document"
    ]
].to_dict("records")

with open(METADATA_FILE, "wb") as f:
    pickle.dump(metadata, f)

print()
print("==============================")
print("FILTERED INDEX BUILD COMPLETE")
print(f"Index: {INDEX_FILE}")
print(f"Metadata: {METADATA_FILE}")
print("==============================")