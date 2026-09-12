"""Evaluation Harness for FAISS Retrieval Engine.

Measures cosine similarity score distributions, relevance hit rates, and search latency
across diverse customer problem queries.
"""

import os
import sys
import time
from typing import Any, Dict, List
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.models.retriever import FaissRetriever

SAMPLE_RETRIEVAL_QUERIES = [
    "My iPhone battery is draining in less than 3 hours after updating to iOS 11.",
    "I cannot connect to my home Wi-Fi network even though the password is correct.",
    "The letter 'i' is autocorrecting to an exclamation mark and a question mark symbol.",
    "My iPhone screen is completely black and unresponsive to touch.",
    "How do I cancel my Apple Music subscription before the free trial ends?",
    "My phone microphone is not working during FaceTime calls, but speaker is fine.",
    "I forgot my Apple ID passcode and my account is temporarily disabled.",
    "My phone keeps freezing on the Apple logo when I try to turn it on.",
    "Bluetooth disconnects randomly from my car audio system while driving.",
    "How do I transfer photos from my iPhone to my Windows PC without iTunes?"
]


def evaluate_retrieval_engine(
    index_path: str = "data/apple_support_filtered.index",
    metadata_path: str = "data/retriever_filtered_metadata.pkl",
    k: int = 3,
    min_threshold: float = 0.35
) -> Dict[str, Any]:
    """Evaluate FAISS retriever performance and return structured metrics."""
    print("=" * 75)
    print("FAISS DENSE RETRIEVAL EVALUATION")
    print("=" * 75)

    retriever = FaissRetriever(index_path=index_path, metadata_path=metadata_path)
    print(f"Loaded Index Corpus Size: {retriever.index.ntotal:,} documents")

    top1_scores = []
    top3_avg_scores = []
    latencies_ms = []
    hits = 0

    print("\n" + "-" * 75)
    print(f"{'Query Snippet':<40} | {'Top-1 Sim':<10} | {'Top-3 Avg':<10} | {'Latency':<10}")
    print("-" * 75)

    for query in SAMPLE_RETRIEVAL_QUERIES:
        start = time.perf_counter()
        results = retriever.retrieve(query, k=k, min_similarity_threshold=min_threshold)
        elapsed_ms = (time.perf_counter() - start) * 1000
        latencies_ms.append(elapsed_ms)

        if results:
            t1 = results[0]["score"]
            t3 = np.mean([r["score"] for r in results])
            top1_scores.append(t1)
            top3_avg_scores.append(t3)
            if t1 >= min_threshold:
                hits += 1
        else:
            top1_scores.append(0.0)
            top3_avg_scores.append(0.0)

        snippet = (query[:37] + "...") if len(query) > 40 else query
        print(f"{snippet:<40} | {top1_scores[-1]:<10.4f} | {top3_avg_scores[-1]:<10.4f} | {elapsed_ms:<8.2f} ms")

    total_queries = len(SAMPLE_RETRIEVAL_QUERIES)
    mean_top1 = float(np.mean(top1_scores))
    mean_top3 = float(np.mean(top3_avg_scores))
    mean_latency = float(np.mean(latencies_ms))
    hit_rate = hits / total_queries

    print("-" * 75)
    print("RETRIEVAL SUMMARY METRICS:")
    print(f"  Total Queries Evaluated     : {total_queries}")
    print(f"  Mean Top-1 Cosine Similarity: {mean_top1:.4f}")
    print(f"  Mean Top-3 Avg Similarity   : {mean_top3:.4f}")
    print(f"  Relevance Hit Rate (>=0.35) : {hit_rate * 100:.1f}%")
    print(f"  Mean Retrieval Latency      : {mean_latency:.2f} ms per query")
    print("=" * 75)

    return {
        "total_queries": total_queries,
        "corpus_size": retriever.index.ntotal,
        "mean_top1_similarity": round(mean_top1, 4),
        "mean_top3_similarity": round(mean_top3, 4),
        "relevance_hit_rate": round(hit_rate, 4),
        "mean_latency_ms": round(mean_latency, 2)
    }


if __name__ == "__main__":
    evaluate_retrieval_engine()
