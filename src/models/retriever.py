"""FAISS-based Dense Retrieval Module using pre-indexed AppleSupport corpus."""

import os
import pickle
from typing import Any, Dict, List, Optional
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


class FaissRetriever:
    """FAISS Dense Retriever using Sentence Transformers for customer support resolution matching."""

    def __init__(
        self,
        index_path: str = "data/apple_support_filtered.index",
        metadata_path: str = "data/retriever_filtered_metadata.pkl",
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    ):
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.model_name = model_name

        self.index = None
        self.metadata = None
        self.encoder = None

        self._load_resources()

    def _load_resources(self) -> None:
        """Load the FAISS index, metadata, and embedding encoder."""
        if not os.path.exists(self.index_path):
            raise FileNotFoundError(f"FAISS index file not found at: {self.index_path}")
        if not os.path.exists(self.metadata_path):
            raise FileNotFoundError(f"Metadata file not found at: {self.metadata_path}")

        # Load FAISS index
        self.index = faiss.read_index(self.index_path)

        # Load metadata
        with open(self.metadata_path, "rb") as f:
            self.metadata = pickle.load(f)

        # Load embedding model
        self.encoder = SentenceTransformer(self.model_name)

    def retrieve(
        self,
        query: str,
        k: int = 3,
        min_similarity_threshold: float = 0.35
    ) -> List[Dict[str, Any]]:
        """Retrieve top-k most similar historical support cases for a given customer query.
        
        Args:
            query: The cleaned customer issue text.
            k: Number of nearest neighbors to return.
            min_similarity_threshold: Minimum cosine similarity score required for relevance.
            
        Returns:
            List of dictionaries containing score, customer query, support reply, and document.
        """
        if not query or not query.strip():
            return []

        # Encode query to normalized 384-dim vector
        query_vector = self.encoder.encode(
            [query],
            normalize_embeddings=True,
            show_progress_bar=False
        )
        query_vector = np.asarray(query_vector, dtype="float32")

        # Perform FAISS similarity search (Inner Product on normalized vectors == Cosine Similarity)
        scores, indices = self.index.search(query_vector, k)

        results: List[Dict[str, Any]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.metadata):
                continue

            item = self.metadata[idx]
            sim_score = float(score)

            results.append({
                "index_id": int(idx),
                "score": round(sim_score, 4),
                "is_relevant": sim_score >= min_similarity_threshold,
                "customer_text": item.get("customer_text_clean", ""),
                "support_response": item.get("support_text_clean", ""),
                "document": item.get("document", ""),
                "customer_tweet_id": item.get("customer_tweet_id", ""),
                "support_tweet_id": item.get("support_tweet_id", "")
            })

        return results

    def format_context(self, retrieved_cases: List[Dict[str, Any]]) -> str:
        """Format retrieved cases into a clean grounding context for generation."""
        if not retrieved_cases:
            return "No relevant historical resolutions found."

        formatted_blocks = []
        for i, case in enumerate(retrieved_cases, 1):
            block = (
                f"[Historical Case #{i}] (Similarity: {case['score']:.2f})\n"
                f"Customer: {case['customer_text']}\n"
                f"AppleSupport Resolution: {case['support_response']}"
            )
            formatted_blocks.append(block)

        return "\n\n".join(formatted_blocks)
