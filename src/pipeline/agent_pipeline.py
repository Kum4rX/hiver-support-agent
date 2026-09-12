"""End-to-End Customer Support Agent Pipeline.

Coordinates the complete workflow:
1. Preprocessing (Unicode normalization, tweet cleaning).
2. Intent Classification (Hybrid ML + Rule Engine + Out-of-Domain Guard + Multi-Intent).
3. Risk & Escalation Engine (Deterministic safety checks).
4. FAISS Dense Retrieval (Filtered AppleSupport corpus).
5. Grounded Response Generation (Resolution synthesis & boundary enforcement).
6. Response Guardrails (Twitter <= 280 char limit & PII checks).
"""

import time
from typing import Any, Dict, List, Optional

from src.preprocessing import clean_text, is_valid_query
from src.models.intent_classifier import HybridIntentClassifier
from src.models.escalation_engine import EscalationEngine
from src.models.retriever import FaissRetriever
from src.models.generator import CustomerSupportResponseGenerator
from src.models.guardrails import ResponseGuardrails


class SupportAgentPipeline:
    """Orchestrator for the AppleSupport customer inquiry resolution agent."""

    def __init__(
        self,
        index_path: str = "data/apple_support_filtered.index",
        metadata_path: str = "data/retriever_filtered_metadata.pkl",
        model_path: Optional[str] = "data/intent_baseline.joblib",
        max_response_length: int = 280
    ):
        print("Initializing SupportAgentPipeline components...")
        self.intent_classifier = HybridIntentClassifier(model_path=model_path)
        self.escalation_engine = EscalationEngine()
        self.retriever = FaissRetriever(index_path=index_path, metadata_path=metadata_path)
        self.generator = CustomerSupportResponseGenerator()
        self.guardrails = ResponseGuardrails(max_length=max_response_length)
        print("SupportAgentPipeline initialized successfully.")

    def process(
        self,
        raw_query: str,
        top_k: int = 3,
        max_chars: Optional[int] = None
    ) -> Dict[str, Any]:
        """Execute the end-to-end support resolution pipeline for a customer query.
        
        Args:
            raw_query: Raw input tweet or message from customer.
            top_k: Number of historical cases to retrieve from FAISS index.
            max_chars: Character limit for final output (defaults to 280).
            
        Returns:
            Dictionary containing full pipeline trace, decisions, and final response.
        """
        start_time = time.perf_counter()
        limit = max_chars or self.guardrails.max_length

        # 1. Preprocessing
        cleaned_query = clean_text(raw_query)
        if not is_valid_query(cleaned_query, min_chars=3):
            elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
            default_reply = "We're here to help with your Apple device. Could you please provide a few more details about what you're experiencing?"
            return {
                "raw_query": raw_query,
                "cleaned_query": cleaned_query,
                "intent": "HOW_TO_OTHER",
                "secondary_intent": None,
                "has_multi_intent": False,
                "is_out_of_domain": False,
                "ood_entity": None,
                "intent_confidence": 0.0,
                "intent_method": "empty_query_fallback",
                "is_escalated": False,
                "escalation_details": None,
                "retrieved_cases": [],
                "generation_source": "empty_query_fallback",
                "raw_response": default_reply,
                "final_response": default_reply,
                "guardrails": {"all_passed": True, "length_compliant": True, "final_length": len(default_reply), "modifications": []},
                "latency_ms": elapsed_ms
            }

        # 2. Intent Classification (with Out-of-Domain Guard & Multi-Intent detection)
        intent_info = self.intent_classifier.classify(cleaned_query)
        detected_intent = intent_info["intent"]
        secondary_intent = intent_info.get("secondary_intent")
        has_multi_intent = intent_info.get("has_multi_intent", False)
        is_out_of_domain = intent_info.get("is_out_of_domain", False)
        ood_entity = intent_info.get("ood_entity")
        intent_confidence = intent_info.get("confidence", 0.0)

        # 3. Deterministic Escalation & Safety Engine
        escalation_info = self.escalation_engine.evaluate(cleaned_query)
        is_escalated = escalation_info["is_escalated"]

        retrieved_cases: List[Dict[str, Any]] = []

        if is_escalated:
            # High-risk / safety issue: route immediately via safety protocol
            gen_result = self.generator.generate_response(
                query=cleaned_query,
                intent=detected_intent,
                retrieved_cases=[],
                escalation_result=escalation_info
            )
        elif is_out_of_domain:
            # Non-Apple query: return polite boundary response
            gen_result = self.generator.generate_response(
                query=cleaned_query,
                intent=detected_intent,
                retrieved_cases=[],
                escalation_result=None,
                is_out_of_domain=True,
                ood_entity=ood_entity
            )
        else:
            # Safe for automated resolution: retrieve from FAISS index
            retrieved_cases = self.retriever.retrieve(cleaned_query, k=top_k)
            gen_result = self.generator.generate_response(
                query=cleaned_query,
                intent=detected_intent,
                retrieved_cases=retrieved_cases,
                escalation_result=None,
                secondary_intent=secondary_intent,
                is_out_of_domain=False
            )

        raw_response = gen_result["response"]

        # 4. Response Guardrails
        guardrail_result = self.guardrails.apply(
            raw_response=raw_response,
            query=cleaned_query,
            is_escalated=is_escalated,
            max_chars=limit
        )

        final_response = guardrail_result["final_response"]
        elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)

        return {
            "raw_query": raw_query,
            "cleaned_query": cleaned_query,
            "intent": detected_intent,
            "secondary_intent": secondary_intent,
            "has_multi_intent": has_multi_intent,
            "is_out_of_domain": is_out_of_domain,
            "ood_entity": ood_entity,
            "intent_confidence": intent_confidence,
            "intent_method": intent_info.get("method", "unknown"),
            "is_provisional_intent": intent_info.get("is_provisional", False),
            "is_escalated": is_escalated,
            "escalation_details": escalation_info,
            "retrieved_cases": retrieved_cases,
            "generation_source": gen_result.get("source", "unknown"),
            "raw_response": raw_response,
            "final_response": final_response,
            "guardrails": guardrail_result,
            "latency_ms": elapsed_ms
        }
