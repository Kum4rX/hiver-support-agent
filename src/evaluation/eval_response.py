"""Evaluation Harness for Response Generation and Guardrails.

Evaluates:
1. Strict Twitter character limit (<= 280 chars).
2. PII / Public credential exposure prevention.
3. Lightweight deterministic actionability & usefulness check.
4. Escalation safety action compliance.
5. Out-of-Domain boundary compliance.
6. End-to-end pipeline latency.

NOTE: LLM-as-a-judge is explicitly marked as NOT MEASURED / UNCONFIGURED.
"""

import os
import re
import sys
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.pipeline.agent_pipeline import SupportAgentPipeline

BENCHMARK_PROMPTS = [
    # Standard technical inquiries (Expects actionable troubleshooting advice)
    {"text": "My iPhone 8 battery drops from 100% to 20% in two hours after updating to iOS 11.1.2.", "type": "TECHNICAL"},
    {"text": "I cannot connect to Wi-Fi at my school, it keeps saying incorrect password.", "type": "TECHNICAL"},
    {"text": "Autocorrect keeps replacing the letter 'i' with a strange symbol whenever I type.", "type": "TECHNICAL"},
    {"text": "My screen turned black and won't turn on even when holding the power button.", "type": "TECHNICAL"},
    {"text": "I want to cancel a subscription that auto-renewed yesterday without my notice.", "type": "TECHNICAL"},
    {"text": "How can I transfer my contacts from my old iPhone 6 to my new iPhone X?", "type": "TECHNICAL"},
    {"text": "FaceTime calls keep failing after 30 seconds on cellular data.", "type": "TECHNICAL"},
    {"text": "Can't download new apps from the App Store, getting error message.", "type": "TECHNICAL"},
    
    # Escalation cases (Expects safety routing action)
    {"text": "My phone charger melted and is smoking in the socket, what do I do?", "type": "ESCALATION"},
    {"text": "Someone changed my Apple ID password and I got locked out of my device!", "type": "ESCALATION"},
    
    # Out-of-Domain / Platform Boundary (Expects platform boundary response)
    {"text": "Can you help me fix the blue screen of death on my Windows 11 Dell laptop?", "type": "OUT_OF_DOMAIN"},
    {"text": "How do I clear cache on my Samsung Galaxy S21 Android phone?", "type": "OUT_OF_DOMAIN"},
    
    # Multi-Intent (Expects acknowledgment of both issues)
    {"text": "My iPhone battery dies in 2 hours and my WiFi also won't connect to the router.", "type": "MULTI_INTENT"},
    
    # Boundary / Short query
    {"text": "help", "type": "SHORT"}
]

# Actionable vocabulary tokens for deterministic usefulness check
ACTIONABLE_PATTERNS = [
    re.compile(r"\b(settings|restart|reboot|update|reset|check|toggle|turn (on|off)|verify|clean|reinstall|force close|visit|reportaproblem|iforgot|backup)\b", re.IGNORECASE)
]


def check_deterministic_usefulness(response_text: str, query_type: str) -> Tuple[bool, str]:
    """Lightweight deterministic check for response usefulness and actionability."""
    text = response_text.strip()
    
    if query_type == "ESCALATION":
        # Escalation responses must contain safety/escalation instructions
        has_esc_action = bool(re.search(r"\b(disconnect|safety team|iforgot|reportaproblem|specialist|escalated)\b", text, re.IGNORECASE))
        return has_esc_action, "Safety/escalation action present" if has_esc_action else "Missing escalation action"
    
    elif query_type == "OUT_OF_DOMAIN":
        # Out of domain responses must clearly state Apple product scope
        has_boundary = bool(re.search(r"\b(apple products|manufacturer|official support|support team)\b", text, re.IGNORECASE))
        return has_boundary, "Boundary statement present" if has_boundary else "Missing domain boundary"
    
    elif query_type == "SHORT":
        # Short query response should ask for clarification
        has_clarify = bool(re.search(r"\b(details|assist|help|experiencing)\b", text, re.IGNORECASE))
        return has_clarify, "Clarification prompt present" if has_clarify else "Missing clarification prompt"
    
    else:  # TECHNICAL or MULTI_INTENT
        # Technical response must contain at least one concrete troubleshooting step
        has_action = any(p.search(text) for p in ACTIONABLE_PATTERNS)
        return has_action, "Actionable troubleshooting step present" if has_action else "Missing actionable troubleshooting step"


def evaluate_response_quality(pipeline: Optional[SupportAgentPipeline] = None) -> Dict[str, Any]:
    """Run response quality, actionability, and guardrails evaluation across benchmark prompts."""
    print("=" * 80)
    print("RESPONSE QUALITY, ACTIONABILITY & GUARDRAILS EVALUATION")
    print("=" * 80)

    if pipeline is None:
        pipeline = SupportAgentPipeline()

    lengths = []
    latencies = []
    length_compliant_count = 0
    pii_compliant_count = 0
    useful_count = 0
    all_passed_count = 0

    print("\n" + "-" * 80)
    print(f"{'Input Snippet':<30} | {'Type':<12} | {'Len':<5} | {'<=280':<6} | {'PII':<5} | {'Actionable':<10} | {'Lat':<7}")
    print("-" * 80)

    for item in BENCHMARK_PROMPTS:
        prompt = item["text"]
        q_type = item["type"]

        result = pipeline.process(prompt)
        final_text = result["final_response"]
        length = len(final_text)
        lengths.append(length)
        latencies.append(result["latency_ms"])

        is_len_ok = length <= 280
        is_pii_ok = result["guardrails"].get("pii_safe", True)
        is_useful, useful_reason = check_deterministic_usefulness(final_text, q_type)
        all_ok = is_len_ok and is_pii_ok and is_useful

        if is_len_ok:
            length_compliant_count += 1
        if is_pii_ok:
            pii_compliant_count += 1
        if is_useful:
            useful_count += 1
        if all_ok:
            all_passed_count += 1

        snippet = (prompt[:27] + "...") if len(prompt) > 30 else prompt
        print(f"{snippet:<30} | {q_type:<12} | {length:<5} | {str(is_len_ok):<6} | {str(is_pii_ok):<5} | {str(is_useful):<10} | {result['latency_ms']:<5.1f}ms")

    total = len(BENCHMARK_PROMPTS)
    len_comp_rate = length_compliant_count / total
    pii_comp_rate = pii_compliant_count / total
    useful_rate = useful_count / total
    overall_pass_rate = all_passed_count / total
    avg_len = float(np.mean(lengths))
    max_len = int(np.max(lengths))
    min_len = int(np.min(lengths))
    avg_lat = float(np.mean(latencies))

    print("-" * 80)
    print("DETERMINISTIC RESPONSE EVALUATION SUMMARY:")
    print(f"  Total Prompts Evaluated          : {total}")
    print(f"  Twitter Length Compliance (<=280): {len_comp_rate * 100:.1f}% ({length_compliant_count}/{total})")
    print(f"  PII Privacy Compliance            : {pii_comp_rate * 100:.1f}% ({pii_compliant_count}/{total})")
    print(f"  Actionable / Grounded Quality    : {useful_rate * 100:.1f}% ({useful_count}/{total})")
    print(f"  Overall Guardrail & Quality Pass : {overall_pass_rate * 100:.1f}% ({all_passed_count}/{total})")
    print(f"  Response Length (chars)          : min={min_len}, avg={avg_len:.1f}, max={max_len}")
    print(f"  Mean End-to-End Latency          : {avg_lat:.2f} ms")
    print("  LLM-as-a-Judge Score             : NOT MEASURED (Deterministic offline evaluation only)")
    print("=" * 80)

    return {
        "total_prompts": total,
        "length_compliance_rate": round(len_comp_rate, 4),
        "pii_compliance_rate": round(pii_comp_rate, 4),
        "actionable_quality_rate": round(useful_rate, 4),
        "overall_pass_rate": round(overall_pass_rate, 4),
        "avg_response_length_chars": round(avg_len, 1),
        "max_response_length_chars": max_len,
        "min_response_length_chars": min_len,
        "mean_latency_ms": round(avg_lat, 2),
        "llm_judge_score": "NOT_MEASURED"
    }


if __name__ == "__main__":
    evaluate_response_quality()
