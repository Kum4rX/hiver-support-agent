"""Interactive CLI Demo for Hiver AppleSupport AI Agent.

Allows users to test real customer queries or pick from curated examples,
displaying full step-by-step pipeline diagnostics in real time.
"""

import argparse
import sys
import os

# Set UTF-8 encoding for Windows terminals
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.pipeline.agent_pipeline import SupportAgentPipeline

SAMPLE_PRESETS = [
    ("Battery Drain (Standard Query)", "My iPhone 8 battery drains from 100% to 15% in two hours after updating to iOS 11."),
    ("Critical Safety Hazard (Escalation)", "My charger is smoking, melted the wire, and the phone battery is expanding!"),
    ("Out-of-Domain (Non-Apple Device)", "Can you help me fix the blue screen of death on my Windows 11 Dell laptop?"),
    ("Multi-Intent Query", "My iPhone battery dies in 2 hours and my WiFi also won't connect to the router."),
    ("Wi-Fi Connectivity", "I cannot connect my iPad to my home Wi-Fi network, keeps saying invalid password."),
    ("Autocorrect Glitch", "The letter 'i' is automatically changing to an exclamation mark symbol when typing."),
    ("Account Security Alert", "Someone hacked my Apple ID, changed my email, and locked me out of my phone!"),
    ("Billing Dispute", "There is a fraudulent charge of $79 on my Apple Pay that I never authorized.")
]


def print_separator(char="=", length=75):
    print(char * length)


def run_pipeline_query(pipeline: SupportAgentPipeline, query: str):
    """Execute a single query through the pipeline and render formatted diagnostics."""
    print_separator("=")
    print(f"CUSTOMER QUERY: \"{query}\"")
    print_separator("-")

    result = pipeline.process(query)

    # 1. Preprocessing
    print(f"[1. PREPROCESSING]")
    print(f"    Cleaned Query : \"{result['cleaned_query']}\"")

    # 2. Intent Classification
    intent_flag = " (PROVISIONAL MODEL)" if result.get("is_provisional_intent") else ""
    multi_note = f" (Secondary Intent: {result['secondary_intent']})" if result.get("has_multi_intent") else ""
    ood_note = f" (Non-Apple Device: {result['ood_entity']})" if result.get("is_out_of_domain") else ""

    print(f"\n[2. INTENT CLASSIFICATION]")
    print(f"    Intent        : {result['intent']}{multi_note}{ood_note}")
    print(f"    Confidence    : {result['intent_confidence']:.2f}")
    print(f"    Method        : {result['intent_method']}{intent_flag}")

    # 3. Escalation & Safety Engine
    esc = result["escalation_details"]
    print(f"\n[3. DETERMINISTIC ESCALATION ENGINE]")
    if result["is_escalated"]:
        print(f"    Status        : ESCALATED (SAFE == FALSE)")
        print(f"    Category      : {esc['category']} (Severity: {esc['severity']})")
        print(f"    Reason        : {esc['reason']}")
        print(f"    Trigger       : \"{esc.get('matched_pattern')}\"")
        print(f"    Protocol      : {esc['recommended_action']}")
    elif result["is_out_of_domain"]:
        print(f"    Status        : OUT_OF_DOMAIN (Boundary response generated)")
    else:
        print(f"    Status        : SAFE (No risk triggers detected - Proceed to Automated Resolution)")

    # 4. FAISS Dense Retrieval
    if not result["is_escalated"] and not result["is_out_of_domain"]:
        cases = result.get("retrieved_cases", [])
        print(f"\n[4. FAISS RETRIEVAL] ({len(cases)} cases retrieved from 65,239 docs)")
        for i, case in enumerate(cases, 1):
            print(f"    Case #{i} (Similarity: {case['score']:.4f}):")
            print(f"      Customer: \"{case['customer_text'][:80]}...\"")
            print(f"      Support : \"{case['support_response'][:100]}...\"")

    # 5. Guardrails & Output
    gr = result["guardrails"]
    print(f"\n[5. RESPONSE & GUARDRAILS]")
    print(f"    Source        : {result['generation_source']}")
    print(f"    Length Check  : {gr['final_length']}/280 chars (Compliant: {gr['length_compliant']})")
    print(f"    PII Safe      : {gr['pii_safe']}")
    if gr.get("modifications"):
        print(f"    Guardrails Act: {', '.join(gr['modifications'])}")

    print_separator("-")
    print(f"FINAL AGENT REPLY ({len(result['final_response'])} chars | Latency: {result['latency_ms']:.2f} ms):")
    print(f">>> {result['final_response']}")
    print_separator("=")
    print()


def main():
    parser = argparse.ArgumentParser(description="Hiver AppleSupport AI Agent CLI Demo")
    parser.add_argument("--query", "-q", type=str, help="Single customer query to process")
    args = parser.parse_args()

    pipeline = SupportAgentPipeline()

    if args.query:
        run_pipeline_query(pipeline, args.query)
        return

    # Interactive Loop
    print("\n" + "=" * 75)
    print("      HIVER APPLESUPPORT CUSTOMER AGENT - INTERACTIVE CLI DEMO")
    print("=" * 75)
    print("Type a customer query, pick a preset number, or type 'q' to quit.\n")

    while True:
        print("Preset Examples:")
        for idx, (title, sample_q) in enumerate(SAMPLE_PRESETS, 1):
            print(f"  [{idx}] {title}: \"{sample_q[:55]}...\"")
        print("  [c] Enter custom query")
        print("  [q] Quit\n")

        choice = input("Select option (1-8, c, q): ").strip().lower()

        if choice == "q":
            print("\nExiting demo. Goodbye!")
            break
        elif choice == "c":
            custom_q = input("\nEnter customer query: ").strip()
            if custom_q:
                run_pipeline_query(pipeline, custom_q)
        elif choice.isdigit() and 1 <= int(choice) <= len(SAMPLE_PRESETS):
            selected_query = SAMPLE_PRESETS[int(choice) - 1][1]
            run_pipeline_query(pipeline, selected_query)
        else:
            print("Invalid selection. Please enter 1-8, 'c', or 'q'.\n")


if __name__ == "__main__":
    main()
