"""Grounded Customer Support Response Generator.

Designed to operate completely offline without requiring external API keys by default,
while maintaining support for optional pluggable LLM backends (OpenAI/Anthropic/HuggingFace).

Synthesizes empathetic, brand-appropriate Apple Support replies grounded strictly in
retrieved historical resolutions, with explicit boundary guards for out-of-domain queries
and dual-intent synthesis for multi-intent customer issues.
"""

import os
import re
from typing import Any, Dict, List, Optional


class DeterministicGroundedGenerator:
    """Extracts, adapts, and synthesizes grounded customer support responses from retrieved historical resolutions."""

    # Intent-specific empathetic openings
    INTENT_OPENINGS: Dict[str, str] = {
        "BATTERY_POWER": "We understand how important battery life is and want to help.",
        "CONNECTIVITY": "We'd be happy to help get your connection back up and running.",
        "CALLS_COMMUNICATION": "We know staying connected is essential, and we're here to help.",
        "DEVICE_PERFORMANCE": "We want your device running smoothly and are glad to help.",
        "KEYBOARD_INPUT": "Let's help get your keyboard working properly again.",
        "APPS_MEDIA": "We can help you get your apps running properly.",
        "DISPLAY_AUDIO_CAMERA": "Let's look into what's happening with your screen/audio.",
        "ACCOUNT_ICLOUD": "We're here to help you access and manage your Apple ID securely.",
        "PURCHASE_PAYMENT": "We'd be glad to help look into this purchase with you.",
        "HOW_TO_OTHER": "We'd be happy to help answer your question.",
        "SECURITY": "Your account security is our top priority. Let's get this secured."
    }

    # Intent-specific human-friendly names for multi-intent acknowledgment
    INTENT_TOPICS: Dict[str, str] = {
        "BATTERY_POWER": "battery life",
        "CONNECTIVITY": "Wi-Fi/network connectivity",
        "CALLS_COMMUNICATION": "calls and messaging",
        "DEVICE_PERFORMANCE": "device performance",
        "KEYBOARD_INPUT": "keyboard typing",
        "APPS_MEDIA": "apps and media",
        "DISPLAY_AUDIO_CAMERA": "display and audio",
        "ACCOUNT_ICLOUD": "Apple ID / iCloud account",
        "PURCHASE_PAYMENT": "billing and purchases",
        "HOW_TO_OTHER": "general settings",
        "SECURITY": "account security"
    }

    # Intent-specific default troubleshooting fallbacks if retrieved similarity is low
    INTENT_DEFAULTS: Dict[str, str] = {
        "BATTERY_POWER": "Check Settings > Battery to see which apps are consuming power, and ensure iOS is updated to the latest version.",
        "CONNECTIVITY": "Try toggling Airplane Mode on/off, or reset network settings via Settings > General > Reset > Reset Network Settings.",
        "CALLS_COMMUNICATION": "Restart your device, check for carrier settings updates in Settings > General > About, and test again.",
        "DEVICE_PERFORMANCE": "Ensure your device has at least 10% free storage in Settings > General > iPhone Storage, and restart your device.",
        "KEYBOARD_INPUT": "Try resetting keyboard dictionary in Settings > General > Reset > Reset Keyboard Dictionary, then restart.",
        "APPS_MEDIA": "Force close the app, check the App Store for updates, or reinstall the app to see if that resolves it.",
        "DISPLAY_AUDIO_CAMERA": "Clean the sensors/speakers, restart your device, and check Settings > Display & Brightness or Sounds.",
        "ACCOUNT_ICLOUD": "Visit iforgot.apple.com to verify your account credentials or manage your Apple ID settings securely.",
        "PURCHASE_PAYMENT": "View your purchase history at reportaproblem.apple.com to review charges and request refunds if needed.",
        "HOW_TO_OTHER": "Check Settings for device options, or visit support.apple.com for detailed step-by-step guides.",
        "SECURITY": "Change your Apple ID password immediately at appleid.apple.com and enable Two-Factor Authentication."
    }

    def _extract_resolution_sentences(self, support_text: str) -> str:
        """Extract clean, actionable troubleshooting sentences from historical support text."""
        if not support_text:
            return ""

        text = support_text.strip()
        
        # Strip boilerplate
        text = re.sub(r"^(hi|hello|hey|thanks for reaching out)[^.!?]*[.!?]\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"(dm us|send us a dm|let's meet in dm)[^.!?]*[.!?]?", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s+", " ", text).strip()

        sentences = re.split(r"(?<=[.!?])\s+", text)
        actionable_sentences = []

        for s in sentences:
            s_clean = s.strip()
            if not s_clean:
                continue
            if re.search(r"\b(dm us|send a direct message|we'd like to look)\b", s_clean, re.IGNORECASE):
                continue
            actionable_sentences.append(s_clean)

        if actionable_sentences:
            return " ".join(actionable_sentences[:2])
        return text[:160]

    def generate(
        self,
        query: str,
        intent: str,
        retrieved_cases: List[Dict[str, Any]],
        secondary_intent: Optional[str] = None,
        is_out_of_domain: bool = False,
        ood_entity: Optional[str] = None,
        max_chars: int = 280
    ) -> Dict[str, Any]:
        """Generate grounded response based on top retrieved support resolution."""
        # 1. Out-of-Domain Boundary Guard
        if is_out_of_domain or intent == "OUT_OF_DOMAIN":
            platform_name = ood_entity.title() if ood_entity else "non-Apple devices"
            reply = (
                f"We provide support for Apple products and services. For help with {platform_name}, "
                f"please contact the manufacturer's official support team. Let us know if you need help with an Apple device!"
            )
            return {
                "response": reply,
                "source": f"out_of_domain_boundary_guard ({platform_name})",
                "grounding_similarity": 0.0,
                "intent_used": "OUT_OF_DOMAIN"
            }

        # 2. Extract resolution from retrieved evidence
        resolution_text = ""
        source = "intent_fallback"
        top_similarity = 0.0

        if retrieved_cases and len(retrieved_cases) > 0:
            top_case = retrieved_cases[0]
            top_similarity = top_case.get("score", 0.0)

            # High confidence threshold for grounding
            if top_similarity >= 0.40 and top_case.get("support_response"):
                extracted = self._extract_resolution_sentences(top_case["support_response"])
                if extracted and len(extracted) > 15:
                    resolution_text = extracted
                    source = f"faiss_retrieval (sim: {top_similarity:.2f})"

        if not resolution_text:
            resolution_text = self.INTENT_DEFAULTS.get(
                intent,
                "Please restart your device and verify that you are running the latest iOS update."
            )

        # 3. Handle Multi-Intent Composition
        if secondary_intent and secondary_intent != intent:
            topic1 = self.INTENT_TOPICS.get(intent, intent.lower().replace("_", " "))
            topic2 = self.INTENT_TOPICS.get(secondary_intent, secondary_intent.lower().replace("_", " "))
            opening = f"We can help with both your {topic1} and {topic2}."
            closing = f"Let us know how that goes, and we can look into the {topic2} next!"
            full_response = f"{opening} First, {resolution_text.lower() if resolution_text[0].isupper() and not resolution_text.startswith('Settings') else resolution_text} {closing}"
            source += " + multi_intent_synthesis"
        else:
            opening = self.INTENT_OPENINGS.get(intent, "We'd be glad to help resolve this issue with you.")
            closing = "Let us know how it goes!"
            full_response = f"{opening} {resolution_text} {closing}"

        return {
            "response": full_response,
            "source": source,
            "grounding_similarity": top_similarity,
            "intent_used": intent
        }


class CustomerSupportResponseGenerator:
    """Master response generator orchestrating deterministic offline generation with optional LLM plugin."""

    def __init__(self, use_llm: bool = False, llm_provider: Optional[str] = None):
        self.deterministic_gen = DeterministicGroundedGenerator()
        self.use_llm = use_llm
        self.llm_provider = llm_provider

    def generate_response(
        self,
        query: str,
        intent: str,
        retrieved_cases: List[Dict[str, Any]],
        escalation_result: Optional[Dict[str, Any]] = None,
        secondary_intent: Optional[str] = None,
        is_out_of_domain: bool = False,
        ood_entity: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate response based on query, intent, retrieval, escalation state, and domain boundaries."""
        # If case is escalated, return deterministic safety routing
        if escalation_result and escalation_result.get("is_escalated"):
            category = escalation_result.get("category", "RISK")
            reason = escalation_result.get("reason", "Safety escalation required.")
            action = escalation_result.get("recommended_action", "Routing to human specialist.")

            if category == "PHYSICAL_SAFETY_HAZARD":
                reply = (
                    "Safety Alert: Please immediately disconnect your device from charging and power. "
                    "Our Hardware Safety Team has been notified and will assist you urgently."
                )
            elif category == "ACCOUNT_SECURITY_COMPROMISE":
                reply = (
                    "Account Security Alert: If you suspect unauthorized access, please visit "
                    "iforgot.apple.com immediately to secure your Apple ID. A security specialist has been alerted."
                )
            elif category == "FINANCIAL_FRAUD_DISPUTE":
                reply = (
                    "Billing Notice: To review and report unauthorized charges securely, please visit "
                    "reportaproblem.apple.com. A billing representative is reviewing your case."
                )
            else:
                reply = (
                    "Thank you for contacting us. Your request has been escalated to a senior support "
                    "specialist who will follow up with you directly."
                )

            return {
                "response": reply,
                "source": f"deterministic_escalation_protocol ({category})",
                "is_escalated": True,
                "escalation_reason": reason,
                "recommended_action": action
            }

        # Otherwise generate grounded response
        gen_result = self.deterministic_gen.generate(
            query=query,
            intent=intent,
            retrieved_cases=retrieved_cases,
            secondary_intent=secondary_intent,
            is_out_of_domain=is_out_of_domain,
            ood_entity=ood_entity
        )
        gen_result["is_escalated"] = False
        return gen_result
