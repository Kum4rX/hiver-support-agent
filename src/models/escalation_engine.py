"""Deterministic Risk and Escalation Engine.

Evaluates customer queries against strict, prioritized safety and policy rules:
1. Physical safety & hardware hazards (swelling battery, smoke, sparks, fire, electrical shock).
2. Account security & compromise (hacked Apple ID, unauthorized access, stolen device lockout, phishing).
3. Financial fraud & billing disputes (unauthorized charges, bank fraud, disputed transactions).
4. Legal, regulatory, or severe harassment issues.
5. Chronic unresolved service failure (repeated failed attempts, supervisor demand).
"""

import re
from typing import Any, Dict, List, Optional


class EscalationEngine:
    """Deterministic, rule-based escalation engine for customer support queries."""

    # Escalation Rule Definitions (Ordered by Severity Priority)
    RULES: List[Dict[str, Any]] = [
        {
            "category": "PHYSICAL_SAFETY_HAZARD",
            "severity": "CRITICAL",
            "patterns": [
                r"\b(smoke|smoking|spark|sparks|sparking|fire|exploded|exploding|explosion)\b",
                r"\b(swollen|swelling|expanded|expanding|bulging)\b.*\bbatter(y|ies)\b",
                r"\bbatter(y|ies)\b.*\b(swollen|swelling|expanded|expanding|bulging)\b",
                r"\b(melted|melting|burnt|burning|scorch(ed)?)\b.*\b(cable|charger|phone|port|wire|battery)\b",
                r"\b(electric(al)? shock|burned my (hand|finger|face)|caught fire)\b",
            ],
            "reason": "Physical safety hazard or hardware danger detected (smoke/fire/swelling/shock).",
            "recommended_action": "PRIORITY_SAFETY_ESCALATION: Instruct user to safely disconnect power and route immediately to Senior Hardware Safety Team."
        },
        {
            "category": "ACCOUNT_SECURITY_COMPROMISE",
            "severity": "HIGH",
            "patterns": [
                r"\b(hacked|compromised|breached)\b.*\b(account|apple ?id|icloud|iphone|ipad|mac)\b",
                r"\b(someone|stranger)\b.*\b(changed my password|logged into my|stole my apple ?id)\b",
                r"\b(phishing|scam|ransomware|blackmail)\b.*\b(message|link|email|call|text)\b",
                r"\b(stolen|lost)\b.*\b(iphone|ipad|macbook|phone)\b.*\b(locked out|erased|tracked)\b",
                r"\bunauthorized\b.*\b(password reset|apple ?id login|2fa code|access)\b"
            ],
            "reason": "Account security breach, compromised credentials, or active theft detected.",
            "recommended_action": "SECURITY_ESCALATION: Provide official recovery link (iforgot.apple.com) and route to Account Security Specialist."
        },
        {
            "category": "FINANCIAL_FRAUD_DISPUTE",
            "severity": "HIGH",
            "patterns": [
                r"\b(fraudulent|unauthorized|stolen)\b.*\b(charge|transaction|credit card|apple ?pay|purchase|billing)\b",
                r"\b(bank dispute|chargeback|stole my money|unauthorized billing)\b",
                r"\bcharged\b.*\b(without (my )?permission|i didn't (buy|authorize|make))\b"
            ],
            "reason": "Unauthorized financial charges or payment fraud detected.",
            "recommended_action": "BILLING_FRAUD_ESCALATION: Direct to reportaproblem.apple.com and route to Apple Billing & Payments Team."
        },
        {
            "category": "LEGAL_OR_REGULATORY",
            "severity": "HIGH",
            "patterns": [
                r"\b(lawyer|attorney|lawsuit|suing apple|sue you|legal action)\b",
                r"\b(police report|file a claim|ftc complaint|consumer protection|court)\b"
            ],
            "reason": "Legal action, law enforcement, or formal regulatory threat detected.",
            "recommended_action": "LEGAL_ESCALATION: Route case immediately to Executive Customer Relations & Legal Support."
        },
        {
            "category": "CHRONIC_UNRESOLVED_SERVICE",
            "severity": "MEDIUM",
            "patterns": [
                r"\b(speak with|talk to|transfer to)\b.*\b(supervisor|manager|human agent|senior agent)\b",
                r"\b(been to (the )?apple store \d+ times|called \d+ times)\b",
                r"\b(second|third|fourth|5th|fifth|\d+th)\b.*\breplacement\b.*\b(broken|still|faulty|issue)\b",
                r"\bunacceptable service\b.*\b(after weeks|for months|days)\b"
            ],
            "reason": "Customer demands supervisor or reports chronic unresolved technical failure.",
            "recommended_action": "SUPERVISOR_ESCALATION: Route to Senior Tier-2 Technical Advisor for expedited resolution."
        }
    ]

    def __init__(self):
        # Precompile regular expressions for microsecond latency
        self._compiled_rules = []
        for rule in self.RULES:
            compiled_patterns = [re.compile(p, re.IGNORECASE) for p in rule["patterns"]]
            self._compiled_rules.append({
                "category": rule["category"],
                "severity": rule["severity"],
                "compiled_patterns": compiled_patterns,
                "reason": rule["reason"],
                "recommended_action": rule["recommended_action"]
            })

    def evaluate(self, text: str) -> Dict[str, Any]:
        """Evaluate text for safety and risk escalation triggers."""
        text_clean = (text or "").strip()
        
        for rule in self._compiled_rules:
            for pattern in rule["compiled_patterns"]:
                match = pattern.search(text_clean)
                if match:
                    return {
                        "is_escalated": True,
                        "category": rule["category"],
                        "severity": rule["severity"],
                        "reason": rule["reason"],
                        "matched_pattern": match.group(0),
                        "recommended_action": rule["recommended_action"]
                    }

        # Safe for automated resolution
        return {
            "is_escalated": False,
            "category": "NONE",
            "severity": "LOW",
            "reason": "No risk triggers detected. Safe for automated AI resolution.",
            "matched_pattern": None,
            "recommended_action": "PROCEED_AUTOMATED_RESOLUTION"
        }
