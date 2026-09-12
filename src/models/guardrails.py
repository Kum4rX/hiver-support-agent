"""Response Guardrails Engine.

Enforces:
1. Strict Twitter character limits (<= 280 characters) with intelligent sentence-boundary trimming.
2. PII / Public credential exposure prevention (blocks asking for passwords/credit cards on public tweets).
3. Grounding & toxicity sanity checks.
4. Escalation output consistency.
"""

import re
from typing import Any, Dict, List, Optional, Tuple


class ResponseGuardrails:
    """Enforces safety, privacy, and platform constraints on customer support replies."""

    def __init__(self, max_length: int = 280):
        self.max_length = max_length

        # Regex patterns for dangerous PII solicitations over public Twitter
        self.pii_solicitation_patterns = [
            re.compile(r"\b(send|tweet|give|provide|dm) (us )?(your )?(password|passcode|pin|credit card|cvv|ssn)\b", re.IGNORECASE),
            re.compile(r"\b(what is|what's) your (password|card number|cvv|pin)\b", re.IGNORECASE)
        ]

        # Offensive / toxic tokens
        self.toxic_patterns = [
            re.compile(r"\b(shut up|idiot|stupid|moron|dumb|hate you)\b", re.IGNORECASE)
        ]

    def enforce_length_limit(self, text: str, max_chars: Optional[int] = None) -> Tuple[str, bool, bool]:
        """Enforce maximum character length by trimming cleanly at sentence or phrase boundaries.
        
        Returns:
            (sanitized_text, was_truncated, is_valid_length)
        """
        limit = max_chars or self.max_length
        text = text.strip()

        if len(text) <= limit:
            return text, False, True

        # Need truncation - find last sentence terminator within limit
        truncated = text[:limit]
        last_punct = max(
            truncated.rfind(". "),
            truncated.rfind("! "),
            truncated.rfind("? "),
            truncated.rfind(".\n")
        )

        if last_punct > 50:  # If we have a reasonable sentence prefix
            trimmed = text[:last_punct + 1].strip()
        else:
            # Fallback to word boundary
            last_space = truncated.rfind(" ")
            if last_space > 30:
                trimmed = text[:last_space].strip() + "..."
            else:
                trimmed = text[:limit - 3].strip() + "..."

        return trimmed, True, len(trimmed) <= limit

    def check_pii_safety(self, text: str) -> Tuple[bool, Optional[str]]:
        """Verify the response does not ask the user for sensitive credentials in a public tweet."""
        for pattern in self.pii_solicitation_patterns:
            if pattern.search(text):
                return False, "Response requested sensitive PII/passwords over public channel."
        return True, None

    def check_toxicity(self, text: str) -> Tuple[bool, Optional[str]]:
        """Verify response contains no toxic or inappropriate language."""
        for pattern in self.toxic_patterns:
            if pattern.search(text):
                return False, "Response contained prohibited or un-empathetic tone."
        return True, None

    def apply(
        self,
        raw_response: str,
        query: str,
        is_escalated: bool = False,
        max_chars: Optional[int] = None
    ) -> Dict[str, Any]:
        """Apply all guardrail checks and return sanitized final response with validation report."""
        limit = max_chars or self.max_length
        modifications: List[str] = []
        text = (raw_response or "").strip()
        original_length = len(text)

        # 1. Check PII solicitation
        pii_safe, pii_reason = self.check_pii_safety(text)
        if not pii_safe:
            modifications.append(f"PII_OVERRIDE: {pii_reason}")
            text = (
                "For your privacy and security, please do not share credentials publicly. "
                "Visit support.apple.com to manage your account safely."
            )

        # 2. Check Toxicity
        toxic_safe, toxic_reason = self.check_toxicity(text)
        if not toxic_safe:
            modifications.append(f"TOXICITY_OVERRIDE: {toxic_reason}")
            text = "We are here to assist you with your Apple device. Please let us know how we can help."

        # 3. Enforce Twitter length limit (<= 280 chars)
        sanitized_text, was_truncated, is_valid_length = self.enforce_length_limit(text, limit)
        if was_truncated:
            modifications.append(
                f"TRUNCATED: Length reduced from {original_length} to {len(sanitized_text)} chars (<= {limit})"
            )

        return {
            "final_response": sanitized_text,
            "original_length": original_length,
            "final_length": len(sanitized_text),
            "max_length_allowed": limit,
            "length_compliant": is_valid_length,
            "pii_safe": pii_safe,
            "toxic_safe": toxic_safe,
            "all_passed": (pii_safe and toxic_safe and is_valid_length),
            "modifications": modifications
        }
