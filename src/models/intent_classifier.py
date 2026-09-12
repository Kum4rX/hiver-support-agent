"""Intent classification module for AppleSupport queries.

Includes:
1. Canonical 11-intent taxonomy definitions.
2. Out-of-Domain (non-Apple) platform and device detector.
3. Multi-intent detector for composite customer issues.
4. Rule-based / Keyword baseline classifier (`KeywordRuleIntentClassifier`).
5. TF-IDF + Logistic Regression statistical classifier (`TfidfLogisticIntentClassifier`).
6. Hybrid intent classifier (`HybridIntentClassifier`) combining ML with deterministic rule fallbacks.
"""

import os
import re
from typing import Any, Dict, List, Optional, Tuple
import joblib
import numpy as np

# Non-Apple ecosystem keywords for deterministic out-of-domain guard
NON_APPLE_ENTITIES = [
    r"\bwindows( 1[01]| 8| 7| xp| vista)?\b",
    r"\bdell\b", r"\bhp\b", r"\blenovo\b", r"\basus\b", r"\bacer\b",
    r"\bandroid\b", r"\bsamsung( galaxy)?\b", r"\bgoogle pixel\b",
    r"\bxbox( one| series [xs])?\b", r"\bplaystation\b|\bps[45]\b",
    r"\bnintendo( switch)?\b", r"\blinux\b", r"\bubuntu\b",
    r"\bblue screen of death\b|\bbsod\b"
]

NON_APPLE_COMPILED = [re.compile(p, re.IGNORECASE) for p in NON_APPLE_ENTITIES]

# Canonical 11-intent taxonomy for Apple Support
INTENT_TAXONOMY: Dict[str, Dict[str, Any]] = {
    "BATTERY_POWER": {
        "description": "Battery drain, charging problems, overheating while charging, power cycles",
        "keywords": [
            "battery", "charging", "charge", "charger", "drain", "draining", "power", "dies",
            "dying", "percentage", "rapidly draining", "dead battery", "won't charge", "overheating"
        ],
        "patterns": [
            r"\bbatter(y|ies)\b", r"\bcharg(e|ing|er|ed)\b", r"\bdrain(ing|s|ed)?\b",
            r"\bpower(ing)? (off|down)\b", r"\b(dies|dying) fast\b", r"\bpercent(age)?\b"
        ]
    },
    "CONNECTIVITY": {
        "description": "Wi-Fi, Bluetooth, cellular, LTE, 5G, signal drops, hotspot",
        "keywords": [
            "wifi", "wi-fi", "bluetooth", "internet", "network", "connection",
            "cellular", "lte", "signal", "hotspot", "airdrop", "no service", "carrier", "router"
        ],
        "patterns": [
            r"\bwi-?fi\b", r"\bbluetooth\b", r"\bcellular\b", r"\bno service\b",
            r"\blte\b", r"\bhotspot\b", r"\bairdrop\b", r"\bnetwork connection\b", r"\brouter\b"
        ]
    },
    "CALLS_COMMUNICATION": {
        "description": "Phone calls, voicemail, FaceTime, iMessage, SMS text messaging",
        "keywords": [
            "call", "calling", "phone call", "facetime", "imessage", "sms",
            "text message", "texts", "voicemail", "messages"
        ],
        "patterns": [
            r"\b(phone )?calls?\b", r"\bfacetime\b", r"\bimessage\b", r"\bsms\b",
            r"\btext(s|ing| message)?\b", r"\bvoicemail\b"
        ]
    },
    "DEVICE_PERFORMANCE": {
        "description": "Freezing, lagging, slow response, iOS update bugs, reboots, crashes",
        "keywords": [
            "slow", "freez", "freeze", "freezing", "lag", "lagging", "crash", "crashing",
            "restart", "reboot", "update", "ios 11", "ios 10", "ios update", "glitch", "stuck"
        ],
        "patterns": [
            r"\b(slow|lag|laggy|lagging)\b", r"\bfreez(e|ing|es|ed)\b", r"\bcrash(es|ing|ed)?\b",
            r"\b(randomly )?(restart|reboot)(s|ing|ed)?\b", r"\bios \d+(\.\d+)*\b", r"\bstuck on apple logo\b"
        ]
    },
    "KEYBOARD_INPUT": {
        "description": "Keyboard typing, autocorrect, predictive text, letter replacement bugs, dictation",
        "keywords": [
            "keyboard", "autocorrect", "auto-correct", "auto correct", "typing", "type",
            "key", "emoji", "predictive", "letter i", "i glitch", "dictation"
        ],
        "patterns": [
            r"\bkeyboard\b", r"\bauto-?correct\b", r"\btyp(e|ing|o)\b",
            r"\bletter ['\"]?i['\"]?\b", r"\bemoji(s)?\b", r"\bpredictive text\b"
        ]
    },
    "APPS_MEDIA": {
        "description": "App Store app crashes, third-party apps, Apple Music, Photos, Podcasts, Camera Roll",
        "keywords": [
            "app", "application", "apps", "itunes", "music", "video", "photo", "photos",
            "youtube", "instagram", "facebook", "spotify", "whatsapp", "download app", "podcast"
        ],
        "patterns": [
            r"\bapp(s|lications?)?\b", r"\bitunes\b", r"\b(apple )?music\b",
            r"\b(photo|video)s?\b", r"\b(instagram|youtube|facebook|spotify|whatsapp)\b"
        ]
    },
    "DISPLAY_AUDIO_CAMERA": {
        "description": "Screen, display brightness, cracked screen, camera blurry, speaker, audio, microphone",
        "keywords": [
            "screen", "display", "brightness", "camera", "speaker", "sound",
            "audio", "microphone", "volume", "black screen", "touch screen", "unresponsive screen"
        ],
        "patterns": [
            r"\b(touch)?screen\b", r"\bdisplay\b", r"\bbrightness\b", r"\bcamera\b",
            r"\b(speaker|sound|audio|volume|mic|microphone)\b", r"\bblack screen\b"
        ]
    },
    "ACCOUNT_ICLOUD": {
        "description": "Apple ID, iCloud backup, iCloud storage, login, password reset, 2FA",
        "keywords": [
            "icloud", "apple id", "appleid", "password", "passcode", "account",
            "login", "sign in", "signin", "storage", "verification code", "2fa"
        ],
        "patterns": [
            r"\b(apple ?id|appleid)\b", r"\bicloud\b", r"\bpass(word|code)\b",
            r"\b(sign|log) ?in\b", r"\baccount locked\b", r"\bverification code\b"
        ]
    },
    "PURCHASE_PAYMENT": {
        "description": "In-app purchases, billing, subscriptions, refunds, Apple Pay, charges",
        "keywords": [
            "purchase", "buy", "payment", "paid", "refund", "charge", "charged",
            "billing", "credit card", "subscription", "apple pay", "receipt", "overcharged"
        ],
        "patterns": [
            r"\b(purchas|buy|bought|payment|bill|charge|refund)(e|ing|ed|s)?\b",
            r"\bapple ?pay\b", r"\bsubscription(s)?\b", r"\bcredit card\b"
        ]
    },
    "HOW_TO_OTHER": {
        "description": "General settings, how-to inquiries, configuration, navigation",
        "keywords": [
            "how do i", "how can i", "where do i", "can i", "how to",
            "setting", "settings", "how does", "feature", "configure", "enable", "disable"
        ],
        "patterns": [
            r"\bhow (do|can|to|would) i\b", r"\bwhere (do|can) i\b",
            r"\bsetting(s)?\b", r"\bhow to\b", r"\bcan i\b"
        ]
    },
    "SECURITY": {
        "description": "Hacked account, phishing, scam, stolen device, unauthorized access, suspicious activity",
        "keywords": [
            "hack", "hacked", "phishing", "scam", "fraud", "stolen",
            "security", "suspicious", "privacy", "unauthorized", "compromised", "breach"
        ],
        "patterns": [
            r"\bhack(ed|er|ing)?\b", r"\bphish(ing)?\b", r"\bscam(med|mer|s)?\b",
            r"\bstolen\b", r"\b(unauthorized|suspicious)\b"
        ]
    }
}

ALL_INTENTS: List[str] = list(INTENT_TAXONOMY.keys())


def detect_out_of_domain(text: str) -> Tuple[bool, Optional[str]]:
    """Check if query explicitly pertains to non-Apple hardware/platforms."""
    text_lower = (text or "").lower()
    for pattern in NON_APPLE_COMPILED:
        match = pattern.search(text_lower)
        if match:
            return True, match.group(0)
    return False, None


class KeywordRuleIntentClassifier:
    """Deterministic, rule-based baseline intent classifier using keyword and regex matching."""

    def __init__(self, taxonomy: Optional[Dict[str, Dict[str, Any]]] = None):
        self.taxonomy = taxonomy or INTENT_TAXONOMY
        self._compiled_patterns = {
            intent: [re.compile(p, re.IGNORECASE) for p in data["patterns"]]
            for intent, data in self.taxonomy.items()
        }

    def predict(self, text: str) -> str:
        """Predict the single best intent for a given text."""
        result = self.predict_with_scores(text)
        return result["intent"]

    def predict_with_scores(self, text: str) -> Dict[str, Any]:
        """Compute matching scores across all intents and return top intents with confidence."""
        text_lower = (text or "").lower().strip()
        scores: Dict[str, float] = {intent: 0.0 for intent in self.taxonomy}

        for intent, data in self.taxonomy.items():
            # Keyword exact count
            for kw in data["keywords"]:
                if kw in text_lower:
                    scores[intent] += 1.5

            # Regex pattern match
            for pat in self._compiled_patterns[intent]:
                if pat.search(text_lower):
                    scores[intent] += 2.0

        # Disambiguation heuristics
        if scores["SECURITY"] > 0:
            scores["SECURITY"] += 2.0  # Safety priority

        sorted_intents = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        primary_intent, primary_score = sorted_intents[0]
        secondary_intent, secondary_score = sorted_intents[1]

        # Check for meaningful multi-intent (>0 and within significant range)
        has_multi_intent = (primary_score >= 2.0 and secondary_score >= 1.5 and secondary_intent != primary_intent)

        # Calculate pseudo-confidence
        total_score = sum(scores.values())
        if total_score > 0:
            confidence = min(0.95, primary_score / total_score)
        else:
            primary_intent = "HOW_TO_OTHER"
            confidence = 0.30

        return {
            "intent": primary_intent,
            "secondary_intent": secondary_intent if has_multi_intent else None,
            "has_multi_intent": has_multi_intent,
            "confidence": round(float(confidence), 4),
            "scores": scores
        }


class TfidfLogisticIntentClassifier:
    """Statistical intent classifier using sublinear TF-IDF and Logistic Regression."""

    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path
        self.pipeline = None
        self.classes_ = None
        if model_path and os.path.exists(model_path):
            self.load(model_path)

    def load(self, model_path: str) -> None:
        """Load trained pipeline from disk."""
        self.pipeline = joblib.load(model_path)
        if hasattr(self.pipeline, "classes_"):
            self.classes_ = list(self.pipeline.classes_)
        elif hasattr(self.pipeline.named_steps.get("classifier", None), "classes_"):
            self.classes_ = list(self.pipeline.named_steps["classifier"].classes_)
        self.model_path = model_path

    def save(self, model_path: str) -> None:
        """Save trained pipeline to disk."""
        if self.pipeline is None:
            raise ValueError("Cannot save an unfitted model pipeline.")
        os.makedirs(os.path.dirname(os.path.abspath(model_path)), exist_ok=True)
        joblib.dump(self.pipeline, model_path)
        self.model_path = model_path

    def is_trained(self) -> bool:
        return self.pipeline is not None

    def predict(self, text: str) -> str:
        if not self.is_trained():
            raise RuntimeError("Model is not trained or loaded.")
        return str(self.pipeline.predict([text])[0])

    def predict_proba(self, text: str) -> Dict[str, float]:
        if not self.is_trained():
            raise RuntimeError("Model is not trained or loaded.")
        probs = self.pipeline.predict_proba([text])[0]
        classes = self.classes_ or self.pipeline.classes_
        return {str(cls): float(prob) for cls, prob in zip(classes, probs)}

    def predict_with_confidence(self, text: str) -> Dict[str, Any]:
        if not self.is_trained():
            raise RuntimeError("Model is not trained or loaded.")
        probs = self.predict_proba(text)
        sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
        best_intent, confidence = sorted_probs[0]
        second_intent = sorted_probs[1][0] if len(sorted_probs) > 1 else None
        return {
            "intent": best_intent,
            "secondary_intent": second_intent if sorted_probs[1][1] >= 0.25 else None,
            "confidence": round(confidence, 4),
            "probabilities": probs
        }


class HybridIntentClassifier:
    """Robust production intent classifier with Out-of-Domain and Multi-Intent detection."""

    def __init__(self, model_path: Optional[str] = "data/intent_baseline.joblib"):
        self.rule_classifier = KeywordRuleIntentClassifier()
        self.ml_classifier = TfidfLogisticIntentClassifier()
        self.model_path = model_path

        if model_path and os.path.exists(model_path):
            try:
                self.ml_classifier.load(model_path)
            except Exception:
                self.ml_classifier.pipeline = None

    def classify(self, text: str) -> Dict[str, Any]:
        """Classify customer query intent with confidence, out-of-domain guard, and multi-intent detection."""
        # 1. Deterministic Out-of-Domain Guard
        is_ood, ood_entity = detect_out_of_domain(text)
        if is_ood:
            return {
                "intent": "OUT_OF_DOMAIN",
                "secondary_intent": None,
                "has_multi_intent": False,
                "is_out_of_domain": True,
                "ood_entity": ood_entity,
                "confidence": 0.99,
                "method": "out_of_domain_guard",
                "is_provisional": False
            }

        rule_res = self.rule_classifier.predict_with_scores(text)
        
        # Check if ML model is provisional (< 5 classes)
        is_ml_provisional = (
            not self.ml_classifier.is_trained() or
            (self.ml_classifier.classes_ is not None and len(self.ml_classifier.classes_) < 5)
        )

        if is_ml_provisional:
            return {
                "intent": rule_res["intent"],
                "secondary_intent": rule_res["secondary_intent"],
                "has_multi_intent": rule_res["has_multi_intent"],
                "is_out_of_domain": False,
                "ood_entity": None,
                "confidence": rule_res["confidence"],
                "method": "rule_taxonomy_engine",
                "is_provisional": True,
                "scores": rule_res["scores"]
            }

        try:
            ml_res = self.ml_classifier.predict_with_confidence(text)
        except Exception:
            return {
                "intent": rule_res["intent"],
                "secondary_intent": rule_res["secondary_intent"],
                "has_multi_intent": rule_res["has_multi_intent"],
                "is_out_of_domain": False,
                "ood_entity": None,
                "confidence": rule_res["confidence"],
                "method": "rule_fallback",
                "is_provisional": True,
                "scores": rule_res["scores"]
            }

        # Multi-intent combination
        secondary = rule_res["secondary_intent"] or ml_res.get("secondary_intent")
        has_multi = rule_res["has_multi_intent"] or (secondary is not None and secondary != ml_res["intent"])

        # Security override
        if rule_res["intent"] == "SECURITY" and rule_res["scores"].get("SECURITY", 0) >= 3.0:
            return {
                "intent": "SECURITY",
                "secondary_intent": secondary,
                "has_multi_intent": has_multi,
                "is_out_of_domain": False,
                "ood_entity": None,
                "confidence": max(0.90, rule_res["confidence"]),
                "method": "rule_security_override",
                "is_provisional": False,
                "ml_intent": ml_res["intent"]
            }

        # Consensus
        if ml_res["intent"] == rule_res["intent"]:
            blended_conf = min(0.99, (ml_res["confidence"] * 0.6) + (rule_res["confidence"] * 0.4) + 0.1)
            return {
                "intent": ml_res["intent"],
                "secondary_intent": secondary,
                "has_multi_intent": has_multi,
                "is_out_of_domain": False,
                "ood_entity": None,
                "confidence": round(blended_conf, 4),
                "method": "hybrid_consensus",
                "is_provisional": False
            }

        # Rule strong match
        if rule_res["confidence"] >= 0.50:
            return {
                "intent": rule_res["intent"],
                "secondary_intent": secondary,
                "has_multi_intent": has_multi,
                "is_out_of_domain": False,
                "ood_entity": None,
                "confidence": rule_res["confidence"],
                "method": "rule_strong_match",
                "is_provisional": False
            }

        return {
            "intent": rule_res["intent"],
            "secondary_intent": secondary,
            "has_multi_intent": has_multi,
            "is_out_of_domain": False,
            "ood_entity": None,
            "confidence": rule_res["confidence"],
            "method": "rule_best_guess",
            "is_provisional": True
        }
