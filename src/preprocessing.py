"""Text preprocessing utilities for Twitter customer support queries."""

import html
import re
import unicodedata
from typing import Optional


def normalize_unicode(text: str) -> str:
    """Normalize unicode characters, smart quotes, and dashes to standard ASCII equivalents."""
    if not text:
        return ""
    
    # Replace common curly quotes and typographic symbols
    replacements = {
        "\u2018": "'",  # Left single quotation mark
        "\u2019": "'",  # Right single quotation mark / apostrophe
        "\u201c": '"',  # Left double quotation mark
        "\u201d": '"',  # Right double quotation mark
        "\u2014": " - ", # Em dash
        "\u2013": " - ", # En dash
        "\u2026": "...", # Horizontal ellipsis
        "\u00a0": " ",   # Non-breaking space
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
        
    # Unescape HTML entities
    text = html.unescape(text)
    
    # Normalize form (NFKD)
    normalized = unicodedata.normalize("NFKD", text)
    return normalized


def strip_twitter_artifacts(text: str, remove_mentions: bool = True, remove_urls: bool = True) -> str:
    """Remove Twitter mentions (@user) and URLs (http/https/t.co links)."""
    if not text:
        return ""
    
    # Remove URLs
    if remove_urls:
        text = re.sub(r"https?://\S+|www\.\S+", " ", text, flags=re.IGNORECASE)
    
    # Remove @mentions
    if remove_mentions:
        text = re.sub(r"@\w+", " ", text)
        
    return text


def clean_text(text: Optional[str]) -> str:
    """Clean and normalize a customer query for NLP pipeline processing.
    
    Performs:
    1. Unicode normalization and HTML unescaping.
    2. URL and mention stripping.
    3. Whitespace deduplication.
    """
    if text is None:
        return ""
    
    text = str(text)
    text = normalize_unicode(text)
    text = strip_twitter_artifacts(text, remove_mentions=True, remove_urls=True)
    
    # Remove excessive whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def is_valid_query(text: Optional[str], min_chars: int = 5) -> bool:
    """Check if query has enough substance to process."""
    if not text:
        return False
    cleaned = clean_text(text)
    return len(cleaned) >= min_chars
