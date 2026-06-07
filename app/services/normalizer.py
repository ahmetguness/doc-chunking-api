"""Text normalization service."""

import re
from typing import Literal

# Pre-compiled regex patterns for performance
_WS_ZW_RE = re.compile(r"[\u200B-\u200D\uFEFF]")  # zero-width + BOM
_MULTI_SPACE_RE = re.compile(r"[ \t\f\r]+")
_NEWLINE_SPACE_RE = re.compile(r"[ \t\f\r]*\n[ \t\f\r]*")


class TextNormalizer:
    """Text normalization service.

    Handles:
    1. NBSP and narrow NBSP  regular space
    2. Zero-width characters and BOM removal
    3. Newline normalization (trim spaces around newlines)
    4. Collapse multiple spaces/tabs to single space
    5. Strip leading/trailing whitespace
    6. Case transformation (lowercase/uppercase/none)
    """

    def normalize(
        self, text: str, mode: Literal["none", "lowercase", "uppercase"] = "none"
    ) -> str:
        """Normalize text with whitespace cleanup and optional case transformation.

        Args:
            text: Input text to normalize
            mode: Case transformation mode - "none", "lowercase", or "uppercase"

        Returns:
            Normalized text string
        """
        if not isinstance(text, str):
            try:
                text = str(text)
            except Exception:
                return ""
        if not text:
            return ""

        # Replace NBSP and narrow NBSP with regular space
        text = text.replace("\u00A0", " ").replace("\u202F", " ")

        # Remove zero-width characters and BOM
        text = _WS_ZW_RE.sub("", text)

        # Normalize newlines: trim spaces around newlines
        text = _NEWLINE_SPACE_RE.sub("\n", text)

        # Collapse multiple spaces/tabs
        text = _MULTI_SPACE_RE.sub(" ", text)

        # Strip
        text = text.strip()

        # Case transformation
        if mode == "lowercase":
            text = text.lower()
        elif mode == "uppercase":
            text = text.upper()

        return text
