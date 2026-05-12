"""Sentence-boundary-aware text chunker for Turkish and English."""

import bisect
import re
from app.schemas.internal import Chunk


class Chunker:
    """Sentence-boundary-aware text chunker for Turkish and English.

    Uses token-to-character ratio estimation to find approximate chunk
    boundaries, then snaps to the nearest sentence boundary. Falls back
    to word boundaries if no sentence boundary is found.
    """

    # Turkish abbreviations that should NOT be treated as sentence endings
    ABBREVIATIONS: set[str] = {
        # Academic
        "dr", "prof", "doc", "yrd", "öğr", "gör", "arş",
        # English titles
        "sr", "jr", "mr", "mrs", "ms", "phd", "md",
        # General
        "vs", "vb", "vd", "v.b", "v.d", "v.s",
        # References
        "yy", "ay", "ör", "bkz", "bknz", "hk",
        # Page, article
        "sy", "sf", "s", "mad", "fık", "bend",
        # Contact
        "tel", "fax", "gsm", "no", "apt", "kat",
        # Company
        "a.ş", "ltd", "şti", "tic", "san", "paz",
        # Languages
        "ing", "alm", "fr", "it", "rus", "çin",
        # Address
        "st", "cd", "sk", "mah", "cad", "blv",
    }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chunk(
        self,
        text: str,
        tokenizer,
        max_tokens: int = 512,
        overlap: int = 0,
        filename: str = "",
    ) -> list[Chunk]:
        """Chunk *text* respecting sentence boundaries.

        Args:
            text: The input text to chunk.
            tokenizer: Any object with an ``encode(text, add_special_tokens=False)`` method.
            max_tokens: Maximum number of tokens per chunk.
            overlap: Approximate token overlap between consecutive chunks.
            filename: Source filename stored in each chunk's metadata.

        Returns:
            A list of :class:`Chunk` dataclass instances.
        """
        if not text or not text.strip():
            return []

        text = text.strip()
        total_tokens = self._count_tokens(text, tokenizer)

        # If the entire text fits in one chunk, return immediately
        if total_tokens <= max_tokens:
            return [
                Chunk(
                    text=text,
                    source_file=filename,
                    chunk_index=0,
                    token_count=total_tokens,
                    sentence_aware=True,
                )
            ]

        sentence_boundaries = self._find_sentence_boundaries(text)
        chunks: list[Chunk] = []
        start = 0
        chunk_index = 0
        text_len = len(text)

        # Safety counter to guarantee forward progress / prevent infinite loops
        max_iterations = text_len + 1000
        iterations = 0

        while start < text_len:
            iterations += 1
            if iterations > max_iterations:
                break  # safety net

            remaining_text = text[start:]
            remaining_tokens = self._count_tokens(remaining_text, tokenizer)

            # If the rest fits in one chunk, take it all
            if remaining_tokens <= max_tokens:
                chunks.append(
                    Chunk(
                        text=remaining_text.strip(),
                        source_file=filename,
                        chunk_index=chunk_index,
                        token_count=remaining_tokens,
                        sentence_aware=True,
                    )
                )
                break

            # Estimate character position using token-to-char ratio
            ratio = len(remaining_text) / max(remaining_tokens, 1)
            estimated_char_pos = int(max_tokens * ratio)
            # Clamp to text length
            estimated_char_pos = min(estimated_char_pos, len(remaining_text))

            # Absolute position in original text
            abs_pos = start + estimated_char_pos

            # Snap to nearest sentence boundary
            end_pos = self._snap_to_sentence_boundary(
                text, start, abs_pos, sentence_boundaries
            )

            # If no good sentence boundary, fall back to word boundary
            if end_pos is None or end_pos <= start:
                end_pos = self._snap_to_word_boundary(text, start, abs_pos)

            # Guarantee forward progress: move at least one character
            if end_pos <= start:
                end_pos = start + max(1, estimated_char_pos)
                end_pos = min(end_pos, text_len)

            chunk_text = text[start:end_pos].strip()

            # Verify token count; shrink if over limit
            chunk_tokens = self._count_tokens(chunk_text, tokenizer)
            shrink_attempts = 0
            while chunk_tokens > max_tokens and shrink_attempts < 20:
                shrink_attempts += 1
                # Reduce estimated position and re-snap
                estimated_char_pos = int(estimated_char_pos * 0.9)
                if estimated_char_pos < 1:
                    estimated_char_pos = 1
                abs_pos = start + estimated_char_pos
                end_pos = self._snap_to_sentence_boundary(
                    text, start, abs_pos, sentence_boundaries
                )
                if end_pos is None or end_pos <= start:
                    end_pos = self._snap_to_word_boundary(text, start, abs_pos)
                if end_pos <= start:
                    end_pos = start + max(1, estimated_char_pos)
                    end_pos = min(end_pos, text_len)
                chunk_text = text[start:end_pos].strip()
                chunk_tokens = self._count_tokens(chunk_text, tokenizer)

            # If still over after shrinking, do a hard word-boundary cut
            if chunk_tokens > max_tokens:
                chunk_text, chunk_tokens = self._hard_token_cut(
                    text, start, tokenizer, max_tokens
                )
                end_pos = start + len(chunk_text)
                # Adjust end_pos to account for leading whitespace that was stripped
                while end_pos < text_len and text[end_pos] == " ":
                    end_pos += 1

            if chunk_text:
                chunks.append(
                    Chunk(
                        text=chunk_text,
                        source_file=filename,
                        chunk_index=chunk_index,
                        token_count=chunk_tokens,
                        sentence_aware=True,
                    )
                )
                chunk_index += 1

            # Calculate next start position with overlap
            if overlap > 0 and end_pos < text_len:
                overlap_chars = self._estimate_overlap_chars(
                    text, end_pos, overlap, tokenizer
                )
                next_start = max(end_pos - overlap_chars, start + 1)
                start = next_start
            else:
                # Guarantee forward progress
                start = max(end_pos, start + 1)

        return chunks

    # ------------------------------------------------------------------
    # Sentence boundary detection
    # ------------------------------------------------------------------

    def _find_sentence_boundaries(self, text: str) -> list[int]:
        """Find reliable sentence-end positions in *text*.

        A sentence boundary is the position right after a sentence-ending
        punctuation mark (., !, ?) followed by whitespace, provided the
        period is not part of a known abbreviation or a decimal number.

        Returns:
            Sorted list of character indices where sentences end.
        """
        boundaries: list[int] = []
        i = 0
        text_len = len(text)

        while i < text_len:
            ch = text[i]
            if ch in ".!?":
                # Check if this is a real sentence boundary
                if ch == ".":
                    if self._is_abbreviation(text, i):
                        i += 1
                        continue
                    if self._is_decimal(text, i):
                        i += 1
                        continue

                # Look ahead: must be followed by whitespace or end-of-text
                next_pos = i + 1
                # Skip any trailing punctuation like quotes or parentheses
                while next_pos < text_len and text[next_pos] in "\"'\u201c\u201d\u2018\u2019)]":
                    next_pos += 1

                if next_pos >= text_len or text[next_pos] in " \t\n\r":
                    boundaries.append(next_pos)

            i += 1

        return boundaries

    # ------------------------------------------------------------------
    # Boundary snapping helpers
    # ------------------------------------------------------------------

    def _snap_to_sentence_boundary(
        self,
        text: str,
        start: int,
        target: int,
        boundaries: list[int],
    ) -> int | None:
        left = bisect.bisect_right(boundaries, start)
        right = bisect.bisect_right(boundaries, target)
        if right > left:
            return boundaries[right - 1]
        return None

    def _snap_to_word_boundary(self, text: str, start: int, target: int) -> int:
        """Snap *target* back to the nearest word boundary (whitespace).

        Never splits a word in the middle.
        """
        if target >= len(text):
            return len(text)

        # If we're already at a space, that's fine
        if text[target] == " ":
            return target

        # Search backwards for a space
        pos = target
        while pos > start and text[pos] != " ":
            pos -= 1

        # If we found a space, return position after it
        if pos > start:
            return pos

        # No space found between start and target — search forward
        pos = target
        while pos < len(text) and text[pos] != " ":
            pos += 1
        return pos

    # ------------------------------------------------------------------
    # Abbreviation / decimal helpers
    # ------------------------------------------------------------------

    def _is_abbreviation(self, text: str, dot_pos: int) -> bool:
        """Check whether the period at *dot_pos* belongs to an abbreviation."""
        # Extract the word before the dot
        word_start = dot_pos - 1
        while word_start >= 0 and (text[word_start].isalpha() or text[word_start] == "."):
            word_start -= 1
        word_start += 1

        word = text[word_start:dot_pos].lower().rstrip(".")
        # Also check with dots included (e.g. "a.ş", "v.b")
        word_with_dots = text[word_start:dot_pos].lower()

        return word in self.ABBREVIATIONS or word_with_dots in self.ABBREVIATIONS

    def _is_decimal(self, text: str, dot_pos: int) -> bool:
        """Check whether the period at *dot_pos* is a decimal point (e.g. 3.14)."""
        if dot_pos == 0 or dot_pos >= len(text) - 1:
            return False
        return text[dot_pos - 1].isdigit() and text[dot_pos + 1].isdigit()

    # ------------------------------------------------------------------
    # Token counting and hard-cut helpers
    # ------------------------------------------------------------------

    def _count_tokens(self, text: str, tokenizer) -> int:
        """Count tokens using the provided tokenizer."""
        try:
            tokens = tokenizer.encode(text, add_special_tokens=False)
            return len(tokens)
        except Exception:
            # Fallback: rough word-based estimate
            return max(1, len(text.split()))

    def _hard_token_cut(
        self, text: str, start: int, tokenizer, max_tokens: int
    ) -> tuple[str, int]:
        """Binary-search for the longest prefix from *start* that fits in *max_tokens*.

        Always cuts at a word boundary (space). Returns ``(chunk_text, token_count)``.
        """
        remaining = text[start:]
        words = remaining.split()
        if not words:
            return ("", 0)

        lo, hi = 1, len(words)
        best_text = words[0]
        best_tokens = self._count_tokens(best_text, tokenizer)

        while lo <= hi:
            mid = (lo + hi) // 2
            candidate = " ".join(words[:mid])
            t = self._count_tokens(candidate, tokenizer)
            if t <= max_tokens:
                best_text = candidate
                best_tokens = t
                lo = mid + 1
            else:
                hi = mid - 1

        return (best_text.strip(), best_tokens)

    def _estimate_overlap_chars(
        self, text: str, end_pos: int, overlap_tokens: int, tokenizer
    ) -> int:
        """Estimate how many characters correspond to *overlap_tokens* tokens
        near *end_pos* in *text*.
        """
        # Take a window of text ending at end_pos
        window_start = max(0, end_pos - 500)
        window = text[window_start:end_pos]
        if not window:
            return 0
        window_tokens = self._count_tokens(window, tokenizer)
        if window_tokens == 0:
            return 0
        ratio = len(window) / window_tokens
        return int(overlap_tokens * ratio)
