"""
TalkBuddy — Sentence-boundary splitter for LLM token stream.
Accumulates tokens and flushes complete sentences to TTS.
First sentence flushes early for minimum time-to-first-audio.
"""

import re
from typing import List

from app.core.constants import SENTENCE_MIN_FIRST_CHARS, SENTENCE_MIN_NORMAL_CHARS


class SentenceChunker:
    """
    Buffers streaming LLM tokens and yields sentence-level chunks
    optimized for TTS prosody and ultra-low latency.
    """

    ABBREVIATIONS = frozenset({
        "mr.", "mrs.", "ms.", "dr.", "prof.", "sr.", "jr.", "vs.",
        "e.g.", "i.e.", "etc.", "approx.", "u.s.", "u.k.", "st.",
        "no.", "vol.", "dept.", "gen.", "sgt.", "lt.", "col.",
    })

    # Split on sentence-ending punctuation followed by whitespace or end of string
    SPLIT_REGEX = re.compile(r"([.?!;:]\s+)")

    def __init__(
        self,
        min_first_chunk_chars: int = SENTENCE_MIN_FIRST_CHARS,
        min_normal_chunk_chars: int = SENTENCE_MIN_NORMAL_CHARS,
    ):
        self.min_first_chunk_chars = min_first_chunk_chars
        self.min_normal_chunk_chars = min_normal_chunk_chars
        self.reset()

    def reset(self) -> None:
        """Reset state for a new generation turn."""
        self._buffer = ""
        self._is_first_chunk = True

    def add_token(self, token: str) -> List[str]:
        """
        Append a token and return any completed sentence chunks.

        Args:
            token: A single streamed LLM token

        Returns:
            List of completed sentence strings (may be empty)
        """
        self._buffer += token
        chunks = []

        while True:
            found_chunk = False
            for match in self.SPLIT_REGEX.finditer(self._buffer):
                end_pos = match.end()
                candidate = self._buffer[:end_pos].strip()

                if not candidate:
                    continue

                # Guard against false splits on abbreviations (e.g. "Dr.", "e.g.")
                words = candidate.split()
                if words:
                    last_word = words[-1].lower()
                    last_word_dot = last_word if last_word.endswith(".") else last_word + "."
                    if last_word in self.ABBREVIATIONS or last_word_dot in self.ABBREVIATIONS:
                        # Skip abbreviation match and continue searching buffer
                        continue

                # Check length threshold
                min_len = self.min_first_chunk_chars if self._is_first_chunk else self.min_normal_chunk_chars
                if len(candidate) >= min_len or ("\n" in match.group(1)):
                    chunks.append(candidate)
                    self._buffer = self._buffer[end_pos:]
                    self._is_first_chunk = False
                    found_chunk = True
                    break
                else:
                    # Candidate too short; continue to next match which covers more text
                    continue

            if not found_chunk:
                break

        return chunks

    def flush(self) -> List[str]:
        """
        Flush any remaining text in the buffer (called on LLM end-of-stream).

        Returns:
            List with the remaining text, or empty if buffer is empty
        """
        remaining = self._buffer.strip()
        self.reset()
        if remaining:
            return [remaining]
        return []
