"""TalkBuddy — Sentence chunker unit tests."""

import pytest
from app.pipeline.chunker import SentenceChunker


class TestSentenceChunker:
    """Tests for the SentenceChunker token accumulator."""

    def setup_method(self):
        """Fresh chunker for each test."""
        self.chunker = SentenceChunker(
            min_first_chunk_chars=15,
            min_normal_chunk_chars=35,
        )

    # --- Basic sentence splitting ---

    def test_single_sentence_with_period(self):
        """A complete sentence ending with period + space flushes."""
        tokens = ["Hello", ", ", "how ", "are ", "you ", "today? "]
        results = []
        for t in tokens:
            results.extend(self.chunker.add_token(t))
        assert len(results) == 1
        assert results[0] == "Hello, how are you today?"

    def test_no_flush_without_punctuation(self):
        """Tokens without sentence-ending punctuation stay buffered."""
        for t in ["Hello ", "world ", "this ", "is "]:
            assert self.chunker.add_token(t) == []

    def test_flush_on_eos(self):
        """flush() returns remaining buffer content."""
        for t in ["Hello ", "world"]:
            self.chunker.add_token(t)
        result = self.chunker.flush()
        assert result == ["Hello world"]

    def test_flush_empty_buffer(self):
        """flush() on empty buffer returns empty list."""
        assert self.chunker.flush() == []

    # --- First chunk vs normal chunk thresholds ---

    def test_short_first_sentence_flushes_early(self):
        """First sentence needs only 15 chars to flush (low TTFA)."""
        # "How are you? " = 12 chars — too short
        result = self.chunker.add_token("How are you? ")
        assert result == []

        # "It's a nice day. " = 16 chars from start — should flush
        self.chunker.reset()
        tokens = ["It's ", "a ", "nice ", "day. "]
        results = []
        for t in tokens:
            results.extend(self.chunker.add_token(t))
        assert len(results) == 1

    def test_second_sentence_needs_more_chars(self):
        """After first flush, subsequent sentences need 35+ chars."""
        # First sentence — long enough
        for t in ["This is the first sentence. "]:
            self.chunker.add_token(t)

        # Short second sentence — should NOT flush yet
        results = self.chunker.add_token("Too short. ")
        assert results == []

    # --- Abbreviation guard ---

    def test_abbreviation_not_split(self):
        """Abbreviations like 'Dr.' should not trigger a sentence split."""
        tokens = ["Dr. ", "Smith ", "is ", "here. "]
        results = []
        for t in tokens:
            results.extend(self.chunker.add_token(t))
        # Should be one sentence, not split at "Dr. "
        assert len(results) == 1
        assert "Dr." in results[0]

    def test_eg_abbreviation(self):
        """'e.g.' should not trigger a split."""
        tokens = ["Use ", "tools ", "e.g. ", "Python ", "and ", "Rust. "]
        results = []
        for t in tokens:
            results.extend(self.chunker.add_token(t))
        assert len(results) == 1
        assert "e.g." in results[0]

    # --- Multiple sentences ---

    def test_two_sentences_stream(self):
        """Two sentences in a stream both flush."""
        # Second sentence is 45 chars, above min_normal_chunk_chars=35
        text = "Hello, I am TalkBuddy. How can I help you with your questions today? "
        results = []
        # Simulate token-by-token streaming
        for char in text:
            results.extend(self.chunker.add_token(char))
        assert len(results) == 2

    def test_mixed_punctuation(self):
        """Question marks and exclamation marks also split."""
        tokens = ["What a wonderful day! ", "Don't you think so? "]
        results = []
        for t in tokens:
            results.extend(self.chunker.add_token(t))
        assert len(results) >= 1

    # --- Reset ---

    def test_reset_clears_state(self):
        """reset() clears buffer and resets first-chunk flag."""
        self.chunker.add_token("some buffered text")
        self.chunker.reset()
        assert self.chunker.flush() == []

    # --- Edge cases ---

    def test_empty_token(self):
        """Empty string tokens do not break anything."""
        assert self.chunker.add_token("") == []

    def test_newline_in_text(self):
        """Newlines in the buffer are handled."""
        tokens = ["Hello world.\n", "Next line. "]
        results = []
        for t in tokens:
            results.extend(self.chunker.add_token(t))
        remaining = self.chunker.flush()
        total = results + remaining
        assert len(total) >= 1
