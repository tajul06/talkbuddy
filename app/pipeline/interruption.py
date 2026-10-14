"""
TalkBuddy — CancellationToken and InterruptionCoordinator for barge-in.
Provides thread-safe cancellation to halt in-flight LLM/TTS generation instantly.
"""

import threading
import uuid
from typing import Optional


class CancellationToken:
    """
    Thread-safe cancellation token.
    Checked by LLM streaming loops and TTS workers to abort generation immediately.
    """

    def __init__(self, token_id: Optional[str] = None):
        self.token_id = token_id or str(uuid.uuid4())
        self._cancelled = threading.Event()

    def cancel(self) -> None:
        """Signal cancellation."""
        self._cancelled.set()

    def is_cancelled(self) -> bool:
        """Check if cancellation has been requested."""
        return self._cancelled.is_set()

    def raise_if_cancelled(self) -> None:
        """Raise an InterruptedError if cancelled."""
        if self._cancelled.is_set():
            raise InterruptedError(f"Generation {self.token_id} was cancelled")


class InterruptionCoordinator:
    """
    Manages active generation IDs and cancellation tokens for a session.
    Thread-safe coordinator that creates new tokens and cancels existing ones.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._current_token: CancellationToken = CancellationToken()
        self._generation_id: int = 0

    @property
    def generation_id(self) -> int:
        """Current generation sequence number."""
        with self._lock:
            return self._generation_id

    def get_current_token(self) -> CancellationToken:
        """Get the active cancellation token without modifying state."""
        with self._lock:
            return self._current_token

    def is_current(self, token: CancellationToken) -> bool:
        """Check if a token is still the active one (has not been superseded)."""
        with self._lock:
            return token.token_id == self._current_token.token_id and not token.is_cancelled()

    def interrupt(self) -> CancellationToken:
        """
        Interrupt the current generation.
        Cancels the active token and returns a fresh one for the next turn.
        """
        with self._lock:
            self._current_token.cancel()
            self._generation_id += 1
            self._current_token = CancellationToken()
            return self._current_token
