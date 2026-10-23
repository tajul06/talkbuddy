"""
TalkBuddy — In-memory session and conversation history manager.
"""

import time
import uuid
from typing import Dict, List, Optional

from pydantic import BaseModel, Field

from app.core.config import settings
from app.core.constants import SESSION_MAX_HISTORY, SESSION_TTL_SECONDS
from app.core.logging import get_logger
from app.pipeline.interruption import InterruptionCoordinator

logger = get_logger()


class ChatMessage(BaseModel):
    """A single chat message in conversation history."""
    role: str  # "system" | "user" | "assistant"
    content: str
    timestamp: float = Field(default_factory=time.time)


class SessionState:
    """
    Holds all state for a single conversation session.
    Includes conversation history and interruption coordinator.
    """

    def __init__(self, session_id: str, system_prompt: Optional[str] = None):
        self.session_id = session_id
        self.created_at = time.time()
        self.last_activity = time.time()
        self.system_prompt = system_prompt or settings.llm.system_prompt
        self.history: List[ChatMessage] = [
            ChatMessage(role="system", content=self.system_prompt)
        ]
        self.coordinator = InterruptionCoordinator()
        self.is_speaking = False

    def add_message(self, role: str, content: str) -> None:
        """Add a message to conversation history and update activity timestamp."""
        self.history.append(ChatMessage(role=role, content=content))
        self.last_activity = time.time()

        # Prune old messages if exceeding max history (preserve system prompt at index 0)
        if len(self.history) > SESSION_MAX_HISTORY:
            self.history = [self.history[0]] + self.history[-(SESSION_MAX_HISTORY - 1):]

    def get_context(self) -> List[Dict[str, str]]:
        """Get conversation history as list of dicts for LLM."""
        return [{"role": m.role, "content": m.content} for m in self.history]

    def is_expired(self) -> bool:
        """Check if session has exceeded TTL idle timeout."""
        return time.time() - self.last_activity > SESSION_TTL_SECONDS


class SessionService:
    """
    Manages all conversation sessions in memory.
    Provides CRUD operations and automatic cleanup of expired sessions.
    """

    def __init__(self, ttl_seconds: int = SESSION_TTL_SECONDS):
        self._sessions: Dict[str, SessionState] = {}
        self.ttl_seconds = ttl_seconds

    def create(self, session_id: Optional[str] = None, system_prompt: Optional[str] = None) -> SessionState:
        """
        Create a new session.

        Args:
            session_id: Optional custom session ID (auto-generated if None)
            system_prompt: Optional custom system prompt

        Returns:
            The newly created SessionState
        """
        self._cleanup_expired()

        session_id = session_id or str(uuid.uuid4())
        session = SessionState(session_id, system_prompt)
        self._sessions[session_id] = session

        logger.debug(f"Created session {session_id}")
        return session

    def get_or_create(self, session_id: Optional[str] = None, system_prompt: Optional[str] = None) -> SessionState:
        """
        Get an existing session or create a new one if not found.

        Args:
            session_id: Session ID to look up (auto-generated if None)
            system_prompt: System prompt for new session (ignored if session exists)

        Returns:
            SessionState for the session
        """
        self._cleanup_expired()

        if session_id and session_id in self._sessions:
            return self._sessions[session_id]

        return self.create(session_id, system_prompt)

    def get(self, session_id: str) -> Optional[SessionState]:
        """
        Get a session by ID without creating.

        Args:
            session_id: Session ID to look up

        Returns:
            SessionState if found, None otherwise
        """
        self._cleanup_expired()
        return self._sessions.get(session_id)

    def delete(self, session_id: str) -> bool:
        """
        Delete a session by ID.

        Args:
            session_id: Session ID to delete

        Returns:
            True if session was deleted, False if not found
        """
        if session_id in self._sessions:
            del self._sessions[session_id]
            logger.debug(f"Deleted session {session_id}")
            return True
        return False

    def count(self) -> int:
        """Get the number of active sessions."""
        return len(self._sessions)

    def _cleanup_expired(self) -> None:
        """Remove sessions that have exceeded the TTL idle timeout."""
        expired = [
            sid for sid, session in self._sessions.items()
            if session.is_expired()
        ]
        for sid in expired:
            del self._sessions[sid]
            logger.debug(f"Expired session {sid} removed")

    def list_sessions(self) -> List[str]:
        """List all active session IDs."""
        self._cleanup_expired()
        return list(self._sessions.keys())


# Global session service singleton
session_service = SessionService()
