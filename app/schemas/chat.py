"""
TalkBuddy — REST chat request/response Pydantic models.
"""

from typing import Optional

from pydantic import BaseModel, Field


class TextChatRequest(BaseModel):
    """Request for POST /api/v1/text/chat."""
    message: str = Field(..., min_length=1, description="User's text message")
    session_id: Optional[str] = Field(default=None, description="Session ID (auto-generated if not provided)")
    stream_audio: bool = Field(default=False, description="Whether to return audio in addition to text")
    voice: Optional[str] = Field(default=None, description="Kokoro voice override (e.g., af_heart)")


class TextChatResponse(BaseModel):
    """Response for POST /api/v1/text/chat (non-streaming)."""
    session_id: str
    user_message: str
    assistant_message: str
    audio_included: bool = Field(default=False, description="True if audio stream was requested")


class SessionCreateRequest(BaseModel):
    """Request for POST /api/v1/sessions."""
    session_id: Optional[str] = Field(default=None, description="Custom session ID (auto-generated if None)")
    system_prompt: Optional[str] = Field(default=None, description="Custom system prompt")


class SessionResponse(BaseModel):
    """Response for session endpoints."""
    session_id: str
    created_at: float
    last_activity: float
    message_count: int
    system_prompt: str


class SessionHistoryResponse(BaseModel):
    """Response for GET /api/v1/sessions/{id}."""
    session_id: str
    created_at: float
    last_activity: float
    system_prompt: str
    history: list[dict[str, str]]  # List of {role, content} messages
