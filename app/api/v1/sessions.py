"""
TalkBuddy — Session CRUD endpoints.
"""

from typing import List

from fastapi import APIRouter, HTTPException

from app.core.logging import get_logger
from app.schemas.chat import (
    SessionCreateRequest,
    SessionHistoryResponse,
    SessionResponse,
)
from app.services.session_service import session_service

logger = get_logger()
router = APIRouter()


@router.post("/sessions", response_model=SessionResponse, summary="Create a new session")
async def create_session(request: SessionCreateRequest) -> SessionResponse:
    """
    Create a new conversation session.
    Optionally provide a custom session ID and system prompt.
    """
    session = session_service.create(
        session_id=request.session_id,
        system_prompt=request.system_prompt,
    )

    return SessionResponse(
        session_id=session.session_id,
        created_at=session.created_at,
        last_activity=session.last_activity,
        message_count=len(session.history),
        system_prompt=session.system_prompt,
    )


@router.get("/sessions", response_model=List[str], summary="List all active sessions")
async def list_sessions() -> List[str]:
    """
    Get a list of all active session IDs.
    Expired sessions are automatically cleaned up.
    """
    return session_service.list_sessions()


@router.get("/sessions/{session_id}", response_model=SessionHistoryResponse, summary="Get session details")
async def get_session(session_id: str) -> SessionHistoryResponse:
    """
    Get full conversation history for a session.
    """
    session = session_service.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

    return SessionHistoryResponse(
        session_id=session.session_id,
        created_at=session.created_at,
        last_activity=session.last_activity,
        system_prompt=session.system_prompt,
        history=[{"role": m.role, "content": m.content} for m in session.history],
    )


@router.delete("/sessions/{session_id}", summary="Delete a session")
async def delete_session(session_id: str) -> dict:
    """
    Delete a conversation session and its history.
    """
    deleted = session_service.delete(session_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

    return {"session_id": session_id, "status": "deleted"}
