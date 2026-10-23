"""
TalkBuddy — WebSocket JSON control message schemas.

Binary frames carry raw PCM audio:
  Client → Server: 16kHz 16-bit mono PCM
  Server → Client: 24kHz 16-bit mono PCM

JSON frames carry control and event messages defined here.
"""

from typing import Any, Dict, Literal, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Client → Server messages
# ---------------------------------------------------------------------------

class WsInitMessage(BaseModel):
    """Sent by client at connection start to configure the session."""
    type: Literal["init"] = "init"
    session_id: Optional[str] = Field(default=None, description="Existing session to resume")
    voice: Optional[str] = Field(default=None, description="Kokoro voice override")
    system_prompt: Optional[str] = Field(default=None, description="Custom system prompt")


class WsInterruptMessage(BaseModel):
    """Sent by client to request immediate barge-in interruption."""
    type: Literal["interrupt"] = "interrupt"


class WsPingMessage(BaseModel):
    """Client keepalive ping."""
    type: Literal["ping"] = "ping"


# ---------------------------------------------------------------------------
# Server → Client messages
# ---------------------------------------------------------------------------

class WsPongMessage(BaseModel):
    """Server keepalive pong."""
    type: Literal["pong"] = "pong"


class WsSessionMessage(BaseModel):
    """Sent after init — confirms session ID and agent is ready."""
    type: Literal["session"] = "session"
    session_id: str
    voice: str


class WsVadStateMessage(BaseModel):
    """VAD state transition notification."""
    type: Literal["vad_state"] = "vad_state"
    state: Literal["speech_start", "speech_end", "silence"]
    probability: float = Field(ge=0.0, le=1.0)


class WsTranscriptMessage(BaseModel):
    """STT transcription result for the user's utterance."""
    type: Literal["transcript"] = "transcript"
    text: str


class WsAgentStateMessage(BaseModel):
    """Agent pipeline state change."""
    type: Literal["agent_state"] = "agent_state"
    state: Literal["LISTENING", "THINKING", "SPEAKING"]


class WsLlmChunkMessage(BaseModel):
    """Streaming LLM text token (for live text display in UI)."""
    type: Literal["llm_chunk"] = "llm_chunk"
    token: str


class WsInterruptedMessage(BaseModel):
    """Signals that agent was interrupted — client should drop audio buffer."""
    type: Literal["interrupted"] = "interrupted"


class WsErrorMessage(BaseModel):
    """Error notification."""
    type: Literal["error"] = "error"
    code: str
    message: str


class WsTurnEndMessage(BaseModel):
    """Signals end of one agent response turn — agent is back to LISTENING."""
    type: Literal["turn_end"] = "turn_end"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_error(code: str, message: str) -> Dict[str, Any]:
    return WsErrorMessage(code=code, message=message).model_dump()


def make_agent_state(state: str) -> Dict[str, Any]:
    return WsAgentStateMessage(state=state).model_dump()


def make_transcript(text: str) -> Dict[str, Any]:
    return WsTranscriptMessage(text=text).model_dump()


def make_llm_chunk(token: str) -> Dict[str, Any]:
    return WsLlmChunkMessage(token=token).model_dump()


def make_interrupted() -> Dict[str, Any]:
    return WsInterruptedMessage().model_dump()


def make_turn_end() -> Dict[str, Any]:
    return WsTurnEndMessage().model_dump()


def make_vad_state(state: str, probability: float) -> Dict[str, Any]:
    return WsVadStateMessage(state=state, probability=probability).model_dump()
