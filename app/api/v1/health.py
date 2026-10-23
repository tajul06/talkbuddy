"""
TalkBuddy — Health check endpoint with GPU and model info.
Pings the external llama.cpp server to verify LLM connectivity.
"""

import httpx
from fastapi import APIRouter
from pydantic import BaseModel

from app.core.config import settings

router = APIRouter()


class ModelStatus(BaseModel):
    vad: str
    stt: str
    llm: str
    tts: str


class HealthResponse(BaseModel):
    status: str
    version: str
    gpu_backend: str
    models: ModelStatus
    llm_server_reachable: bool
    active_sessions: int


@router.get("/health", response_model=HealthResponse, summary="System health and model status")
async def health_check() -> HealthResponse:
    """
    Returns system health, GPU backend info, and model availability.
    Pings the external llama.cpp server at LLM__API_URL to check connectivity.
    """
    # Ping llama.cpp server /health
    llm_reachable = False
    llm_label = f"external @ {settings.llm.api_url}"
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            resp = await client.get(f"{settings.llm.api_url}/health")
            llm_reachable = resp.status_code == 200
            if llm_reachable:
                llm_label = f"llama.cpp @ {settings.llm.api_url} (OK)"
    except Exception:
        llm_label = f"llama.cpp @ {settings.llm.api_url} (UNREACHABLE)"

    # Import session count lazily to avoid circular imports
    active_sessions = 0
    try:
        from app.services.session_service import session_service
        active_sessions = session_service.count()
    except Exception:
        pass

    return HealthResponse(
        status="healthy" if llm_reachable else "degraded",
        version="0.1.0",
        gpu_backend="vulkan",
        models=ModelStatus(
            vad=settings.vad.model_name,
            stt=settings.stt.model_size,
            llm=llm_label,
            tts=settings.tts.model_name,
        ),
        llm_server_reachable=llm_reachable,
        active_sessions=active_sessions,
    )
