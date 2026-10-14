"""
TalkBuddy — Health check endpoint with GPU and model info.
"""

from pathlib import Path
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
    llm_model_found: bool
    active_sessions: int


@router.get("/health", response_model=HealthResponse, summary="System health and model status")
async def health_check() -> HealthResponse:
    """
    Returns system health, GPU backend info, and model availability.
    Useful for verifying the server is up and all models are configured.
    """
    # Check if the GGUF model file exists at the configured path
    llm_model_path = Path(settings.llm.model_path)
    llm_model_found = llm_model_path.exists()

    # Import session count lazily to avoid circular imports
    active_sessions = 0
    try:
        from app.services.session_service import session_service
        active_sessions = session_service.count()
    except Exception:
        pass

    return HealthResponse(
        status="healthy",
        version="0.1.0",
        gpu_backend="vulkan",
        models=ModelStatus(
            vad=settings.vad.model_name,
            stt=settings.stt.model_size,
            llm=llm_model_path.name if llm_model_found else f"NOT FOUND ({settings.llm.model_path})",
            tts=settings.tts.model_name,
        ),
        llm_model_found=llm_model_found,
        active_sessions=active_sessions,
    )
