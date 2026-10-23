"""
TalkBuddy — Runtime configuration inspection and update endpoints.
"""

from typing import Any, Dict

from fastapi import APIRouter

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger()
router = APIRouter()


@router.get("/config", response_model=Dict[str, Any], summary="Get current configuration")
async def get_config() -> Dict[str, Any]:
    """
    Get the current runtime configuration.
    Returns all settings grouped by section.
    """
    return {
        "audio": {
            "input_sample_rate": settings.audio.input_sample_rate,
            "output_sample_rate": settings.audio.output_sample_rate,
            "channels": settings.audio.channels,
            "sample_width": settings.audio.sample_width,
        },
        "vad": {
            "model_name": settings.vad.model_name,
            "threshold": settings.vad.threshold,
            "min_speech_duration_ms": settings.vad.min_speech_duration_ms,
            "min_silence_duration_ms": settings.vad.min_silence_duration_ms,
        },
        "stt": {
            "model_size": settings.stt.model_size,
            "device": settings.stt.device,
            "compute_type": settings.stt.compute_type,
            "language": settings.stt.language,
        },
        "llm": {
            "api_url": settings.llm.api_url,
            "temperature": settings.llm.temperature,
            "top_p": settings.llm.top_p,
            "max_tokens": settings.llm.max_tokens,
        },
        "tts": {
            "model_name": settings.tts.model_name,
            "voice": settings.tts.voice,
            "speed": settings.tts.speed,
            "lang_code": settings.tts.lang_code,
        },
        "server": {
            "host": settings.server.host,
            "port": settings.server.port,
            "log_level": settings.server.log_level,
        },
    }


@router.get("/config/llm", response_model=Dict[str, Any], summary="Get LLM configuration")
async def get_llm_config() -> Dict[str, Any]:
    """Get LLM-specific configuration."""
    return {
        "api_url": settings.llm.api_url,
        "temperature": settings.llm.temperature,
        "top_p": settings.llm.top_p,
        "max_tokens": settings.llm.max_tokens,
    }


@router.get("/config/tts", response_model=Dict[str, Any], summary="Get TTS configuration")
async def get_tts_config() -> Dict[str, Any]:
    """Get TTS-specific configuration."""
    return {
        "model_name": settings.tts.model_name,
        "voice": settings.tts.voice,
        "speed": settings.tts.speed,
        "lang_code": settings.tts.lang_code,
    }


@router.get("/config/vad", response_model=Dict[str, Any], summary="Get VAD configuration")
async def get_vad_config() -> Dict[str, Any]:
    """Get VAD-specific configuration."""
    return {
        "model_name": settings.vad.model_name,
        "threshold": settings.vad.threshold,
        "min_speech_duration_ms": settings.vad.min_speech_duration_ms,
        "min_silence_duration_ms": settings.vad.min_silence_duration_ms,
    }
