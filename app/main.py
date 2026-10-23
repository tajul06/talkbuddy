"""
TalkBuddy — FastAPI application factory.
Handles app lifespan, service initialization, CORS, and router registration.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.core.logging import setup_logging, get_logger
from app.api.v1 import health, text_rest, voice_rest, sessions, config, websocket
from app.services.vad_service import VADService
from app.services.stt_service import STTService
from app.services.whisper_cpp_service import WhisperCppService
from app.services.llm_service import LLMService
from app.services.tts_service import TTSService
from app.utils.executor import shutdown_executors

logger = get_logger()

# Global service instances
vad_service: VADService = None
stt_service: STTService = None
llm_service: LLMService = None
tts_service: TTSService = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    Handles startup and shutdown tasks.
    """
    global vad_service, stt_service, llm_service, tts_service

    # Startup
    setup_logging(settings.server.log_level)
    logger.info("TalkBuddy starting up...")
    logger.info(f"LLM server URL: {settings.llm.api_url}")
    logger.info(f"Server listening on {settings.server.host}:{settings.server.port}")

    # Initialize AI services
    logger.info("Initializing AI services...")

    vad_service = VADService(settings.vad)
    vad_service.initialize()

    # Choose STT backend
    if settings.stt.backend == "whisper.cpp":
        logger.info("Using whisper.cpp STT backend (GPU-accelerated)")
        stt_service = WhisperCppService(settings.stt)
        stt_service.initialize()
    else:
        logger.info("Using faster-whisper STT backend (CPU)")
        stt_service = STTService(settings.stt)
        stt_service.initialize()

    llm_service = LLMService(settings.llm)
    llm_service.initialize()

    tts_service = TTSService(settings.tts)
    tts_service.initialize()

    # Pass services to endpoint modules
    text_rest.init_services(llm_service, tts_service)
    voice_rest.init_services(stt_service, llm_service, tts_service)
    websocket.init_services(vad_service, stt_service, llm_service, tts_service)

    logger.info("All services initialized successfully")

    yield

    # Shutdown
    logger.info("TalkBuddy shutting down...")

    if vad_service:
        vad_service.shutdown()
    if stt_service:
        stt_service.shutdown()
    if llm_service:
        await llm_service.shutdown()
    if tts_service:
        tts_service.shutdown()

    shutdown_executors()
    logger.info("All services shut down")


def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.

    Returns:
        FastAPI: Configured FastAPI application instance
    """
    app = FastAPI(
        title="TalkBuddy",
        description="Real-time conversational voice agent with local AI models",
        version="0.1.0",
        lifespan=lifespan,
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.server.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register routers
    app.include_router(health.router, prefix="/api/v1", tags=["health"])
    app.include_router(text_rest.router, prefix="/api/v1", tags=["text"])
    app.include_router(voice_rest.router, prefix="/api/v1", tags=["voice"])
    app.include_router(websocket.router, prefix="/api/v1", tags=["websocket"])
    app.include_router(sessions.router, prefix="/api/v1", tags=["sessions"])
    app.include_router(config.router, prefix="/api/v1", tags=["config"])

    return app


# Application instance
app = create_app()
