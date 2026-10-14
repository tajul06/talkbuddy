"""
TalkBuddy — FastAPI application factory.
Handles app lifespan, service initialization, CORS, and router registration.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.core.logging import setup_logging, get_logger
from app.api.v1 import health

logger = get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    Handles startup and shutdown tasks.
    """
    # Startup
    setup_logging(settings.server.log_level)
    logger.info("TalkBuddy starting up...")
    logger.info(f"LLM model path: {settings.llm.model_path}")
    logger.info(f"Server listening on {settings.server.host}:{settings.server.port}")

    yield

    # Shutdown
    logger.info("TalkBuddy shutting down...")


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

    return app


# Application instance
app = create_app()
