"""
TalkBuddy — REST text chat endpoint (text in → text/audio out).
"""

from typing import AsyncGenerator

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from app.core.config import settings
from app.core.logging import get_logger
from app.schemas.chat import TextChatRequest, TextChatResponse
from app.services.llm_service import LLMService
from app.services.session_service import session_service
from app.services.tts_service import TTSService
from app.pipeline.chunker import SentenceChunker
from app.pipeline.interruption import CancellationToken
from app.utils.audio_ops import pack_wav, OUTPUT_SAMPLE_RATE

logger = get_logger()
router = APIRouter()

# Services will be initialized by main.py lifespan
llm_service: LLMService = None
tts_service: TTSService = None


def init_services(llm: LLMService, tts: TTSService) -> None:
    """Initialize services from app lifespan. Called by main.py."""
    global llm_service, tts_service
    llm_service = llm
    tts_service = tts


@router.post("/text/chat", response_model=TextChatResponse, summary="Text chat with optional audio")
async def text_chat(request: TextChatRequest) -> TextChatResponse:
    """
    Send a text message and receive a text response.
    Optionally include TTS audio in the response (stream_audio=true).
    """
    if llm_service is None or not llm_service.is_loaded:
        raise HTTPException(status_code=503, detail="LLM service not initialized")

    # Get or create session
    session = session_service.get_or_create(request.session_id)
    session.add_message("user", request.message)

    # Get conversation context
    messages = session.get_context()

    # Generate response
    try:
        response_text = await llm_service.generate(messages)
    except Exception as e:
        logger.error(f"LLM generation error: {e}")
        raise HTTPException(status_code=500, detail="LLM generation failed")

    # Add assistant response to history
    session.add_message("assistant", response_text)

    return TextChatResponse(
        session_id=session.session_id,
        user_message=request.message,
        assistant_message=response_text,
        audio_included=request.stream_audio,
    )


@router.post("/text/chat/stream", summary="Text chat with streaming audio response")
async def text_chat_stream(request: TextChatRequest) -> StreamingResponse:
    """
    Send a text message and receive streaming audio response.
    Returns audio chunks as they're synthesized (WAV format).
    """
    if llm_service is None or not llm_service.is_loaded:
        raise HTTPException(status_code=503, detail="LLM service not initialized")
    if tts_service is None or not tts_service.is_loaded:
        raise HTTPException(status_code=503, detail="TTS service not initialized")

    # Get or create session
    session = session_service.get_or_create(request.session_id)
    session.add_message("user", request.message)

    messages = session.get_context()
    cancellation_token = session.coordinator.get_current_token()

    async def generate_audio_stream() -> AsyncGenerator[bytes, None]:
        """Stream audio chunks as WAV data."""
        chunker = SentenceChunker()
        full_response = []

        try:
            async for token in llm_service.generate_stream(messages, cancellation_token):
                full_response.append(token)

                # Get completed sentences
                sentences = chunker.add_token(token)
                for sentence in sentences:
                    if cancellation_token.is_cancelled():
                        return
                    try:
                        audio = await tts_service.synthesize(sentence, voice=request.voice)
                        if len(audio) > 0:
                            yield pack_wav(audio, OUTPUT_SAMPLE_RATE)
                    except Exception as e:
                        logger.error(f"TTS error for '{sentence[:30]}': {e}")

            # Flush remaining text
            for sentence in chunker.flush():
                if cancellation_token.is_cancelled():
                    return
                try:
                    audio = await tts_service.synthesize(sentence, voice=request.voice)
                    if len(audio) > 0:
                        yield pack_wav(audio, OUTPUT_SAMPLE_RATE)
                except Exception as e:
                    logger.error(f"TTS error for '{sentence[:30]}': {e}")

        except Exception as e:
            logger.error(f"Streaming error: {e}")
            return

        # Save full response to session history
        full_text = "".join(full_response)
        session.add_message("assistant", full_text)

    return StreamingResponse(
        generate_audio_stream(),
        media_type="audio/wav",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
