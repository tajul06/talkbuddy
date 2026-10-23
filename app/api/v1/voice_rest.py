"""
TalkBuddy — REST voice chat endpoint (upload audio → streamed audio response).
"""

from typing import AsyncGenerator, Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from app.core.config import settings
from app.core.logging import get_logger
from app.services.llm_service import LLMService
from app.services.session_service import session_service
from app.services.stt_service import STTService
from app.services.tts_service import TTSService
from app.pipeline.chunker import SentenceChunker
from app.pipeline.interruption import CancellationToken
from app.utils.audio_ops import (
    prepare_input_audio,
    pack_wav,
    OUTPUT_SAMPLE_RATE,
)

logger = get_logger()
router = APIRouter()

# Services will be initialized by main.py lifespan
stt_service: STTService = None
llm_service: LLMService = None
tts_service: TTSService = None


def init_services(stt: STTService, llm: LLMService, tts: TTSService) -> None:
    """Initialize services from app lifespan. Called by main.py."""
    global stt_service, llm_service, tts_service
    stt_service = stt
    llm_service = llm
    tts_service = tts


@router.post("/voice/chat", summary="Upload audio and receive streamed audio response")
async def voice_chat(
    audio: UploadFile = File(..., description="Audio file (WAV or raw PCM)"),
    session_id: Optional[str] = Form(default=None, description="Optional session ID"),
    voice: Optional[str] = Form(default=None, description="Optional Kokoro voice override"),
) -> StreamingResponse:
    """
    Voice chat endpoint:
    1. Upload user audio (WAV format)
    2. Transcribe via faster-whisper
    3. Generate response via local LLM (streaming)
    4. Synthesize to audio via Kokoro-82M (sentence by sentence)
    5. Stream back audio chunks as WAV data
    """
    if stt_service is None or not stt_service.is_loaded:
        raise HTTPException(status_code=503, detail="STT service not initialized")
    if llm_service is None or not llm_service.is_loaded:
        raise HTTPException(status_code=503, detail="LLM service not initialized")
    if tts_service is None or not tts_service.is_loaded:
        raise HTTPException(status_code=503, detail="TTS service not initialized")

    # Read uploaded audio
    try:
        audio_bytes = await audio.read()
        audio_data = prepare_input_audio(audio_bytes)
    except Exception as e:
        logger.error(f"Audio processing error: {e}")
        raise HTTPException(status_code=400, detail="Invalid audio file")

    # Get or create session
    session = session_service.get_or_create(session_id)
    cancellation_token = session.coordinator.get_current_token()

    # Transcribe user audio
    try:
        user_transcript = await stt_service.transcribe(audio_data)
        logger.info(f"User transcript: {user_transcript}")
    except Exception as e:
        logger.error(f"STT transcription error: {e}")
        raise HTTPException(status_code=500, detail="Transcription failed")

    if not user_transcript:
        raise HTTPException(status_code=400, detail="No speech detected in audio")

    # Update conversation history
    session.add_message("user", user_transcript)
    messages = session.get_context()

    async def generate_response_audio() -> AsyncGenerator[bytes, None]:
        """Stream synthesized audio chunks as WAV bytes."""
        chunker = SentenceChunker()
        full_response = []

        try:
            async for token in llm_service.generate_stream(messages, cancellation_token):
                if cancellation_token.is_cancelled():
                    return

                full_response.append(token)

                # Synthesize any completed sentences
                sentences = chunker.add_token(token)
                for sentence in sentences:
                    if cancellation_token.is_cancelled():
                        return
                    try:
                        audio_chunk = await tts_service.synthesize(sentence, voice=voice)
                        if len(audio_chunk) > 0:
                            yield pack_wav(audio_chunk, OUTPUT_SAMPLE_RATE)
                    except Exception as e:
                        logger.error(f"TTS error: {e}")

            # Flush remaining text
            for sentence in chunker.flush():
                if cancellation_token.is_cancelled():
                    return
                try:
                    audio_chunk = await tts_service.synthesize(sentence, voice=voice)
                    if len(audio_chunk) > 0:
                        yield pack_wav(audio_chunk, OUTPUT_SAMPLE_RATE)
                except Exception as e:
                    logger.error(f"TTS error: {e}")

        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            return

        # Save assistant response to session
        full_text = "".join(full_response)
        session.add_message("assistant", full_text)

    return StreamingResponse(
        generate_response_audio(),
        media_type="audio/wav",
        headers={
            "X-Transcript": user_transcript,
            "X-Session-ID": session.session_id,
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
