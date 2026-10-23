"""
TalkBuddy — WebSocket endpoint for real-time bidirectional voice streaming.

Protocol:
  - Binary frames: raw PCM audio (client: 16kHz 16-bit mono; server: 24kHz 16-bit mono)
  - Text frames: JSON control messages (see ws_messages.py)

Features:
  - Live VAD-based speech detection
  - STT → LLM → TTS pipelined streaming
  - Instant barge-in interruption
  - Session management with conversation history
"""

import asyncio
import json
from typing import Optional

import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from websockets.exceptions import ConnectionClosed

from app.core.config import settings
from app.core.constants import VAD_END, VAD_FRAME_SAMPLES, VAD_START
from app.core.logging import get_logger
from app.pipeline.orchestrator import PipelineOrchestrator
from app.schemas import ws_messages
from app.services.llm_service import LLMService
from app.services.session_service import session_service
from app.services.stt_service import STTService
from app.services.tts_service import TTSService
from app.services.vad_service import VADService
from app.utils.audio_ops import float32_to_pcm16, pcm16_to_float32

logger = get_logger()
router = APIRouter()

# Services initialized by main.py
vad_service: VADService = None
stt_service: STTService = None
llm_service: LLMService = None
tts_service: TTSService = None


def init_services(vad: VADService, stt: STTService, llm: LLMService, tts: TTSService) -> None:
    """Called by main.py to inject service instances."""
    global vad_service, stt_service, llm_service, tts_service
    vad_service = vad
    stt_service = stt
    llm_service = llm
    tts_service = tts


@router.websocket("/voice/stream")
async def voice_stream(websocket: WebSocket):
    """
    Full-duplex WebSocket endpoint.

    Client sends:
      - JSON init message first
      - Binary frames: 16kHz 16-bit PCM audio (512-sample chunks = 1024 bytes each)
      - JSON interrupt message to trigger barge-in

    Server sends:
      - JSON control messages (vad_state, transcript, agent_state, llm_chunk, interrupted, turn_end)
      - Binary frames: 24kHz 16-bit PCM audio chunks
    """
    await websocket.accept()

    session_id: Optional[str] = None
    voice: str = settings.tts.voice
    orchestrator: Optional[PipelineOrchestrator] = None
    audio_sender_task: Optional[asyncio.Task] = None

    try:
        # Wait for init message
        init_raw = await websocket.receive_text()
        try:
            init_msg = ws_messages.WsInitMessage.model_validate_json(init_raw)
        except Exception as e:
            await websocket.send_json(ws_messages.make_error("INVALID_INIT", str(e)))
            await websocket.close()
            return

        # Get or create session
        session = session_service.get_or_create(init_msg.session_id)
        session_id = session.session_id

        # Override system prompt if provided
        if init_msg.system_prompt:
            session.history[0] = {"role": "system", "content": init_msg.system_prompt}

        # Override voice if provided
        if init_msg.voice:
            voice = init_msg.voice

        # Confirm session ready
        await websocket.send_json({
            "type": "session",
            "session_id": session_id,
            "voice": voice,
        })

        logger.info(f"WebSocket session {session_id} connected")

        # Create pipeline orchestrator with callbacks
        def on_state_change(state: str):
            asyncio.create_task(websocket.send_json(ws_messages.make_agent_state(state)))

        def on_vad_event(state: str, probability: float):
            asyncio.create_task(websocket.send_json(ws_messages.make_vad_state(state, probability)))

        def on_transcript(text: str):
            # Add user message to history
            session.add_message("user", text)
            asyncio.create_task(websocket.send_json(ws_messages.make_transcript(text)))

        def on_llm_chunk(token: str):
            asyncio.create_task(websocket.send_json(ws_messages.make_llm_chunk(token)))

        def on_interrupted():
            asyncio.create_task(websocket.send_json(ws_messages.make_interrupted()))

        orchestrator = PipelineOrchestrator(
            vad=vad_service,
            stt=stt_service,
            llm=llm_service,
            tts=tts_service,
            on_state_change=on_state_change,
            on_transcript=on_transcript,
            on_llm_chunk=on_llm_chunk,
            on_interrupted=on_interrupted,
            on_vad_event=on_vad_event,
            conversation_history_provider=lambda: session.get_context(),
        )

        # Start audio sender coroutine
        async def send_audio_loop():
            """Consume TTS audio from orchestrator and send as binary frames."""
            try:
                async for audio_float32 in orchestrator.audio_output_stream():
                    if audio_float32 is None:
                        # End of turn
                        await websocket.send_json(ws_messages.make_turn_end())
                        continue

                    # Convert 24kHz float32 → 16-bit PCM
                    audio_pcm16 = float32_to_pcm16(audio_float32)
                    await websocket.send_bytes(audio_pcm16)
            except (WebSocketDisconnect, ConnectionClosed):
                pass
            except Exception as e:
                logger.error(f"Audio sender error: {e}")

        audio_sender_task = asyncio.create_task(send_audio_loop())

        # Main receiver loop
        while True:
            message = await websocket.receive()

            if "text" in message:
                # JSON control message
                try:
                    msg_dict = json.loads(message["text"])
                    msg_type = msg_dict.get("type")

                    if msg_type == "ping":
                        await websocket.send_json({"type": "pong"})

                    elif msg_type == "interrupt":
                        logger.debug("Client requested interrupt")
                        orchestrator.interrupt()

                    else:
                        logger.warning(f"Unknown message type: {msg_type}")

                except json.JSONDecodeError as e:
                    await websocket.send_json(ws_messages.make_error("INVALID_JSON", str(e)))

            elif "bytes" in message:
                # Binary audio frame (16kHz 16-bit PCM)
                audio_bytes = message["bytes"]

                # Expect 512 samples × 2 bytes = 1024 bytes per frame
                if len(audio_bytes) != VAD_FRAME_SAMPLES * 2:
                    logger.warning(f"Unexpected frame size: {len(audio_bytes)} bytes (expected 1024)")
                    continue

                # Convert to float32 for VAD
                audio_pcm16 = np.frombuffer(audio_bytes, dtype=np.int16)
                audio_float32 = pcm16_to_float32(audio_pcm16)

                # Debug: log audio level
                max_amp = np.max(np.abs(audio_float32))
                if max_amp > 0.01:
                    logger.debug(f"Audio frame received, max amplitude: {max_amp:.3f}")

                # Feed to pipeline (VAD → STT → LLM → TTS)
                await orchestrator.process_audio_chunk(audio_float32)

    except WebSocketDisconnect:
        logger.info(f"WebSocket session {session_id} disconnected")

    except ConnectionClosed:
        logger.info(f"WebSocket session {session_id} connection closed")

    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        try:
            await websocket.send_json(ws_messages.make_error("SERVER_ERROR", str(e)))
        except Exception:
            pass

    finally:
        # Cleanup
        if audio_sender_task and not audio_sender_task.done():
            audio_sender_task.cancel()
            try:
                await audio_sender_task
            except asyncio.CancelledError:
                pass

        if orchestrator:
            orchestrator.interrupt()  # Stop any ongoing generation

        try:
            await websocket.close()
        except Exception:
            pass

        logger.info(f"WebSocket session {session_id} cleaned up")
