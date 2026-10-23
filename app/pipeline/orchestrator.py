"""
TalkBuddy — Async queue pipeline orchestrator.
Wires VAD → STT → LLM → Chunker → TTS stages together using asyncio.Queue.
Each stage runs concurrently: TTS starts on Sentence 1 while LLM generates Sentence 2.
"""

import asyncio
from typing import AsyncGenerator, Callable, Dict, List, Optional

import numpy as np

from app.core.constants import (
    VAD_END,
    VAD_FRAME_SAMPLES,
    VAD_START,
    STATE_LISTENING,
    STATE_SPEAKING,
    STATE_THINKING,
)
from app.core.logging import get_logger
from app.pipeline.chunker import SentenceChunker
from app.pipeline.interruption import CancellationToken, InterruptionCoordinator
from app.services.llm_service import LLMService
from app.services.stt_service import STTService
from app.services.tts_service import TTSService
from app.services.vad_service import VADService

logger = get_logger()


class PipelineOrchestrator:
    """
    Manages the full voice pipeline for a single session.
    Coordinates VAD → STT → LLM → TTS and handles barge-in interruption.
    """

    def __init__(
        self,
        vad: VADService,
        stt: STTService,
        llm: LLMService,
        tts: TTSService,
        on_state_change: Optional[Callable[[str], None]] = None,
        on_transcript: Optional[Callable[[str], None]] = None,
        on_llm_chunk: Optional[Callable[[str], None]] = None,
        on_interrupted: Optional[Callable[[], None]] = None,
        on_vad_event: Optional[Callable[[str, float], None]] = None,
        conversation_history_provider: Optional[Callable[[], List[Dict[str, str]]]] = None,
    ):
        self.vad = vad
        self.stt = stt
        self.llm = llm
        self.tts = tts

        # Callbacks for WebSocket/REST event propagation
        self.on_state_change = on_state_change
        self.on_transcript = on_transcript
        self.on_llm_chunk = on_llm_chunk
        self.on_interrupted = on_interrupted
        self.on_vad_event = on_vad_event
        self.conversation_history_provider = conversation_history_provider

        # Pipeline state
        self._state = STATE_LISTENING
        self._speech_buffer: List[np.ndarray] = []
        self._coordinator = InterruptionCoordinator()
        self._audio_out_queue: asyncio.Queue[Optional[np.ndarray]] = asyncio.Queue()

    def _set_state(self, new_state: str) -> None:
        """Update agent state and notify via callback."""
        if self._state != new_state:
            self._state = new_state
            logger.debug(f"Agent state: {new_state}")
            if self.on_state_change:
                self.on_state_change(new_state)

    def interrupt(self) -> None:
        """
        Trigger a barge-in interrupt.
        Cancels current generation and purges the audio output queue.
        """
        logger.info("Barge-in detected — interrupting pipeline")
        self._coordinator.interrupt()
        self._drain_audio_queue()
        if self.on_interrupted:
            self.on_interrupted()
        self._set_state(STATE_LISTENING)
        self.vad.reset_state()

    def _drain_audio_queue(self) -> None:
        """Empty the audio output queue immediately."""
        while not self._audio_out_queue.empty():
            try:
                self._audio_out_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

    async def process_audio_chunk(self, audio_chunk: np.ndarray) -> None:
        """
        Feed a 512-sample PCM chunk through the VAD stage.
        Accumulates speech into a buffer and triggers the STT→LLM→TTS
        pipeline when a complete utterance is detected.

        Args:
            audio_chunk: Float32 numpy array, 512 samples at 16kHz
        """
        changed, state, prob = self.vad.process_and_detect(audio_chunk)

        if state in (VAD_START, "SPEAKING"):
            # Buffer ongoing speech audio
            self._speech_buffer.append(audio_chunk)

            if state == VAD_START:
                # Barge-in: interrupt if agent is currently speaking
                if self._state == STATE_SPEAKING:
                    self.interrupt()
                self._set_state(STATE_LISTENING)
                if self.on_vad_event:
                    self.on_vad_event("speech_start", prob)

        elif state == VAD_END and self._speech_buffer:
            # Full utterance collected — run STT → LLM → TTS
            if self.on_vad_event:
                self.on_vad_event("speech_end", 0.0)
            utterance = np.concatenate(self._speech_buffer)
            self._speech_buffer = []
            self._set_state(STATE_THINKING)
            asyncio.create_task(self._run_inference_pipeline(utterance))

    async def _run_inference_pipeline(
        self,
        utterance: np.ndarray,
        conversation_history: Optional[List[Dict[str, str]]] = None,
    ) -> None:
        """
        Run the full STT → LLM → TTS pipeline for one utterance.
        Each stage is pipelined: TTS starts as soon as the first sentence is ready.

        Args:
            utterance: Float32 audio array of the full user utterance
            conversation_history: LLM conversation messages (with system prompt)
        """
        import time
        pipeline_start = time.perf_counter()

        token = self._coordinator.get_current_token()

        # --- Stage 1: STT ---
        stt_start = time.perf_counter()
        try:
            user_text = await self.stt.transcribe(utterance)
        except Exception as e:
            logger.error(f"STT error: {e}")
            self._set_state(STATE_LISTENING)
            return
        stt_duration = time.perf_counter() - stt_start
        logger.info(f"⏱️ STT took {stt_duration*1000:.0f}ms")

        if not user_text:
            logger.debug("STT returned empty transcript, skipping")
            self._set_state(STATE_LISTENING)
            return

        if self.on_transcript:
            self.on_transcript(user_text)

        # --- Stage 2: LLM streaming → Sentence Chunking → TTS ---
        messages = conversation_history or (self.conversation_history_provider() if self.conversation_history_provider else [])
        chunker = SentenceChunker()
        self._set_state(STATE_SPEAKING)

        llm_first_token_time = None
        tts_first_audio_time = None
        token_count = 0

        try:
            async for llm_token in self.llm.generate_stream(messages, token):
                if token.is_cancelled():
                    break

                token_count += 1
                if llm_first_token_time is None:
                    llm_first_token_time = time.perf_counter()
                    logger.info(f"⏱️ LLM first token took {(llm_first_token_time - stt_start)*1000:.0f}ms")

                # Notify subscribers of each text chunk (e.g. for WebSocket)
                if self.on_llm_chunk:
                    self.on_llm_chunk(llm_token)

                # Feed token to chunker and synthesize completed sentences
                sentences = chunker.add_token(llm_token)
                for sentence in sentences:
                    if token.is_cancelled():
                        break
                    tts_start = time.perf_counter()
                    await self._synthesize_and_enqueue(sentence, token)
                    if tts_first_audio_time is None:
                        tts_first_audio_time = time.perf_counter()
                        ttfa = (tts_first_audio_time - pipeline_start) * 1000
                        logger.info(f"🎯 TTFA (Time-To-First-Audio): {ttfa:.0f}ms")

            # Flush any remaining text in the buffer
            if not token.is_cancelled():
                for sentence in chunker.flush():
                    await self._synthesize_and_enqueue(sentence, token)

            total_duration = time.perf_counter() - pipeline_start
            logger.info(f"⏱️ Total pipeline: {total_duration*1000:.0f}ms, tokens: {token_count}")

        except Exception as e:
            logger.error(f"LLM/TTS pipeline error: {e}")
        finally:
            # Signal end of audio stream
            await self._audio_out_queue.put(None)
            if not token.is_cancelled():
                self._set_state(STATE_LISTENING)

    async def _synthesize_and_enqueue(
        self, sentence: str, token: CancellationToken
    ) -> None:
        """
        Synthesize a sentence to audio and place it in the output queue.

        Args:
            sentence: Text sentence to synthesize
            token: Cancellation token to abort early
        """
        if token.is_cancelled():
            return
        try:
            audio = await self.tts.synthesize(sentence)
            if not token.is_cancelled() and len(audio) > 0:
                await self._audio_out_queue.put(audio)
        except Exception as e:
            logger.error(f"TTS synthesis error for '{sentence[:40]}': {e}")

    async def audio_output_stream(self) -> AsyncGenerator[np.ndarray, None]:
        """
        Async generator yielding synthesized audio chunks as they become ready.
        Yields None to signal end of current turn, then continues for next turn.
        """
        while True:
            chunk = await self._audio_out_queue.get()
            yield chunk
            # Don't break on None - yield it and continue for next turn
