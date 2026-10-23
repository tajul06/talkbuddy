"""
TalkBuddy — Kokoro-82M text-to-speech synthesis service.
Synthesizes text to 24kHz audio, runs in a dedicated thread pool.
"""

import asyncio
from typing import Optional

import numpy as np

from app.core.config import TTSSettings
from app.core.logging import get_logger
from app.utils.executor import get_tts_executor

logger = get_logger()


class TTSService:
    """
    Text-to-Speech service using Kokoro-82M.
    Generates natural-sounding 24kHz float32 audio from text.
    """

    def __init__(self, settings: TTSSettings):
        self.settings = settings
        self._pipeline = None

    def initialize(self) -> None:
        """Load the Kokoro-82M model and voice pipeline."""
        from kokoro import KPipeline

        logger.info(f"Loading TTS pipeline: {self.settings.model_name}")
        logger.info(f"Voice: {self.settings.voice}, Lang: {self.settings.lang_code}")

        self._pipeline = KPipeline(lang_code=self.settings.lang_code)
        logger.info("TTS pipeline loaded successfully")

    @property
    def is_loaded(self) -> bool:
        """Check if the pipeline is loaded."""
        return self._pipeline is not None

    def _synthesize_sync(self, text: str, voice: Optional[str] = None) -> np.ndarray:
        """
        Synchronous synthesis — runs in a worker thread.

        Args:
            text: Text to synthesize
            voice: Kokoro voice ID override (uses default if None)

        Returns:
            Float32 numpy array at 24kHz
        """
        if self._pipeline is None:
            raise RuntimeError("TTSService not initialized. Call initialize() first.")

        voice = voice or self.settings.voice
        audio_chunks = []

        generator = self._pipeline(
            text,
            voice=voice,
            speed=self.settings.speed,
            split_pattern=r"\n+",
        )

        for _, _, audio in generator:
            if audio is not None:
                audio_chunks.append(audio)

        if not audio_chunks:
            logger.warning(f"TTS produced no audio for text: {text[:50]}")
            return np.array([], dtype=np.float32)

        result = np.concatenate(audio_chunks)
        logger.debug(f"TTS synthesized {len(result)/24000:.2f}s audio for: {text[:50]}")
        return result

    async def synthesize(self, text: str, voice: Optional[str] = None) -> np.ndarray:
        """
        Async non-blocking synthesis.
        Dispatches to a dedicated thread pool executor.

        Args:
            text: Text to synthesize
            voice: Kokoro voice ID override

        Returns:
            Float32 numpy array at 24kHz
        """
        loop = asyncio.get_running_loop()
        executor = get_tts_executor()
        return await loop.run_in_executor(
            executor,
            self._synthesize_sync,
            text,
            voice,
        )

    def shutdown(self) -> None:
        """Release pipeline resources."""
        if self._pipeline is not None:
            logger.info("Shutting down TTS service")
            self._pipeline = None
