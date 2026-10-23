"""
TalkBuddy — faster-whisper transcription engine.
Runs transcription in a dedicated thread pool to avoid blocking the event loop.
"""

import asyncio
from typing import Optional

import numpy as np

from app.core.config import STTSettings
from app.core.logging import get_logger
from app.utils.executor import get_stt_executor

logger = get_logger()


class STTService:
    """
    Speech-to-Text service using faster-whisper (CTranslate2).
    Downloads the model on first use and caches it locally.
    """

    def __init__(self, settings: STTSettings):
        self.settings = settings
        self._model = None

    def initialize(self) -> None:
        """Load the faster-whisper CTranslate2 model."""
        from faster_whisper import WhisperModel

        logger.info(f"Loading STT model: {self.settings.model_size}")
        logger.info(f"Device: {self.settings.device}, Compute: {self.settings.compute_type}")

        self._model = WhisperModel(
            model_size_or_path=self.settings.model_size,
            device=self.settings.device,
            compute_type=self.settings.compute_type,
            cpu_threads=8,
            num_workers=2,
        )
        logger.info("STT model loaded successfully")

    @property
    def is_loaded(self) -> bool:
        """Check if the model is loaded."""
        return self._model is not None

    def _transcribe_sync(self, audio_data: np.ndarray) -> str:
        """
        Synchronous transcription — runs in a worker thread.

        Args:
            audio_data: Float32 numpy array at 16kHz mono

        Returns:
            Transcribed text string
        """
        if self._model is None:
            raise RuntimeError("STTService not initialized. Call initialize() first.")

        segments, info = self._model.transcribe(
            audio=audio_data,
            beam_size=self.settings.beam_size,
            language=self.settings.language,
            vad_filter=False,  # We already ran Silero VAD upstream
            condition_on_previous_text=False,
        )

        text = " ".join(segment.text for segment in segments).strip()
        logger.debug(f"Transcribed ({info.duration:.1f}s audio): {text}")
        return text

    async def transcribe(self, audio_data: np.ndarray) -> str:
        """
        Async non-blocking transcription.
        Dispatches to a dedicated thread pool executor.

        Args:
            audio_data: Float32 numpy array at 16kHz mono

        Returns:
            Transcribed text string
        """
        loop = asyncio.get_running_loop()
        executor = get_stt_executor()
        return await loop.run_in_executor(
            executor,
            self._transcribe_sync,
            audio_data,
        )

    def shutdown(self) -> None:
        """Release model resources."""
        if self._model is not None:
            logger.info("Shutting down STT service")
            self._model = None
