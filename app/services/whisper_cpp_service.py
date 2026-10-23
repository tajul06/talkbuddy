"""
TalkBuddy — whisper.cpp HTTP client for GPU-accelerated STT.
Uses external whisper-server running with Vulkan support.
"""

import asyncio
import io
from typing import Optional

import httpx
import numpy as np
import soundfile as sf

from app.core.config import STTSettings
from app.core.logging import get_logger

logger = get_logger()


class WhisperCppService:
    """
    Speech-to-Text service using external whisper.cpp server with Vulkan GPU acceleration.
    Sends audio to whisper-server via HTTP and receives transcription.
    """

    def __init__(self, settings: STTSettings):
        self.settings = settings
        # whisper.cpp server URL
        self.server_url = "http://localhost:8081"
        self._client: Optional[httpx.AsyncClient] = None

    def initialize(self) -> None:
        """Initialize HTTP client for whisper-server."""
        logger.info(f"Initializing whisper.cpp STT client")
        logger.info(f"Server: {self.server_url}")
        logger.info(f"Language: {self.settings.language}")

        self._client = httpx.AsyncClient(timeout=30.0)
        logger.info("Whisper.cpp STT client initialized")

    @property
    def is_loaded(self) -> bool:
        """Check if the client is initialized."""
        return self._client is not None

    async def transcribe(self, audio_data: np.ndarray) -> str:
        """
        Async transcription via whisper-server HTTP API.

        Args:
            audio_data: Float32 numpy array at 16kHz mono

        Returns:
            Transcribed text string
        """
        if not self.is_loaded:
            raise RuntimeError("WhisperCppService not initialized. Call initialize() first.")

        try:
            # Convert numpy array to WAV bytes
            wav_buffer = io.BytesIO()
            sf.write(wav_buffer, audio_data, 16000, format='WAV', subtype='PCM_16')
            wav_buffer.seek(0)
            wav_bytes = wav_buffer.read()

            # Send to whisper-server
            files = {
                'file': ('audio.wav', wav_bytes, 'audio/wav')
            }
            data = {
                'temperature': '0.0',
                'temperature_inc': '0.2',
                'response_format': 'json'
            }

            response = await self._client.post(
                f"{self.server_url}/inference",
                files=files,
                data=data
            )
            response.raise_for_status()

            result = response.json()
            text = result.get('text', '').strip()

            logger.debug(f"Transcribed: {text}")
            return text

        except httpx.ConnectError:
            logger.error(
                f"Cannot connect to whisper-server at {self.server_url}. "
                "Start it with: whisper-server.exe -m <model.bin> --host 0.0.0.0 --port 8081 -fa"
            )
            raise
        except httpx.HTTPStatusError as e:
            logger.error(f"Whisper-server returned error: {e.response.status_code}")
            raise
        except Exception as e:
            logger.error(f"STT transcription error: {e}")
            raise

    async def shutdown(self) -> None:
        """Release HTTP client resources."""
        if self._client is not None:
            logger.info("Shutting down whisper.cpp STT service")
            await self._client.aclose()
            self._client = None
